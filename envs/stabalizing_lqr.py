"""
Swing up stabalizer using LQR. Details about deriving the equaton of motion can be found in the /doc subdir of the repo
"""

import numpy as np
import scipy
from typing import Optional


class LinearQuadraticRegulator:
    def __init__(
        self,
        m_cart: float = 1.0,
        m_pole: float = 0.25,
        ell: float = 0.6,
        g: float = 9.81,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
    ):
        self.m_cart = m_cart
        self.m_pole = m_pole
        self.ell = ell
        self.g = g

        # The writeup models a massless rod with a point mass at the end.
        # The point-mass inertia is included in J below, so there is no extra
        # pole inertia term to add here.
        self.Inertia = 0.0

        self.A, self.B = self.linearize()

        if Q is not None:
            self.Q = Q
        else:
            self.Q = np.diag([1.0, 1.0, 100.0, 100.0, 1.0, 1.0, 1.0, 1.0])

        if R is not None:
            self.R = R
        else:
            self.R = np.diag([1.0, 1.0])

        self.K = self.solve()

    def linearize(self) -> tuple[np.ndarray, np.ndarray]:
        """
        State: [x, y, θ_x, θ_y, ẋ, ẏ, θ̇_x, θ̇_y]
        """

        J = self.Inertia + self.m_pole * self.ell**2
        D = (self.m_cart + self.m_pole) * J - (self.m_pole * self.ell) ** 2

        p = J / D
        q = (self.g * (self.ell**2) * (self.m_pole**2)) / D
        r = (self.m_pole * self.ell) / D
        s = ((self.m_cart + self.m_pole) * (self.m_pole * self.ell * self.g)) / D

        A = np.zeros((8, 8))

        A[0, 4] = 1
        A[1, 5] = 1
        A[2, 6] = 1
        A[3, 7] = 1
        A[4, 2] = -q
        A[5, 3] = -q
        A[6, 2] = s
        A[7, 3] = s

        B = np.zeros((8, 2))

        B[4, 0] = p
        B[5, 1] = p
        B[6, 0] = -r
        B[7, 1] = -r

        return A, B

    def solve(self):
        # K = scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R) # for continious time
        P = scipy.linalg.solve_discrete_are(self.A, self.B, self.Q, self.R)
        K = np.linalg.inv(self.R) @ self.B.T @ P
        return K

    def control(self, cur_state: np.ndarray):
        """
        current_state will be obtained via backend (either mujoco or gymnaisum)
        """
        return -self.K @ cur_state


class BangBang:
    # 0.35 radians ≈ 20 degrees
    # 0.25 radians ≈ 15 degree
    # 0.175 radians ≈ 10 degree
    def __init__(self, theta_threshold: float = 0.35):
        self.theta_threshold = theta_threshold

    def control(self, cur_state: np.ndarray):
        """
        A Bang-Bang style controller for the swing up.
        """
        theta_x, theta_y = cur_state[2], cur_state[3]
        F_x, F_y = 0.0, 0.0

        if theta_x > self.theta_threshold:
            F_x = 1.0
        elif theta_x < -self.theta_threshold:
            F_x = -1.0

        if theta_y > self.theta_threshold:
            F_y = -1.0
        elif theta_y < -self.theta_threshold:
            F_y = 1.0

        return np.array([F_x, F_y])


class EnergySwingUp:
    """
    Energy-based swing-up controller (Astrom & Furuta, 2000): https://www.sciencedirect.com/science/article/abs/pii/S0005109899001405

    Pumps mechanical energy into the pendulum until it reaches the energy of
    the upright equilibrium, at which point LQR takes over for stabilization.

    The control law is:  F = k * ΔE * cos(θ) * θ̇

    where ΔE = E - E* is the energy error relative to upright.
    This is the continuous (non-switching) form — force is proportional to how
    fast the pendulum's energy is changing, so it tapers naturally to zero at
    the top and bottom of each swing instead of chattering between ±1.

    The two axes are fully decoupled so the law is applied independently to
    (θ_x, θ̇_x, F_x) and (θ_y, θ̇_y, F_y).
    """

    def __init__(
        self,
        m_pole: float = 0.25,
        ell: float = 0.6,
        g: float = -9.81,
        k: float = 2.0,
    ):
        self.m_pole = m_pole
        self.ell = ell
        self.g = g
        self.k = k
        # Rotational inertia of point mass about pivot: J = m*l²
        self.J = m_pole * ell**2

    def _delta_energy(self, theta: float, theta_dot: float) -> float:
        """Energy error relative to upright equilibrium (E* = m*g*l)."""
        E = 0.5 * self.J * theta_dot**2 + self.m_pole * self.g * self.ell * np.cos(
            theta
        )
        E_star = self.m_pole * self.g * self.ell
        return E - E_star

    def control(self, cur_state: np.ndarray) -> np.ndarray:
        _, _, theta_x, theta_y, _, _, theta_x_dot, theta_y_dot = cur_state

        dE_x = self._delta_energy(theta_x, theta_x_dot)
        dE_y = self._delta_energy(theta_y, theta_y_dot)

        F_x = self.k * dE_x * np.cos(theta_x) * theta_x_dot
        F_y = self.k * dE_y * np.cos(theta_y) * theta_y_dot

        return np.clip(np.array([F_x, F_y]), -1.0, 1.0)


if __name__ == "__main__":
    import mujoco
    import mujoco.viewer
    import time
    from numpy import random
    from mujoco_interface import TwoAxisInvertedPendulum

    sys = TwoAxisInvertedPendulum()

    lqr_threshold = 0.25  # radians (~15 degrees)
    ESUcntr = EnergySwingUp()
    LQRcntr = LinearQuadraticRegulator()

    rng = random.default_rng(42)
    sys.reset(rng)

    with mujoco.viewer.launch_passive(sys.model, sys.data) as viewer:
        while viewer.is_running():
            x = sys.get_obs()
            theta_x, theta_y = x[2], x[3]

            if abs(theta_x) < lqr_threshold and abs(theta_y) < lqr_threshold:
                u = LQRcntr.control(x)
                mode = "LQR"
            else:
                u = ESUcntr.control(x)
                mode = "SwingUp"

            print(
                f"[{mode:<7}] θx: {theta_x:+.2f}  θy: {theta_y:+.2f}  u = {np.round(u, 2)}"
            )

            sys.control(u)

            viewer.sync()
            time.sleep(0.01)
