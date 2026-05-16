"""Energy-based swing-up controller for the two-axis cart-pole."""

import numpy as np


class EnergySwingUp:
    """
    Energy-based swing-up controller (Astrom & Furuta, 2000):
    https://www.sciencedirect.com/science/article/abs/pii/S0005109899001405

    Pumps mechanical energy into the pendulum until it reaches the energy of
    the upright equilibrium, at which point a stabilizing controller such as LQR
    can take over.

    The control law is:  F = k * ΔE * cos(θ) * θ̇

    The two axes are treated independently, so the law is applied to
    (θ_x, θ̇_x, F_x) and (θ_y, θ̇_y, F_y).
    """

    def __init__(
        self,
        m_pole: float = 0.25,
        ell: float = 0.6,
        g: float = 9.81,
        k: float = 2.0,
        position_gain: float = 0.0,
        velocity_gain: float = 0.0,
        mujoco_y_axis: bool = False,
        control_limit: float = 1.0,
    ):
        self.m_pole = m_pole
        self.ell = ell
        self.g = g
        self.k = k
        self.position_gain = position_gain
        self.velocity_gain = velocity_gain
        self.mujoco_y_axis = mujoco_y_axis
        self.control_limit = control_limit
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
        x, y, theta_x, theta_y, x_dot, y_dot, theta_x_dot, theta_y_dot = np.asarray(
            cur_state, dtype=float
        )

        theta_x = (theta_x + np.pi) % (2.0 * np.pi) - np.pi
        theta_y = (theta_y + np.pi) % (2.0 * np.pi) - np.pi

        if self.mujoco_y_axis:
            theta_y *= -1.0
            theta_y_dot *= -1.0

        dE_x = self._delta_energy(theta_x, theta_x_dot)
        dE_y = self._delta_energy(theta_y, theta_y_dot)

        F_x = self.k * dE_x * np.cos(theta_x) * theta_x_dot
        F_y = self.k * dE_y * np.cos(theta_y) * theta_y_dot

        # Keep the cart from spending the whole swing-up pinned against a rail.
        F_x += -self.position_gain * x - self.velocity_gain * x_dot
        F_y += -self.position_gain * y - self.velocity_gain * y_dot

        return np.clip(np.array([F_x, F_y]), -self.control_limit, self.control_limit)
