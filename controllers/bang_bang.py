"""Bang-bang controller for the two-axis cart-pole."""

import numpy as np


class BangBang:
    # 0.35 radians ≈ 20 degrees
    # 0.25 radians ≈ 15 degree
    # 0.175 radians ≈ 10 degree
    def __init__(self, theta_threshold: float = 0.35):
        self.theta_threshold = theta_threshold

    def control(self, cur_state: np.ndarray):
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
