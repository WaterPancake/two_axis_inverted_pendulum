"""
Mujoco backend for the the two axis inverted pendulum
"""

import argparse
import mujoco
import mujoco.viewer
import numpy as np
from numpy import random
from numpy.random import Generator
import time
from pathlib import Path
from typing import Callable, Optional

MODEL_PATH = Path(__file__).resolve().parent.parent / "assets" / "mk2.xml"


class ArrowKeyController:
    """Maps arrow key presses into x/y cart forces."""

    def __init__(
        self, force_step: float = 0.25, max_force: float = 1.0, decay: float = 0.9
    ):
        self.force_step = float(force_step)
        self.max_force = float(max_force)
        self.decay = float(decay)
        self._action = np.zeros(2, dtype=float)

    def on_key(self, keycode: int) -> None:
        if keycode == 265:
            self._action[1] += self.force_step
        elif keycode == 264:
            self._action[1] -= self.force_step
        elif keycode == 263:
            self._action[0] -= self.force_step
        elif keycode == 262:
            self._action[0] += self.force_step

        np.clip(self._action, -self.max_force, self.max_force, out=self._action)

    def action(self) -> np.ndarray:
        action = self._action.copy()
        self._action *= self.decay
        self._action[np.abs(self._action) < 1e-3] = 0.0
        return action


class TwoAxisInvertedPendulum:
    def __init__(self, xml_path: Path = MODEL_PATH):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

    def reset(self, rng: Optional[Generator] = None):
        mujoco.mj_resetData(self.model, self.data)

        if rng is not None:
            pos = rng.uniform(-0.5, 0.5, 2)
            angle = rng.uniform(-0.5, 0.5, 2)
            cart_vel = rng.uniform(-0.5, 0.5, 2)
            pole_vel = rng.uniform(-0.5, 0.5, 2)

            self.data.qpos[0:2] = pos
            self.data.qpos[2:4] = angle
            self.data.qvel[0:2] = cart_vel
            self.data.qvel[2:4] = pole_vel

        mujoco.mj_forward(self.model, self.data)

    def get_obs(self):
        """
        | Idx | Observation                                                |
        |-----+------------------------------------------------------------|
        |  0  | x position of the cart                                     |
        |  1  | y position of the cart                                     |
        |  2  | angle of cart's pole from the x axis expressed in radians  |
        |  3  | angle of cart's pole from the y axis expressed in raidans  |
        |  4  | x velocity of the cart                                     |
        |  5  | y velocity of the cart                                     |
        |  6  | angular velocity of cart along the x axis                  |
        |  7  | angular velocity of cart along the y axis                  |
        """

        return np.concatenate([self.data.qpos, self.data.qvel])

    def get_obs_2p(self):
        x = self.get_obs()

        x = [float(round(a, 2)) for a in x]

        return x

    def control(self, action: np.ndarray) -> np.ndarray:
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

    def run_passive_viewer(
        self,
        step_callback: Optional[
            Callable[["TwoAxisInvertedPendulum"], np.ndarray]
        ] = None,
        timestep: float = 0.01,
    ) -> None:
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                action = random.uniform(-1, 1, size=2)
                if step_callback is not None:
                    action = np.asarray(step_callback(self), dtype=float)

                self.control(action)
                viewer.sync()
                time.sleep(timestep)

    def run_interactive_viewer(
        self,
        timestep: float = 0.01,
        force_step: float = 0.25,
        max_force: float = 1.0,
        decay: float = 0.9,
    ) -> None:
        controller = ArrowKeyController(
            force_step=force_step,
            max_force=max_force,
            decay=decay,
        )

        print("Interactive controls: arrow keys for x/y control.")

        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            key_callback=controller.on_key,
        ) as viewer:
            while viewer.is_running():
                self.control(controller.action())
                viewer.sync()
                time.sleep(timestep)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mode for the Mujoco viewer.")
    parser.add_argument(
        "--viewer",
        choices=("passive", "interactive"),
        default="passive",
    )
    parser.add_argument(
        "--random-reset",
        action="store_true",
        help="Randomize the initial state before launching the viewer.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys = TwoAxisInvertedPendulum()

    rng = random.default_rng(42)
    sys.reset(rng if args.random_reset else None)
    print(sys.get_obs_2p())

    if args.viewer == "interactive":
        sys.run_interactive_viewer()
    else:
        sys.run_passive_viewer()
