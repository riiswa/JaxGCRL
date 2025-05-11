# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A cart-pole swing-up environment."""

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp


class CartPole(PipelineEnv):
    # pyformat: disable
    """### Description

    This environment consists of a cart that can move along a frictionless track and
    a pole attached to the cart with a hinge joint. The goal is to swing the pole up
    and balance it by applying forces to the cart.

    The pole starts pointing downward, and the agent must learn to swing it up and
    keep it balanced in an upright position.

    ### Action Space

    The agent takes a 1-element vector for actions. The action space is a continuous
    `(action)` in `[-1, 1]`, where `action` represents the numerical force applied
    to the cart (with magnitude representing the amount of force and sign representing
    the direction).

    | Num | Action                    | Control Min | Control Max | Name (in
    corresponding config) | Joint | Unit      |
    |-----|---------------------------|-------------|-------------|--------------------------------|-------|-----------|
    | 0   | Force applied on the cart | -1          | 1           | slide
    | slider | Force (N) |

    ### Observation Space

    The observation is a `ndarray` with shape `(4,)` where the elements correspond to:

    | Num | Observation                                   | Min  | Max | Name (in
    corresponding config) | Joint | Unit                     |
    |-----|-----------------------------------------------|------|-----|--------------------------------|-------|--------------------------|
    | 0   | Position of the cart along the track          | -1.8 | 1.8 | slider
    | slider | position (m)             |
    | 1   | Angle of the pole with the vertical           | -Inf | Inf | hinge_1
    | hinge_1 | angle (rad)              |
    | 2   | Linear velocity of the cart                   | -Inf | Inf | slider
    | slider | velocity (m/s)           |
    | 3   | Angular velocity of the pole                  | -Inf | Inf | hinge_1
    | hinge_1 | angular velocity (rad/s) |

    ### Rewards

    The reward is sparse:
    - 1.0 when the cart is within [-0.25, 0.25] and the pole's cosine angle is within [0.995, 1.0]
    - 0.0 otherwise

    ### Starting State

    The pole starts pointing downward (π radians from vertical) with small random noise.
    The cart starts at the center of the track with small random noise.
    Small random initial velocities are added to both the cart and pole.

    ### Physical Limits

    The physical limits are enforced by the simulation:
    - The cart position is limited to the range [-1.8, 1.8] by joint limits.
    - The pole can rotate freely.
    """
    # pyformat: enable

    # Constants for reward calculation
    _CART_RANGE = (-0.25, 0.25)
    _ANGLE_COSINE_RANGE = (0.995, 1.0)

    def __init__(self, backend='generalized', **kwargs):
        """Initialize the CartPoleSwingUp environment.

        Args:
          backend: Physics backend to use (generalized, spring, or positional).
          **kwargs: Additional keyword arguments passed to the parent class.
        """
        # The XML content should be saved as a file, but we'll use mjcf.loads for simplicity
        xml_content = """
            <mujoco model="cart-pole">
              <option timestep="0.01">
                <flag contact="disable"/>
              </option>

              <default>
                <default class="pole">
                  <joint type="hinge" axis="0 1 0"  damping="2e-6"/>
                  <geom type="capsule" fromto="0 0 0 0 0 1" size="0.045" mass=".1"/>
                </default>
              </default>

              <worldbody>
                <geom name="floor" pos="0 0 -.05" size="4 4 .2" type="plane"/>
                <geom name="rail1" type="capsule" pos="0  .07 1" zaxis="1 0 0" size="0.02 2"/>
                <geom name="rail2" type="capsule" pos="0 -.07 1" zaxis="1 0 0" size="0.02 2"/>
                <body name="cart" pos="0 0 1">
                  <joint name="slider" type="slide" limited="true" axis="1 0 0" range="-1.8 1.8" solreflimit=".08 1" damping="5e-4"/>
                  <geom name="cart" type="box" size="0.2 0.15 0.1" mass="1"/>
                  <body name="pole_1" childclass="pole">
                    <joint name="hinge_1"/>
                    <geom name="pole_1"/>
                  </body>
                </body>
              </worldbody>

              <actuator>
                <motor name="slide" joint="slider" gear="10" ctrllimited="true" ctrlrange="-1 1" />
              </actuator>
            </mujoco>
            """

        # Load the model from the XML string
        sys = mjcf.loads(xml_content)

        # Configure the number of frames based on the backend
        n_frames = 2
        if backend in ['spring', 'positional']:
            sys = sys.tree_replace({'opt.timestep': 0.005})
            n_frames = 4

        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        # Store limits for easier access
        self._cart_pos_limit = 1.8  # From the XML range attribute

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state with pole pointing down."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        # Initialize q with pole pointing down (π radians from vertical)
        q = self.sys.init_q.copy()

        # Add small noise to cart position (near center)
        q = q.at[0].set(0.01 * jax.random.normal(rng1))

        # Set pole angle to π (pointing down) with small noise
        q = q.at[1].set(jp.pi + 0.01 * jax.random.normal(rng2))

        # Initialize velocities with small random values
        rng, rng3 = jax.random.split(rng)
        qd = 0.01 * jax.random.normal(rng3, (self.sys.qd_size(),))

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        reward, done = jp.zeros(2)
        metrics = {}

        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""

        # Scale action from [-1,1] to actuator limits
        action_min = self.sys.actuator.ctrl_range[:, 0]
        action_max = self.sys.actuator.ctrl_range[:, 1]
        scaled_action = (action + 1) * (action_max - action_min) * 0.5 + action_min

        pipeline_state = self.pipeline_step(state.pipeline_state, scaled_action)
        obs = self._get_obs(pipeline_state)

        # Calculate sparse reward
        cart_position = obs[0]
        pole_angle = obs[1]

        # Compute cosine of pole angle (1.0 when perfectly upright)
        pole_angle_cosine = jp.cos(pole_angle)

        # Check if cart is within bounds
        cart_in_bounds = jp.logical_and(
            cart_position >= self._CART_RANGE[0],
            cart_position <= self._CART_RANGE[1]
        )

        # Check if pole angle is within bounds
        angle_in_bounds = jp.logical_and(
            pole_angle_cosine >= self._ANGLE_COSINE_RANGE[0],
            pole_angle_cosine <= self._ANGLE_COSINE_RANGE[1]
        )

        reward = jp.logical_and(cart_in_bounds, angle_in_bounds) * 1.0

        # Never terminate the episode
        done = jp.zeros(())

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    @property
    def action_size(self):
        return 1

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observe cart-pole state: [cart_pos, pole_angle, cart_vel, pole_ang_vel]."""
        return jp.concatenate([pipeline_state.q, pipeline_state.qd])