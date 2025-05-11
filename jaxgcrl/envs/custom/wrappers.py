import jax
import jax.numpy as jnp
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from typing import Any, Dict, Optional, Tuple, Union
from flax import struct


# Extend the Brax State class to include gymnax_state
@struct.dataclass
class GymnaxBraxState(State):
    """Extended State class that includes gymnax_state."""
    gymnax_state: Any = None
    rng: jax.random.PRNGKey = None


class GymnaxToBraxWrapper(PipelineEnv):
    """Wrapper to convert a Gymnax environment to a Brax environment.

    This wrapper maintains Gymnax state in the Brax State dataclass,
    following JAX's functional paradigm with immutable state.
    It uses the Gymnax environment for the actual dynamics, while providing a
    compatible interface with Brax's PipelineEnv API.
    """

    def __init__(
            self,
            gymnax_env,
            env_params=None,
            sys=None,
            backend="generalized",
            n_frames=5,
            **kwargs
    ):
        """Initialize the wrapper with a Gymnax environment.

        Args:
            gymnax_env: The Gymnax environment to wrap
            env_params: Parameters for the Gymnax environment. If None, default params are used.
            sys: Optional Brax system. If None, a minimal dummy system is created.
            backend: The Brax physics backend to use
            n_frames: Number of physics frames per step
            **kwargs: Additional arguments to pass to PipelineEnv
        """
        # Store the Gymnax environment and its parameters
        self.gymnax_env = gymnax_env
        self.env_params = env_params if env_params is not None else gymnax_env.default_params

        # Create a minimal system if none is provided
        if sys is None:
            # This is a minimal dummy system
            # For a real implementation, you might want to create a more appropriate system
            # based on the Gymnax environment's characteristics
            sys = self._create_minimal_system()

        # Initialize the PipelineEnv with the system
        super().__init__(sys=sys, backend=backend, n_frames=n_frames, **kwargs)

        # Extract action and observation dimensions from Gymnax
        self._action_dim = self._get_gymnax_action_size()
        self._obs_dim = self._get_gymnax_obs_size()

    def _create_minimal_system(self):
        """Create a minimal Brax system for the wrapper.

        This creates a very basic physical system that satisfies Brax's requirements.
        In a real implementation, you might want to create a system that better
        represents the Gymnax environment's physical characteristics.

        Returns:
            A minimal Brax system
        """
        # For a real implementation, you would create a system that matches
        # the Gymnax environment's characteristics
        # This is just a placeholder using a simple system
        try:
            # Try to load a simple system
            from etils import epath
            path = epath.resource_path("brax") / "envs/assets/inverted_pendulum.xml"
            return mjcf.load(path)
        except:
            # If that fails, create a truly minimal system
            raise ValueError("Could not create a minimal system. Please provide a system.")

    def _get_gymnax_action_size(self) -> int:
        """Get the action dimension from the Gymnax environment.

        Returns:
            The size of the action space
        """
        return self.gymnax_env.action_space(self.env_params).shape[0]

    def _get_gymnax_obs_size(self) -> int:
        """Get the observation dimension from the Gymnax environment.

        Returns:
            The size of the observation space
        """
        return self.gymnax_env.observation_space(self.env_params).shape[0]

    def reset(self, rng: jax.Array) -> GymnaxBraxState:
        """Reset both the Gymnax environment and the Brax pipeline.

        Args:
            rng: Random key for initialization

        Returns:
            GymnaxBraxState containing both Brax and Gymnax state information
        """
        # Split the RNG keys for Gymnax and Brax
        rng, gymnax_rng, brax_rng = jax.random.split(rng, 3)

        # Reset the Gymnax environment with params
        gymnax_obs, gymnax_state = self.gymnax_env.reset(gymnax_rng, self.env_params)

        # Initialize a minimal pipeline state
        # In a more complete implementation, you might want to
        # derive qpos and qvel from the Gymnax state
        qpos = self.sys.init_q
        qvel = jnp.zeros(self.sys.qd_size())
        pipeline_state = self.pipeline_init(qpos, qvel)

        # Convert Gymnax observation to Brax format if needed
        obs = self._convert_gymnax_obs(gymnax_state)

        # Initialize with zero reward and not done
        reward = jnp.array(0.0)
        done = jnp.array(0.0)
        metrics = {}  # Add any metrics you want to track

        # Create and return the extended state that includes gymnax_state
        return GymnaxBraxState(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            gymnax_state=gymnax_state,
            rng=rng
        )

    def step(self, state: GymnaxBraxState, action: jax.Array) -> GymnaxBraxState:
        """Step both the Gymnax environment and the Brax pipeline.

        Args:
            state: Current GymnaxBraxState
            action: Action to take

        Returns:
            Next GymnaxBraxState
        """
        assert state.gymnax_state is not None, "Environment must be reset before stepping"

        # Get stored RNG and params from info
        rng = state.rng

        # Split RNG for stepping
        rng, step_rng = jax.random.split(rng)

        # Step the Gymnax environment using the gymnax_state from the state object
        gymnax_obs, gymnax_next_state, gymnax_reward, gymnax_done, gymnax_info = self.gymnax_env.step(
            step_rng, state.gymnax_state, action, self.env_params
        )

        # Step the Brax pipeline
        # This maintains the physical simulation, though in this wrapper
        # it may not be fully synchronized with the Gymnax state
        # pipeline_state = self.pipeline_step(state.pipeline_state, action)

        # Convert Gymnax observation to Brax format if needed
        obs = self._convert_gymnax_obs(gymnax_next_state)

        # Create and return the new state with updated gymnax_state
        return GymnaxBraxState(
            pipeline_state=state.pipeline_state,
            obs=obs,
            reward=gymnax_reward,
            done=gymnax_done * 1.,
            metrics=state.metrics,
            info=state.info,
            gymnax_state=gymnax_next_state,
            rng=rng
        )

    def _convert_gymnax_obs(self, gymnax_state):
        """Convert Gymnax state to Brax observation format.

        Uses the Gymnax environment's get_obs method to extract the observation
        from the state, then ensures it's in the correct format for Brax.

        Args:
            gymnax_state: Gymnax environment state

        Returns:
            Observation formatted for Brax compatibility
        """
        # Get the observation from the Gymnax state using the environment's method
        return self.gymnax_env.get_obs(gymnax_state)

    @property
    def observation_size(self) -> int:
        """Get the size of the observation space.

        Returns:
            The size of the observation vector
        """
        return self._obs_dim

    @property
    def action_size(self) -> int:
        """Get the size of the action space.

        Returns:
            The size of the action vector
        """
        return self._action_dim