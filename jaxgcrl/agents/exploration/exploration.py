"""JAX-compatible exploration bonuses for reinforcement learning."""

from typing import Any, Callable, Dict, Optional, Tuple, Union, List

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.core import FrozenDict
from flax import linen as nn
from brax.training.acme import running_statistics, specs

# Parameter definitions for exploration bonuses
@struct.dataclass
class ExplorationBonusParams:
    """Base parameters for exploration bonuses."""
    reward_scale: float = 1.0
    normalize_bonus: bool = False
    bonus_normalize_observations: bool = True

    embedding_size: int = 256
    hidden_layer_sizes: Tuple[int, int] = (256, 256)
    bonus_learning_rate: float = 1e-4
    layer_norm: bool = False

    architecture: Tuple[Union[str, int], ...] = (256, "gelu", 256, "gelu")
    initial_phi_scale: float = 1e-3


@struct.dataclass
class RNDParams(ExplorationBonusParams):
    """Parameters for Random Network Distillation."""
    embedding_size: int = 256
    hidden_layer_sizes: Tuple[int, ...] = (256, 256)
    bonus_learning_rate: float = 1e-4
    layer_norm: bool = False


@struct.dataclass
class RNKParams(ExplorationBonusParams):
    """Parameters for Random Network for Knowledge."""
    architecture: Tuple = (256, "relu", 256, "relu", 256)
    initial_phi_scale: float = 1e-6
    normalize_bonus: bool = False  # RNK typically doesn't need normalization


# States for exploration bonuses
@struct.dataclass
class RNDState:
    """State for Random Network Distillation."""
    rnd_params: Any
    rnd_optimizer_state: optax.OptState
    bonus_rms: running_statistics.RunningStatisticsState
    apply_fn: Callable  # Store the apply function


@struct.dataclass
class RNKState:
    """State for Random Network for Knowledge."""
    params: FrozenDict
    Phi: jnp.ndarray
    bonus_rms: running_statistics.RunningStatisticsState
    apply_fn: Callable  # Store the apply function


# RND encoder creation function
def create_rnd_encoder(embedding_size: int, hidden_layer_sizes: Tuple[int, ...], layer_norm: bool = False):
    """Create the RND encoder network."""
    class RNDModule(nn.Module):
        """Module for RND with target and predictor networks."""

        @nn.compact
        def __call__(self, x):
            # Target network
            target = x
            for i, size in enumerate(hidden_layer_sizes):
                target = nn.Dense(size, name=f"target_{i}")(target)
                if layer_norm:
                    target = nn.LayerNorm(name=f"target_ln_{i}")(target)
                target = nn.relu(target)
            target = nn.Dense(embedding_size, name="target_out")(target)
            target = jax.lax.stop_gradient(target)

            # Predictor network
            predictor = x
            for i, size in enumerate(hidden_layer_sizes):
                predictor = nn.Dense(size, name=f"predictor_{i}")(predictor)
                if layer_norm:
                    predictor = nn.LayerNorm(name=f"predictor_ln_{i}")(predictor)
                predictor = nn.relu(predictor)
            predictor = nn.Dense(embedding_size, name="predictor_out")(predictor)

            # Compute squared error
            return jnp.square(target - predictor).mean(axis=-1, keepdims=True)

    return RNDModule()


# RNK encoder creation function
def create_rnk_encoder(architecture: Tuple, obs_size: int):
    """Create the feature encoder network for RNK."""
    layers = []
    for item in architecture:
        if isinstance(item, int):
            layers.append(nn.Dense(
                item,
                kernel_init=nn.initializers.he_normal(),
                bias_init=nn.initializers.uniform(2 * jnp.pi)
            ))
        elif item == "relu":
            layers.append(nn.relu)
        elif item == "tanh":
            layers.append(nn.tanh)
        elif item == "gelu":
            layers.append(nn.gelu)
        elif item == "sigmoid":
            layers.append(nn.sigmoid)
        else:
            raise ValueError(f"Unknown layer type: {item}")

    return nn.Sequential(layers)


# Pure functions for RND
def init_rnd(
    key: jnp.ndarray,
    obs_size: int,
    params: RNDParams
) -> RNDState:
    """Initialize RND state."""
    # Create RND encoder
    rnd_encoder = create_rnd_encoder(
        params.embedding_size,
        params.hidden_layer_sizes,
        params.layer_norm
    )

    # Create dummy input for initialization
    dummy_obs = jnp.zeros((1, obs_size))

    # Initialize parameters
    rnd_params = rnd_encoder.init(key, dummy_obs)

    # Initialize optimizer
    rnd_optimizer = optax.adam(learning_rate=params.bonus_learning_rate)
    rnd_optimizer_state = rnd_optimizer.init(rnd_params)

    # Initialize bonus normalization statistics
    bonus_rms = running_statistics.init_state(
        specs.Array((1,), jnp.dtype("float32"))
    )

    # Create a partial function for apply
    apply_fn = jax.tree_util.Partial(rnd_encoder.apply)

    return RNDState(
        rnd_params=rnd_params,
        rnd_optimizer_state=rnd_optimizer_state,
        bonus_rms=bonus_rms,
        apply_fn=apply_fn
    )


def compute_rnd_bonus(
    rnd_state: RNDState,
    observations: jnp.ndarray,
    params: RNDParams,
    normalizer_params: Optional[running_statistics.RunningStatisticsState] = None
) -> jnp.ndarray:
    """Compute RND bonus."""
    # Preprocess observations if normalizer is provided
    if normalizer_params is not None and params.bonus_normalize_observations:
        observations = running_statistics.normalize(
            observations, normalizer_params
        )

    # Direct call to the stored apply function
    bonus = rnd_state.apply_fn(rnd_state.rnd_params, observations)

    return jnp.squeeze(bonus, axis=-1)


def update_rnd_on_step(
    rnd_state: RNDState,
    observations: jnp.ndarray,
    params: RNDParams,
    normalizer_params: Optional[running_statistics.RunningStatisticsState] = None
) -> RNDState:
    """RND doesn't need updates during steps."""
    return rnd_state  # No-op for RND


def update_rnd(
    rnd_state: RNDState,
    observations: jnp.ndarray,
    params: RNDParams,
    normalizer_params: Optional[running_statistics.RunningStatisticsState] = None,
    key: Optional[jnp.ndarray] = None,
    pmap_axis_name: str = "i"
) -> Tuple[RNDState, Dict[str, Any]]:
    """Update RND predictor network."""
    # Preprocess observations if normalizer is provided
    if normalizer_params is not None and params.bonus_normalize_observations:
        observations = running_statistics.normalize(
            observations, normalizer_params
        )

    # Create RND encoder for this update
    rnd_encoder = create_rnd_encoder(
        params.embedding_size,
        params.hidden_layer_sizes,
        params.layer_norm
    )

    # Define loss function
    def loss_fn(rnd_params):
        return jnp.mean(rnd_encoder.apply(rnd_params, observations))

    # Compute loss and gradients
    value_and_grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = value_and_grad_fn(rnd_state.rnd_params)

    # Synchronize gradients across devices
    if pmap_axis_name:
        grads = jax.lax.pmean(grads, axis_name=pmap_axis_name)

    # Apply updates
    optimizer = optax.adam(learning_rate=params.bonus_learning_rate)
    updates, new_optimizer_state = optimizer.update(
        grads, rnd_state.rnd_optimizer_state
    )
    new_rnd_params = optax.apply_updates(rnd_state.rnd_params, updates)

    new_rnd_state = RNDState(
        rnd_params=new_rnd_params,
        rnd_optimizer_state=new_optimizer_state,
        bonus_rms=rnd_state.bonus_rms,
        apply_fn=rnd_state.apply_fn
    )

    return new_rnd_state, {"rnd_loss": loss}


# Pure functions for RNK
def init_rnk(
    key: jnp.ndarray,
    obs_size: int,
    params: RNKParams
) -> RNKState:
    """Initialize RNK state."""
    # Create network for feature extraction
    encoder = create_rnk_encoder(params.architecture, obs_size)

    # Create dummy input to determine feature dimension
    dummy_obs = jnp.zeros((1, obs_size))
    network_params = encoder.init(key, dummy_obs)

    # Determine feature dimension from a forward pass
    features = encoder.apply(network_params, dummy_obs)
    feature_dim = features.shape[-1]

    # Initialize Phi (inverse covariance matrix)
    Phi = jnp.eye(feature_dim) / params.initial_phi_scale

    # Initialize bonus normalization if needed
    bonus_rms = running_statistics.init_state(
        specs.Array((1,), jnp.dtype("float32"))
    )

    # Create a partial function for apply
    apply_fn = jax.tree_util.Partial(encoder.apply)

    return RNKState(
        params=network_params,
        Phi=Phi,
        bonus_rms=bonus_rms,
        apply_fn=apply_fn
    )


@jax.jit
def woodbury_update(Phi: jnp.ndarray, phi_batch: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Update inverse matrix using Woodbury identity for batch updates."""
    batch_size = phi_batch.shape[0]

    # For numerical stability, we do C + VA^(-1)U first
    inner_term = jnp.eye(batch_size) + phi_batch @ Phi @ phi_batch.T + eps * jnp.eye(batch_size)

    # Now compute inner_term^(-1)
    inner_term_inv = jnp.linalg.inv(inner_term)

    # Compute A^(-1)U
    APhi_U = Phi @ phi_batch.T  # shape: [feature_dim, batch]

    # Compute final update: A^(-1) - A^(-1)U(C^(-1) + VA^(-1)U)^(-1)VA^(-1)
    updated_Phi = Phi - APhi_U @ inner_term_inv @ phi_batch @ Phi

    return updated_Phi


@jax.jit
def woodbury_update_stable(Phi: jnp.ndarray, phi_batch: jnp.ndarray, reg_factor: float = 1e-8) -> jnp.ndarray:
    """Update inverse matrix using Woodbury identity with enhanced numerical stability."""
    batch_size = phi_batch.shape[0]
    feature_dim = Phi.shape[0]

    # 1. Compute intermediate products with better ordering
    PhiX = Phi @ phi_batch.T  # [feature_dim, batch_size]

    # 2. Scale regularization by trace of the matrix
    trace_PhiXXt = jnp.sum(phi_batch @ PhiX)
    reg = reg_factor * trace_PhiXXt / batch_size

    # 3. Form inner term with adaptive regularization
    XPhiXt = phi_batch @ PhiX  # [batch_size, batch_size]
    inner_term = jnp.eye(batch_size) + XPhiXt

    # 4. Use Cholesky decomposition instead of direct inversion
    # Add regularization to ensure positive definiteness
    inner_term_reg = inner_term + reg * jnp.eye(batch_size)

    # 5. Solve the system using Cholesky for better numerical stability
    # L = cholesky(inner_term_reg)
    # inner_term_inv_X_Phi = jax.scipy.linalg.solve_triangular(L.T,
    #                          jax.scipy.linalg.solve_triangular(L, phi_batch @ Phi, lower=True),
    #                          lower=False)

    # For simplicity, use jax.scipy.linalg.solve which internally uses an appropriate method
    inner_term_inv_X_Phi = jax.scipy.linalg.solve(inner_term_reg, phi_batch @ Phi)

    # 6. Compute final update with better ordering of operations
    updated_Phi = Phi - PhiX @ inner_term_inv_X_Phi

    # 7. Ensure symmetry (optional, for precision matrices)
    updated_Phi = (updated_Phi + updated_Phi.T) / 2

    return updated_Phi


def compute_rnk_bonus(
    rnk_state: RNKState,
    observations: jnp.ndarray,
    params: RNKParams,
    normalizer_params: Optional[running_statistics.RunningStatisticsState] = None
) -> jnp.ndarray:
    """Compute RNK bonus."""
    # Preprocess observations if normalizer is provided
    if normalizer_params is not None and params.bonus_normalize_observations:
        observations = running_statistics.normalize(
            observations, normalizer_params
        )

    # Extract features using stored apply function
    phi = rnk_state.apply_fn(rnk_state.params, observations)

    # Compute bonus: 0.5 * log(1 + phi^T Phi phi)
    # We use einsum for efficient batch computation
    bonus = 0.5 * jnp.log(1 + jnp.einsum('bi,ij,bj->b', phi, rnk_state.Phi, phi))

    return bonus


def update_rnk_on_step(
    rnk_state: RNKState,
    observations: jnp.ndarray,
    params: RNKParams,
    normalizer_params: Optional[running_statistics.RunningStatisticsState] = None
) -> RNKState:
    """Update RNK state during environment interaction."""
    # Preprocess observations if normalizer is provided
    if normalizer_params is not None and params.bonus_normalize_observations:
        observations = running_statistics.normalize(
            observations, normalizer_params
        )

    # Extract features using stored apply function
    phi = rnk_state.apply_fn(rnk_state.params, observations)

    # Update Phi using Woodbury identity
    new_Phi = woodbury_update_stable(rnk_state.Phi, phi)

    # Create updated state (params and apply_fn stay the same)
    return RNKState(
        params=rnk_state.params,
        Phi=new_Phi,
        bonus_rms=rnk_state.bonus_rms,
        apply_fn=rnk_state.apply_fn
    )


def update_rnk(
    rnk_state: RNKState,
    observations: jnp.ndarray,
    params: RNKParams,
    normalizer_params: Optional[running_statistics.RunningStatisticsState] = None,
    key: Optional[jnp.ndarray] = None,
    pmap_axis_name: str = "i"
) -> Tuple[RNKState, Dict[str, Any]]:
    """RNK doesn't need updates during training."""
    # Just return metrics about the current state
    metrics = {"rnk_condition_number": jnp.linalg.cond(rnk_state.Phi)}
    return rnk_state, metrics


# Dispatch functions for different bonus types
def init_exploration_bonus(
    bonus_type: str,
    key: jnp.ndarray,
    obs_size: int,
    bonus_params: Dict
) -> Any:
    """Initialize the appropriate exploration bonus state."""
    if bonus_type == "none" or not bonus_type:
        return None

    if bonus_type == "rnd":
        return init_rnd(key, obs_size, bonus_params)
    elif bonus_type == "rnk":
        return init_rnk(key, obs_size, bonus_params)
    else:
        raise ValueError(f"Unknown exploration bonus type: {bonus_type}")


def compute_bonus(
    bonus_type: str,
    bonus_state: Any,
    observations: jnp.ndarray,
    bonus_params: ExplorationBonusParams,
    normalizer_params: Optional[running_statistics.RunningStatisticsState] = None
) -> jnp.ndarray:
    """Compute exploration bonus."""
    if bonus_type == "none" or not bonus_type or bonus_state is None:
        return jnp.zeros(observations.shape[0])

    if bonus_type == "rnd":
        return compute_rnd_bonus(bonus_state, observations, bonus_params, normalizer_params)
    elif bonus_type == "rnk":
        return compute_rnk_bonus(bonus_state, observations, bonus_params, normalizer_params)
    else:
        raise ValueError(f"Unknown exploration bonus type: {bonus_type}")


def update_bonus_on_step(
    bonus_type: str,
    bonus_state: Any,
    observations: jnp.ndarray,
    bonus_params: ExplorationBonusParams,
    normalizer_params: Optional[running_statistics.RunningStatisticsState] = None
) -> Any:
    """Update exploration bonus state during environment interaction."""
    if bonus_type == "none" or not bonus_type or bonus_state is None:
        return bonus_state

    if bonus_type == "rnd":
        return update_rnd_on_step(bonus_state, observations, bonus_params, normalizer_params)
    elif bonus_type == "rnk":
        return update_rnk_on_step(bonus_state, observations, bonus_params, normalizer_params)
    else:
        raise ValueError(f"Unknown exploration bonus type: {bonus_type}")


def update_bonus(
    bonus_type: str,
    bonus_state: Any,
    observations: jnp.ndarray,
    bonus_params: ExplorationBonusParams,
    normalizer_params: Optional[running_statistics.RunningStatisticsState] = None,
    key: Optional[jnp.ndarray] = None,
    pmap_axis_name: str = "i"
) -> Tuple[Any, Dict[str, Any]]:
    """Update exploration bonus during training."""
    if bonus_type == "none" or not bonus_type or bonus_state is None:
        return bonus_state, {}

    if bonus_type == "rnd":
        return update_rnd(bonus_state, observations, bonus_params,
                         normalizer_params, key, pmap_axis_name)
    elif bonus_type == "rnk":
        return update_rnk(bonus_state, observations, bonus_params,
                         normalizer_params, key, pmap_axis_name)
    else:
        raise ValueError(f"Unknown exploration bonus type: {bonus_type}")