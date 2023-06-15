import jax
import jax.numpy as jnp
from collections import namedtuple


def generate_sin(xs, amp, period, phase):
    return amp * jnp.sin(xs * 2 * jnp.pi / period + phase)
generate_sin_batch = jax.jit(jax.vmap(generate_sin, [None, 0, 0, 0], 0))

def generate_sin_base(xs, amp, period, phase, base):
    return base + generate_sin(xs, amp, period, phase)
generate_sin_base_batch = jax.jit(jax.vmap(generate_sin, [None, 0, 0, 0, None], 0)) # base is shared in batch

def generate_damped_oscillator(xs, amp, period, phase, decay):
    return generate_sin(xs, amp, period, phase) * jnp.exp(-decay * xs)

generate_damped_oscillator_batch = jax.jit(jax.vmap(generate_damped_oscillator, [None, 0, 0, 0, None], 0)) # decay is shared in batch