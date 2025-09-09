import jax
import jax.numpy as jnp


def clamp_log(x, eps=1e-5):
    return jnp.log(jnp.clip(x, eps))


def entropy_loss_fn(
        affinity,
        tempature,
        entropy_gamma=1.0,
):
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= tempature

    probability = jax.nn.softmax(flat_affinity, axis=-1)
    average_probability = jnp.mean(probability, axis=0)

    per_sample_entropy = -1 * jnp.mean(jnp.sum(probability * clamp_log(probability), axis=-1))
    avg_entropy = jnp.sum(-1 * average_probability * clamp_log(average_probability))

    return per_sample_entropy, avg_entropy * entropy_gamma
