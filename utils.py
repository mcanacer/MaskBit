import jax
import jax.numpy as jnp

import math


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


def split_factorized_tokens(tokens, codebook_size, splits):
    bit_shift = int(math.log2(codebook_size)) // splits
    bit_mask = (1 << bit_shift) - 1

    split_tokens = []
    for i in range(splits):
        split_tokens.append((tokens & (bit_mask << (i * bit_shift))) >> (i * bit_shift))

    return jnp.stack(split_tokens, axis=2)


def combine_factorized_tokens(tokens, codebook_size, splits):
    combined_tokens = jnp.zeros((tokens.shape[0], tokens.shape[1]))
    bit_shift = int(math.log2(codebook_size)) // splits
    for i in range(splits):
        combined_tokens += (tokens[..., i] << (i * bit_shift))

    return combined_tokens


def get_mask_tokens(rng, tokens, mask_token, mode="arccos"):
    rng, sample_rng = jax.random.split(rng, 2)
    r = jax.random.uniform(sample_rng, shape=(tokens.shape[0],))

    if mode == "linear":
        val_to_mask = 1 - r
    elif mode == "square":
        val_to_mask = 1 - (r ** 2)
    elif mode == "cosine":
        val_to_mask = jnp.cos(r * math.pi * 0.5)
    elif mode == "arccos":
        val_to_mask = jnp.arccos(r) / (math.pi * 0.5)
    else:
        raise ValueError("Invalid mode.")

    masked_tokens = jnp.full_like(tokens, mask_token)
    mask = jax.random.uniform(rng, shape=tokens.shape) < jnp.expand_dims(val_to_mask, axis=(1, 2))

    masked_tokens = mask * masked_tokens + (~mask) * tokens
    return masked_tokens, mask


def get_masking_ratio(progress, mode="arccos"):
    r = jnp.array(progress)
    if mode == "root":
        val_to_mask = 1 - (r ** 0.5)
    elif mode == "square":
        val_to_mask = 1 - (r ** 2)
    elif mode == "cosine":
        val_to_mask = jnp.cos(r * math.pi * 0.5)
    elif mode == "arccos":
        val_to_mask = jnp.arccos(r) / (math.pi * 0.5)
    elif mode == "linear":
        val_to_mask = 1 - r
    else:
        raise ValueError("Invalid mode. Choose between 'linear','square', 'cosine', 'arccos', 'root'.")

    val_to_mask = jax.lax.clamp(val_to_mask, 1e-6, 1.0)
    return val_to_mask
