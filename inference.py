import sys
import yaml
import os

import jax
import jax.numpy as jnp
from flax import serialization
import numpy as np
import torch
from torchvision.utils import save_image

import functools

from utils import get_masking_ratio, combine_factorized_tokens
from vqmodel import ConvVQModel
from transformer import LFQBert


def load_checkpoint(path, state_template):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return serialization.from_bytes(state_template, f.read())


def sample_gumbel(key, shape, loc=0.0, scale=1.0, eps=1e-20):
    u = jax.random.uniform(key, shape=shape, minval=0.0, maxval=1.0)
    return loc - scale * jnp.log(-jnp.log(u + eps))


def sample(
        rng,
        maskbit_apply_fn,
        maskbit_params,
        vqmodel,
        vqmodel_params,
        labels,
        num_samples,
        softmax_temperature,
        randomize_temperature,
        num_steps,
        guidance_scale,
        mask_token,
        patch_size,
        guidance_annealing,
        scale_pow,
        codebook_size,
        codebook_splits,
):
    def make_predict_fn(*, apply_fn):
        return jax.pmap(
            functools.partial(apply_fn, train=False),
            axis_name='batch',
            donate_argnums=()
        )

    def shard(x):
        n, *s = x.shape
        return x.reshape((num_devices, n // num_devices, *s))

    def unshard(x):
        d, b, *s = x.shape
        return x.reshape((d * b, *s))

    if labels is None:
        labels = jnp.array([1, 7, 90, 282, 179, 751, 404, 963], dtype=jnp.int32)
        # labels = jnp.full((num_samples,), 963, dtype=jnp.int32)

    drop_labels = jnp.ones((num_samples,), dtype=bool)
    spatial_size = int(patch_size ** 2)
    num_splits = int(codebook_splits)

    masked_tokens = jnp.full((num_samples, spatial_size, num_splits), mask_token, dtype=jnp.int32)
    num_maskable = spatial_size * num_splits
    mask = (masked_tokens == mask_token)

    devices = jax.local_devices()
    num_devices = len(devices)

    replicate = lambda tree: jax.device_put_replicated(tree, jax.local_devices())
    unreplicate = lambda tree: jax.tree_util.tree_map(lambda x: x[0], tree)

    maskbit_predict_fn = make_predict_fn(apply_fn=maskbit_apply_fn)

    maskbit_params_repl = replicate(maskbit_params)

    del maskbit_params

    l_full_tokens = []
    for i in range(num_steps):
        rng, categorical_rng, gumbel_rng = jax.random.split(rng, 3)
        progress = (i + 1) / num_steps
        if guidance_scale != 0.0:
            cat_masked_tokens = jnp.concatenate([masked_tokens, masked_tokens], axis=0)
            cat_labels = jnp.concatenate([labels, labels], axis=0)
            cat_drop_labels = jnp.concatenate([~drop_labels, drop_labels], axis=0)

            masked_tokens_shard = jax.tree_util.tree_map(lambda x: shard(x), cat_masked_tokens)
            labels_shard = jax.tree_util.tree_map(lambda x: shard(x), cat_labels)
            drop_labels_shard = jax.tree_util.tree_map(lambda x: shard(x), cat_drop_labels)

            logits = maskbit_predict_fn(
                maskbit_params_repl,
                masked_tokens_shard,
                labels_shard,
                drop_labels_shard,
            )

            logits_with_class, logits_without_class = jnp.split(logits, 2, axis=0)

            if guidance_annealing == "none":
                scale_step = 1.0
            elif guidance_annealing == "linear":
                scale_step = i / num_steps
            elif guidance_annealing == "cosine":
                scale_pow = jnp.array(scale_pow)
                scale_step = (1 - jnp.cos(((i / num_steps) ** scale_pow) * jnp.pi)) * 0.5

            scale = guidance_scale * scale_step
            logits = logits_with_class + scale * (logits_with_class - logits_without_class)
        else:
            masked_tokens_shard = jax.tree_util.tree_map(lambda x: shard(x), masked_tokens)
            labels_shard = jax.tree_util.tree_map(lambda x: shard(x), labels)
            drop_labels_shard = jax.tree_util.tree_map(lambda x: shard(x), drop_labels)

            logits = maskbit_predict_fn(
                maskbit_params_repl,
                masked_tokens_shard,
                labels_shard,
                ~drop_labels_shard
            )

        logits = jax.tree_util.tree_map(lambda x: unshard(x), logits)
        probabilities = jax.nn.softmax(logits / softmax_temperature, axis=-1)
        predicted_tokens = jax.random.categorical(categorical_rng, jnp.log(probabilities + 1e-20), axis=-1)

        num_masked = jnp.sum(mask, axis=(1, 2))[0]

        predicted_tokens = jnp.where(mask, predicted_tokens, masked_tokens)

        gather_conf = jnp.take_along_axis(probabilities, predicted_tokens[..., None], axis=-1).squeeze(-1)
        confidence = jnp.where(mask, gather_conf, jnp.inf)

        noise = sample_gumbel(gumbel_rng, shape=predicted_tokens.shape) * randomize_temperature * (1 - progress)
        confidence = jnp.log(confidence + 1e-20) + noise

        mask_ratio = get_masking_ratio(progress)

        mask_len = jnp.floor(mask_ratio * num_maskable)
        num_tokens_to_mask = jax.lax.clamp(mask_len.astype(jnp.int32), jnp.ones_like(num_masked), num_masked - 1, )
        sorted_confidence = jnp.sort(confidence.reshape(num_samples, -1), axis=-1)
        threshold = sorted_confidence[:, num_tokens_to_mask - 1]

        should_mask = (confidence <= jnp.expand_dims(threshold, axis=(-2, -1)))
        masked_tokens = jnp.where(should_mask, mask_token, predicted_tokens)
        mask = (masked_tokens == mask_token)
        l_full_tokens.append(predicted_tokens)

    predicted_tokens = combine_factorized_tokens(predicted_tokens, codebook_size, codebook_splits)

    generated_images = vqmodel.apply(vqmodel_params, predicted_tokens, method=vqmodel.decode_tokens)
    generated_images = (generated_images + 1.0) / 2.0
    return jnp.clip(generated_images, 0.0, 1.0), l_full_tokens


def main(config_path):
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    maskbit_config = config['model']
    vqmodel_config = config['vqmodel']

    seed = maskbit_config['seed']
    rng = jax.random.PRNGKey(seed)

    vqmodel = ConvVQModel(**vqmodel_config['params'])
    maskbit = LFQBert(**maskbit_config['params'])

    checkpoint_path = maskbit_config['checkpoint_path']
    vqmodel_checkpoint_path = vqmodel_config['checkpoint_path']
    masbit_params = load_checkpoint(checkpoint_path, None)['ema_params']
    vqmodel_params = load_checkpoint(vqmodel_checkpoint_path, None)['ema_params']

    generated_samples, _ = sample(
        rng=rng,
        maskbit_apply_fn=maskbit.apply,
        maskbit_params=masbit_params,
        vqmodel=vqmodel,
        vqmodel_params=vqmodel_params,
        labels=None,
        num_samples=8,
        softmax_temperature=maskbit_config['softmax_temperature'],
        randomize_temperature=maskbit_config['randomize_temperature'],
        num_steps=maskbit_config['num_steps'],
        guidance_scale=maskbit_config['guidance_scale'],
        mask_token=maskbit_config['params']['mask_token'],
        patch_size=16,
        guidance_annealing=maskbit_config['guidance_annealing'],
        scale_pow=maskbit_config['scale_pow'],
        codebook_size=2 ** vqmodel_config['params']['token_size'],
        codebook_splits=maskbit_config['params']['splits'],
    )

    for i in range(generated_samples.shape[0]):
        img = np.array(generated_samples[i])

        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)

        save_image(img, f'/content/drive/MyDrive/MaskBit/gen_images/generated_image{i}.png')


if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise ValueError('you must provide config file')
    main(sys.argv[1])
