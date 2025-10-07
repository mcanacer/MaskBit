import sys
import yaml
import os

import jax
import jax.numpy as jnp
import optax
from flax import serialization
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import wandb
from vqmodel import ConvVQModel
from transformer import LFQBert

from utils import split_factorized_tokens, get_mask_tokens

import numpy as np


def save_checkpoint(path, state):
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(state))


def load_checkpoint(path, state_template):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return serialization.from_bytes(state_template, f.read())


def ema_update(ema_params, new_params, decay):
    return jax.tree_util.tree_map(
        lambda e, p: decay * e + (1.0 - decay) * p,
        ema_params,
        new_params
    )


def cross_entropy_loss(logits, targets, label_smoothing):
    num_classes = logits.shape[-1]
    log_probs = jax.nn.log_softmax(logits)

    if label_smoothing > 0.0:
        on_value = 1.0 - label_smoothing
        off_value = label_smoothing / num_classes
        y = jax.nn.one_hot(targets, num_classes)
        y = on_value * y + (1- y) * off_value
        loss = -jnp.sum(y * log_probs, axis=-1)
    else:
        loss = -log_probs[jnp.arange(targets.shape[0]), targets]
    return loss.mean()


def make_update_fn(*, vqmodel_apply_fn, vqmodel_method, maskbit_apply_fn, optimizer, class_label_dropout,
                   label_smoothing, codebook_size, splits, mask_token, ema_decay):
    def update_fn(params, opt_state, vqmodel_params, images, labels, rng, ema_params):
        def loss_fn(params):
            mask_rng, drop_label_rng, dropout_rng = jax.random.split(rng, 3)

            z_quantized, result_dict = vqmodel_apply_fn(vqmodel_params, images, method=vqmodel_method)
            input_tokens = result_dict['min_encoding_indices']
            input_tokens = jnp.reshape(input_tokens, (input_tokens.shape[0], -1))

            input_tokens = split_factorized_tokens(input_tokens, codebook_size, splits)

            masked_tokens, masks = get_mask_tokens(mask_rng, input_tokens, mask_token)

            drop_label_mask = jax.random.uniform(drop_label_rng, shape=labels.shape) < class_label_dropout
            logits = maskbit_apply_fn(params, masked_tokens, labels, drop_label_mask, train=True,
                                      rngs={'dropout': dropout_rng})

            b, n, m, num_codebook = logits.shape

            logits_flat = logits.reshape(-1, num_codebook)
            targets_flat = input_tokens.reshape(-1)

            loss = cross_entropy_loss(logits_flat, targets_flat, label_smoothing)

            return loss

        loss, grad = jax.value_and_grad(loss_fn)(params)

        loss, grad = jax.tree_util.tree_map(
            lambda x: jax.lax.pmean(x, axis_name='batch'),
            (loss, grad),
        )

        updates, opt_state = optimizer.update(grad, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        new_ema_params = ema_update(ema_params, new_params, decay=ema_decay)

        return new_params, opt_state, new_ema_params, loss

    return jax.pmap(update_fn, axis_name='batch', donate_argnums=())


def main(config_path):
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    maskbit_config = config['model']
    dataset_params = config['dataset_params']
    vqmodel_config = config['vqmodel']
    wandb_config = config['wandb']

    seed = maskbit_config['seed']

    transform = transforms.Compose([
        transforms.RandomResizedCrop(
            size=256,
            scale=(0.8, 1.0),
            ratio=(1.0, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),  # Normalize [0, 1]
        transforms.Lambda(lambda t: (t * 2) - 1),  # Scale [-1, 1]
        transforms.Lambda(lambda x: x.permute(1, 2, 0)),  # Convert [C, H, W] to [H, W, C]
    ])

    train_dataset = ImageFolder(
        root=dataset_params['data_path'],
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=dataset_params['batch_size'],
        shuffle=True,
        num_workers=dataset_params['num_workers'],
        drop_last=True,
    )

    vqmodel = ConvVQModel(**vqmodel_config['params'])

    maskbit = LFQBert(**maskbit_config['params'])

    opt_params = config['model']['optim_params']

    optimizer = optax.chain(
        optax.adamw(
            learning_rate=float(opt_params['learning_rate']),
            b1=float(opt_params['b1']),
            b2=float(opt_params['b2']),
            weight_decay=float(opt_params['weight_decay']),
            eps=float(opt_params['eps'])
        )
    )

    epochs = maskbit_config['epochs']

    run = wandb.init(
        project=wandb_config['project'],
        name=wandb_config['name'],
        reinit=True,
        config=config
    )

    checkpoint_path = maskbit_config['checkpoint_path']
    vqmodel_checkpoint = vqmodel_config['checkpoint_path']
    vqmodel_params = load_checkpoint(vqmodel_checkpoint, None)['ema_params']

    inputs, labels = next(iter(train_loader))

    key = jax.random.PRNGKey(seed)
    key, sub_key = jax.random.split(key, 2)
    params = maskbit.init(sub_key, jnp.ones((inputs.shape[0], 256, 2)), jnp.array(labels),
                          jnp.full(inputs.shape[0], True))

    opt_state = optimizer.init(params)

    devices = jax.local_devices()
    replicate = lambda tree: jax.device_put_replicated(tree, devices)
    unreplicate = lambda tree: jax.tree_util.tree_map(lambda x: x[0], tree)

    ema_decay = maskbit_config['ema_decay']
    ema_params = params
    ema_params_repl = replicate(ema_params)

    update_fn = make_update_fn(
        vqmodel_apply_fn=vqmodel.apply,
        vqmodel_method=vqmodel.encode,
        maskbit_apply_fn=maskbit.apply,
        optimizer=optimizer,
        class_label_dropout=maskbit_config['class_label_dropout'],
        label_smoothing=maskbit_config['label_smoothing'],
        codebook_size=2 ** vqmodel_config['params']['token_size'],
        splits=maskbit_config['params']['splits'],
        mask_token=maskbit_config['params']['mask_token'],
        ema_decay=ema_decay,
    )

    params_repl = replicate(params)
    opt_state_repl = replicate(opt_state)
    vqmodel_params_repl = replicate(vqmodel_params)

    del vqmodel_checkpoint
    del params
    del opt_state
    del vqmodel_params
    del ema_params

    num_devices = jax.local_device_count()

    state_template = {
        "params": unreplicate(params_repl),
        "opt_state": unreplicate(opt_state_repl),
        "ema_params": unreplicate(ema_params_repl),
        "epoch": 0,
        "rng": key,
    }

    loaded_state = load_checkpoint(checkpoint_path, state_template)
    if loaded_state is not None:
        print("Resuming from checkpoint...")
        params_repl = replicate(loaded_state["params"])
        opt_state_repl = replicate(loaded_state["opt_state"])
        ema_params_repl = replicate(loaded_state["ema_params"])
        key = loaded_state["rng"]
        start_epoch = loaded_state["epoch"] + 1
    else:
        start_epoch = 0

    def shard(x):
        n, *s = x.shape
        return np.reshape(x, (num_devices, n // num_devices, *s))

    def unshard(inputs):
        num_devices, batch_size, *shape = inputs.shape
        return jnp.reshape(inputs, (num_devices * batch_size, *shape))

    for epoch in range(start_epoch, epochs):
        for step, (images, labels) in enumerate(train_loader):
            key, sample_rng = jax.random.split(key, 2)

            images = jax.tree_util.tree_map(lambda x: shard(np.array(x)), images)
            labels = jax.tree_util.tree_map(lambda x: shard(np.array(x)), labels)
            rng_shard = jax.random.split(sample_rng, num_devices)

            (
                params_repl,
                opt_state_repl,
                ema_params_repl,
                loss
            ) = update_fn(
                params_repl,
                opt_state_repl,
                vqmodel_params_repl,
                images,
                labels,
                rng_shard,
                ema_params_repl,
            )

            loss = unreplicate(loss)

            run.log({
                "total_loss": loss,
                "epoch": epoch})

        checkpoint_state = {
            "params": unreplicate(params_repl),
            "opt_state": unreplicate(opt_state_repl),
            "ema_params": unreplicate(ema_params_repl),
            "epoch": epoch,
            "rng": key,
        }
        save_checkpoint(checkpoint_path, checkpoint_state)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise ValueError('you must provide config file')
    main(sys.argv[1])
