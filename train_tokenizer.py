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
from discriminator import Discriminator
from perceptual_loss import PerceptualLoss

import numpy as np


def save_checkpoint(path, state):
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(state))


def load_checkpoint(path, state_template):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return serialization.from_bytes(state_template, f.read())


def adopt_weight(step, threshold, value=0.0):
    return jnp.where(step < threshold, value, 1.0)


def ema_update(ema_params, new_params, decay):
    return jax.tree_util.tree_map(
        lambda e, p: decay * e + (1.0 - decay) * p,
        ema_params,
        new_params
    )


def make_generator_update_fn(*, vqmodel_apply_fn, vqmodel_optimizer,
                             disc_apply_fn, perceptual_apply_fn,
                             reconstruction_weight, perceptual_weight, discriminator_weight,
                             entropy_annealing_factor, quantizer_weight, ema_decay, disc_start):
    def update_fn(vqmodel_params, vqmodel_opt_state,
                  disc_params, perceptual_params, images, ema_params, annealing_steps, global_step):
        def loss_fn(params):
            recontructed_images, extra_result_dict = vqmodel_apply_fn(params, images)

            reconstruction_loss = jnp.mean((recontructed_images - images) ** 2)
            reconstruction_loss *= reconstruction_weight

            perceptual_loss = perceptual_apply_fn(perceptual_params, recontructed_images, images)

            generator_loss = jnp.zeros(())

            discriminator_factor = adopt_weight(global_step, disc_start)

            d_weight = 1.0

            logits_fake = disc_apply_fn(disc_params, recontructed_images)
            generator_loss = -jnp.mean(logits_fake)

            d_weight *= discriminator_weight

            quantizer_loss = extra_result_dict['quantizer_loss']

            quantizer_loss += (
                annealing_steps * entropy_annealing_factor * extra_result_dict["entropy_loss"]
            )

            total_loss = (
                    reconstruction_loss
                    + perceptual_weight * perceptual_loss
                    + quantizer_weight * quantizer_loss
                    + d_weight * discriminator_factor * generator_loss
            )

            loss_dict = dict(
                total_loss=total_loss,
                recontructed_images=recontructed_images,
                reconstruction_loss=reconstruction_loss,
                perceptual_loss=(perceptual_weight * perceptual_loss),
                quantizer_loss=(quantizer_weight * quantizer_loss),
                weighted_gan_loss=(d_weight * discriminator_factor * generator_loss),
                discriminator_factor=discriminator_factor,
                commitment_loss=extra_result_dict["commitment_loss"],
                entropy_loss=extra_result_dict["entropy_loss"],
                per_sample_entropy=extra_result_dict["per_sample_entropy"],
                avg_entropy=extra_result_dict["avg_entropy"],
                d_weight=d_weight,
                gan_loss=generator_loss,
            )

            return total_loss, loss_dict

        ((loss, loss_dict), grad) = jax.value_and_grad(loss_fn, has_aux=True)(vqmodel_params)  # watch out

        loss, grad = jax.tree_util.tree_map(
            lambda x: jax.lax.pmean(x, axis_name='batch'),
            (loss, grad),
        )

        updates, opt_state = vqmodel_optimizer.update(grad, vqmodel_opt_state, vqmodel_params)
        new_params = optax.apply_updates(vqmodel_params, updates)
        new_ema_params = ema_update(ema_params, new_params, decay=ema_decay)

        return new_params, opt_state, new_ema_params, loss, loss_dict

    return jax.pmap(update_fn, axis_name='batch', donate_argnums=())


def make_disc_update_fn(*, apply_fn, optimizer, lecam_regularization_weight, ema_decay, disc_start):
    def update_fn(params, opt_state, real_images, fake_images, ema_real, ema_fake, global_step):
        def loss_fn(params):
            disc_factor = adopt_weight(global_step, disc_start)

            logits_real = apply_fn(params, real_images)
            logits_fake = apply_fn(params, fake_images)

            loss_real = jnp.mean(jax.nn.relu(1.0 - logits_real))
            loss_fake = jnp.mean(jax.nn.relu(1.0 + logits_fake))

            discriminator_loss = disc_factor * 0.5 * (loss_real + loss_fake)

            lecam_loss = jnp.zeros(())

            lecam_loss = jnp.mean(jnp.pow(jax.nn.relu(logits_real - ema_fake), 2))
            lecam_loss += jnp.mean(jnp.pow(jax.nn.relu(ema_real - logits_fake), 2))
            lecam_loss *= lecam_regularization_weight

            new_ema_real = ema_real * ema_decay + jnp.mean(logits_real) * (1 - ema_decay)
            new_ema_fake = ema_fake * ema_decay + jnp.mean(logits_fake) * (1 - ema_decay)

            discriminator_loss += lecam_loss

            loss_dict = dict(
                discriminator_loss=discriminator_loss,
                logits_real=logits_real,
                logits_fake=logits_fake,
                lecam_loss=lecam_loss,
                ema_real_logits=new_ema_real,
                ema_fake_logits=new_ema_fake,
            )

            return discriminator_loss, loss_dict

        (loss, loss_dict), grad = jax.value_and_grad(loss_fn, has_aux=True)(params)
        loss, grad = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, axis_name='batch'), (loss, grad))
        updates, opt_state = optimizer.update(grad, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return new_params, opt_state, loss, loss_dict

    return jax.pmap(update_fn, axis_name='batch', donate_argnums=())


def main(config_path):
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    vqmodel_config = config['model']
    disc_config = config['discriminator']
    perceptual_config = config['perceptual']
    dataset_params = config['dataset_params']
    wandb_config = config['wandb']

    seed = vqmodel_config['seed']

    transform = transforms.Compose([
        transforms.RandomResizedCrop(
            size=256,
            scale=(0.8, 1.0),
            ratio=(3/4, 4/3),
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(), # Normalize [0, 1]
        transforms.Lambda(lambda t: (t * 2) - 1),  # Scale [-1, 1]
        transforms.Lambda(lambda x: x.permute(1, 2, 0)),  # Convert [C, H, W] to [H, W, C]
    ])

    if dataset_params['dataset'] == 'imagenet':
        train_dataset = ImageFolder(
            root=dataset_params['data_path'],
            transform=transform,
        )
    else:
        raise 'There is no such dataset'

    train_loader = DataLoader(
        train_dataset,
        batch_size=dataset_params['batch_size'],
        shuffle=True,
        num_workers=dataset_params['num_workers'],
        pin_memory=True,
        drop_last=True,
    )

    vqmodel = ConvVQModel(**vqmodel_config['params'])
    disc = Discriminator(**disc_config['params'])

    perceptual = PerceptualLoss()

    vqmodel_optimizer = optax.chain(
        optax.adamw(**vqmodel_config['optim_params'])
    )

    disc_optimizer = optax.chain(
        optax.adam(**disc_config['optim_params'])
    )

    epochs = vqmodel_config['epochs']

    run = wandb.init(
        project=wandb_config['project'],
        name=wandb_config['name'],
        reinit=True,
        config=config
    )

    checkpoint_path = vqmodel_config['checkpoint_path']

    inputs = next(iter(train_loader))

    key = jax.random.PRNGKey(seed)
    vqmodel_params = vqmodel.init(key, np.array(inputs))
    disc_params = disc.init(key, np.array(inputs))
    perceptual_params = perceptual.init(key, jnp.ones((2, 256, 256, 3)), jnp.ones((2, 256, 256, 3)))

    vqmodel_opt_state = vqmodel_optimizer.init(vqmodel_params)
    disc_opt_state = disc_optimizer.init(disc_params)

    replicate = lambda tree: jax.device_put_replicated(tree, jax.local_devices())
    unreplicate = lambda tree: jax.tree_util.tree_map(lambda x: x[0], tree)

    ema_params = vqmodel_params
    ema_params_repl = replicate(ema_params)

    generator_update_fn = make_generator_update_fn(
        vqmodel_apply_fn=vqmodel.apply,
        vqmodel_optimizer=vqmodel_optimizer,
        disc_apply_fn=disc.apply,
        perceptual_apply_fn=perceptual.apply,
        reconstruction_weight=vqmodel_config['reconstrunction_weight'],
        perceptual_weight=perceptual_config['perceptual_weight'],
        discriminator_weight=disc_config['discriminator_weight'],
        entropy_annealing_factor=vqmodel_config['entropy_annealing_factor'],
        quantizer_weight=vqmodel_config['quantizer_weight'],
        ema_decay=vqmodel_config['ema_decay'],
        disc_start=disc_config['disc_start']
    )

    disc_update_fn = make_disc_update_fn(
        apply_fn=disc.apply,
        optimizer=disc_optimizer,
        lecam_regularization_weight=disc_config['lecam_regularization_weight'],
        ema_decay=vqmodel_config['ema_decay'],
        disc_start=disc_config['disc_start'],
    )

    vqmodel_params_repl = replicate(vqmodel_params)
    vqmodel_opt_state_repl = replicate(vqmodel_opt_state)
    disc_params_repl = replicate(disc_params)
    disc_opt_state_repl = replicate(disc_opt_state)
    perceptual_params_repl = replicate(perceptual_params)

    ema_real = jnp.zeros((1,))
    ema_fake = jnp.zeros((1,))

    ema_real_repl = replicate(ema_real)
    ema_fake_repl = replicate(ema_fake)

    state_template = {
        "params": unreplicate(vqmodel_params_repl),
        "opt_state": unreplicate(vqmodel_opt_state_repl),
        "ema_params": unreplicate(ema_params_repl),
        'disc_params': unreplicate(disc_params_repl),
        'disc_opt_state': unreplicate(disc_opt_state_repl),
        "epoch": 0,
    }

    del vqmodel_params
    del vqmodel_opt_state
    del disc_params
    del disc_opt_state
    del perceptual_params
    del ema_params

    loaded_state = load_checkpoint(checkpoint_path, state_template)
    start_epoch = 0
    if loaded_state:
        vqmodel_params_repl = replicate(loaded_state['params'])
        vqmodel_opt_state_repl = replicate(loaded_state['opt_state'])
        ema_params_repl = replicate(loaded_state['ema_params'])
        disc_params_repl = replicate(loaded_state['disc_params'])
        disc_opt_state_repl = replicate(loaded_state['disc_opt_state'])
        start_epoch = loaded_state['epoch'] + 1

    def shard(x):
        n, *s = x.shape
        return np.reshape(x, (jax.local_device_count(), n // jax.local_device_count(), *s))

    def unshard(x):
        ndev, bs, *s = x.shape
        return jnp.reshape(x, (ndev * bs, *s))

    global_step = 0
    global_step_repl = jnp.array([global_step] * jax.local_device_count())

    entropy_annealing_steps = vqmodel_config['entropy_annealing_steps']

    for epoch in range(start_epoch, epochs):
        for step, images in enumerate(train_loader):
            images = jax.tree_util.tree_map(lambda x: shard(np.array(x)), images[0])

            annealing_steps = max(0.0, 1 - global_step / entropy_annealing_steps)
            annealing_steps_repl = jnp.array([annealing_steps] * jax.local_device_count())

            (
                vqmodel_params_repl,
                vqmodel_opt_state_repl,
                ema_params_repl,
                loss,
                generator_loss_dict,
            ) = generator_update_fn(
                vqmodel_params_repl,
                vqmodel_opt_state_repl,
                disc_params_repl,
                perceptual_params_repl,
                images,
                ema_params_repl,
                annealing_steps_repl,
                global_step_repl
            )

            (
                disc_params_repl,
                disc_opt_state_repl,
                disc_loss,
                discriminator_loss_dict,
            ) = disc_update_fn(
                disc_params_repl,
                disc_opt_state_repl,
                images,
                generator_loss_dict['recontructed_images'],
                ema_real_repl,
                ema_fake_repl,
                global_step_repl
            )

            ema_real_repl = discriminator_loss_dict['ema_real_logits']
            ema_fake_repl = discriminator_loss_dict['ema_fake_logits']

            loss = unreplicate(generator_loss_dict['total_loss'])

            if global_step % 1000 == 0:
                import matplotlib.pyplot as plt
                import io
                from PIL import Image as PILImage

                def to_numpy_img(img):
                    img = (img + 1) / 2
                    img = np.clip(np.array(img), 0.0, 1.0)
                    return (img * 255).astype(np.uint8)

                real = to_numpy_img(unshard(images)[0])
                recon = to_numpy_img(unshard(generator_loss_dict['recontructed_images'])[0])

                fig, axs = plt.subplots(1, 2, figsize=(4, 2))
                axs[0].imshow(real)
                axs[0].set_title("Original")
                axs[0].axis("off")
                axs[1].imshow(recon)
                axs[1].set_title("Reconstruction")
                axs[1].axis("off")
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                plt.close(fig)

                run.log({"reconstruction": wandb.Image(PILImage.open(buf))}, step=global_step)

            run.log({
                "reconstruct_loss": unreplicate(generator_loss_dict['reconstruction_loss']),
                "perceptual_loss": unreplicate(generator_loss_dict['perceptual_loss']),
                "quantizer_loss": unreplicate(generator_loss_dict['quantizer_loss']),
                "weighted_generator_loss": unreplicate(generator_loss_dict['weighted_gan_loss']),
                "generator_loss": unreplicate(generator_loss_dict['gan_loss']),
                "entropy_loss": unreplicate(generator_loss_dict['entropy_loss']),
                "commitment_loss": unreplicate(generator_loss_dict['commitment_loss']),
                "discriminator_loss": unreplicate(discriminator_loss_dict['discriminator_loss']),
                "lecam_loss": unreplicate(discriminator_loss_dict['lecam_loss']),
                'disc_loss': unreplicate(disc_loss),
                "total_loss": loss,
                "epoch": epoch,
            })

            global_step += 1
            global_step_repl = jnp.array([global_step] * jax.local_device_count())

        save_checkpoint(checkpoint_path, {
            "params": unreplicate(vqmodel_params_repl),
            "opt_state": unreplicate(vqmodel_opt_state_repl),
            "ema_params": unreplicate(ema_params_repl),
            'disc_params': unreplicate(disc_params_repl),
            'disc_opt_state': unreplicate(disc_opt_state_repl),
            "epoch": epoch,
        })


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise ValueError('you must provide config file')
    main(sys.argv[1])
