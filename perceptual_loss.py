import jax
import jax.numpy as jnp

import flax.linen as nn
import flaxmodels as fm


class PerceptualLoss(nn.Module):

    def setup(self):
        self.model = fm.ResNet50(output='logits', pretrained='imagenet')

        self.mean = jnp.array([0.485, 0.456, 0.406])[None, None, None, :]
        self.std = jnp.array([0.229, 0.224, 0.225])[None, None, None, :]

    def __call__(self, inputs, targets):
        inputs = inputs * self.std + self.mean
        targets = targets * self.std + self.mean

        inputs = jax.image.resize(inputs, (inputs.shape[0], 224, 224, inputs.shape[-1]), method="bilinear")
        targets = jax.image.resize(targets, (targets.shape[0], 224, 224, targets.shape[-1]), method="bilinear")

        features_inputs = self.model(inputs, train=False)
        features_targets = self.model(targets, train=False)

        loss = jnp.mean((features_inputs - features_targets) ** 2)
        return loss
