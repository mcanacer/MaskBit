from typing import Sequence

import jax
import jax.numpy as jnp

import flax.linen as nn

from utils import entropy_loss_fn


class ResidualBlock(nn.Module):
    filters: int

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.GroupNorm(num_groups=32)(x)
        x = jax.nn.silu(x)
        x = nn.Conv(features=self.filters, kernel_size=(3, 3), use_bias=False)(x)

        x = nn.GroupNorm(num_groups=32)(x)
        x = jax.nn.silu(x)
        x = nn.Conv(features=self.filters, kernel_size=(3, 3), use_bias=False)(x)

        if residual.shape[-1] != self.filters:
            residual = nn.Conv(features=self.filters, kernel_size=(1, 1), use_bias=False)(x)

        return x + residual


class ResidualStage(nn.Module):
    filters: int
    num_res_block: int

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_res_block):
            x = ResidualBlock(filters=self.filters)(x)

        return x


class DownsamplingStage(nn.Module):
    filters: int
    num_res_block: int

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_res_block):
            x = ResidualBlock(filters=self.filters)(x)

        x = nn.Conv(features=self.filters, kernel_size=(3, 3), strides=2)(x)
        return x


class UpsamplingStage(nn.Module):
    filters: int
    num_res_block: int
    factor: int = 2
    method: str = 'nearest'

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_res_block):
            x = ResidualBlock(filters=self.filters)(x)

        N, H, W, C = x.shape
        x = jax.image.resize(x, shape=(N, H * self.factor, W * self.factor, C), method=self.method)
        x = nn.Conv(features=self.filters, kernel_size=(3, 3))(x)
        return x


class ConvEncoder(nn.Module):
    filters: int
    token_size: int
    channel_multipliers: Sequence[int]
    num_res_block: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.filters, kernel_size=(3, 3), use_bias=False)(x)
        num_blocks = len(self.channel_multipliers)

        for i in range(num_blocks):
            filters = self.filters * self.channel_multipliers[i]

            if i < num_blocks - 1:
                x = DownsamplingStage(filters, self.num_res_block)(x)
            else:
                x = ResidualStage(filters, self.num_res_block)(x)

        x = ResidualStage(filters, self.num_res_block)(x)

        x = nn.GroupNorm(num_groups=32)(x)
        x = jax.nn.silu(x)
        x = nn.Conv(self.token_size, kernel_size=(1, 1))(x)

        return x


class ConvDecoder(nn.Module):
    filters: int
    token_size: int
    channel_multipliers: Sequence[int]
    num_res_block: int

    @nn.compact
    def __call__(self, x):
        filters = self.filters * self.channel_multipliers[-1]
        x = nn.Conv(features=filters, kernel_size=(3, 3))(x)

        x = ResidualStage(filters, self.num_res_block)(x)

        num_blocks = len(self.channel_multipliers)
        for i in reversed(range(num_blocks)):
            filters = self.filters * self.channel_multipliers[i]
            if i > 0:
                x = UpsamplingStage(filters, self.num_res_block)(x)
            else:
                x = ResidualStage(filters, self.num_res_block)(x)

        x = nn.GroupNorm(num_groups=32)(x)
        x = jax.nn.silu(x)
        x = nn.Conv(self.token_size, kernel_size=(3, 3))(x)

        return x


class LookupFreeQuantizer(nn.Module):
    token_bits: int = 10,
    commitment_cost: float = 0.25,
    entropy_loss_weight: float = 0.1,
    entropy_loss_temperature: float = 0.01,
    entropy_gamma: float = 1.0,

    def setup(self):
        self.codebook_size = 2 ** self.token_bits

        self.bits_to_indices = (2 ** jnp.arange(0, self.token_bits, dtype=jnp.float32)).astype(jnp.int32)

        all_codes = jnp.arange(self.codebook_size)
        bits = ((all_codes[..., None].astype(jnp.int32) & self.bits_to_indices) != 0).astype(jnp.float32)
        self.codebook = bits * 2.0 - 1.0

    def __call__(self, z, train=True):
        ones = jnp.ones_like(z)
        sign_mask = (z > 0.0)
        z_quantized = jnp.where(sign_mask, ones, -ones)

        min_encoding_indices = self.convert_bits_to_indices(z_quantized)

        commitment_loss = self.commitment_cost * jnp.mean((jax.lax.stop_gradient(z_quantized) - z) ** 2)
        entropy_loss = jnp.zeros(())
        per_sample_entropy = jnp.zeros(())
        avg_entropy = jnp.zeros(())

        if self.entropy_loss_weight != 0.0 and train:
            d = - 2 * z @ self.codebook.T

            per_sample_entropy, avg_entropy = entropy_loss_fn(-1*d, self.entropy_loss_temperature, self.entropy_gamma)
            entropy_loss = self.entropy_loss_weight * (per_sample_entropy - avg_entropy)

        loss = commitment_loss + entropy_loss

        z_quantized = z + jax.lax.stop_gradient(z_quantized - z)

        result_dict = dict(
            quantizer_loss=loss,
            commitment_loss=commitment_loss,
            entropy_loss=entropy_loss,
            per_sample_entropy=per_sample_entropy,
            avg_entropy=avg_entropy,
            min_encoding_indices=min_encoding_indices,
        )

        return z_quantized, result_dict

    def convert_bits_to_indices(self, tokens):
        sign_mask = (tokens > 0.0).astype(jnp.int32)
        bits_to_indices = jnp.pow(2.0, jnp.arange(0, self.token_bits, dtype=jnp.float32)).astype(jnp.int32)
        return jnp.sum(sign_mask * bits_to_indices, axis=-1)


class ConvVQModel(nn.Module):
    token_size: int
    commitment_cost: float
    entropy_loss_weight: float
    entropy_loss_temperature: float
    entropy_gamma: float
    filters: int
    channel_multipliers: Sequence[int]
    num_res_blocks: int

    def setup(self):
        self.encoder = ConvEncoder(
            filters=self.filters,
            token_size=self.token_size,
            channel_multipliers=self.channel_multipliers,
            num_res_block=self.num_res_blocks,
        )
        self.quantizer = LookupFreeQuantizer(
            token_bits=self.token_size,
            commitment_cost=self.commitment_cost,
            entropy_loss_weight=self.entropy_loss_weight,
            entropy_loss_temperature=self.entropy_loss_temperature,
            entropy_gamma=self.entropy_gamma,
        )
        self.decoder = ConvDecoder(
            filters=self.filters,
            token_size=self.token_size,
            channel_multipliers=self.channel_multipliers,
            num_res_block=self.num_res_blocks,
        )

    def __call__(self, x, train=True):
        z_quantized, result_dict = self.encode(x, train=train)
        decoded = self.decode(z_quantized)
        return decoded, result_dict

    def encode(self, x, train=True):
        z = self.encoder(x)
        z_quantized, result_dict = self.quantizer(z, train=train)
        return z_quantized, result_dict

    def decode(self, x):
        decoded = self.decoder(x)
        return decoded
