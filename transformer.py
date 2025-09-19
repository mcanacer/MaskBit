import jax
import jax.numpy as jnp

import flax.linen as nn

import math

LAYERNORM_EPSILON = 1e-12


def truncated_normal(stddev, dtype=jnp.float32):
    def init(key, shape, dtype=dtype):
        return jax.random.truncated_normal(
            key=key, lower=-2, upper=2, shape=shape, dtype=dtype) * stddev
    return init


class BertAttention(nn.Module):
    hidden_dim: int
    dropout: float
    num_heads: int

    @nn.compact
    def __call__(self, layer_input, input_mask, train=True):
        attention_mask = nn.make_attention_mask(input_mask, input_mask)
        attention_output = nn.attention.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            dropout_rate=self.dropout,
            deterministic=not train,
            kernel_init=truncated_normal(0.02),
            bias_init=jax.nn.initializers.zeros,
            name='self_attention',
        )(layer_input, attention_mask)

        attention_output = nn.Dropout(rate=self.dropout)(
            attention_output, deterministic=not train)
        attention_output = nn.LayerNorm(
            epsilon=LAYERNORM_EPSILON, name='attention_output_ln')(
            attention_output + layer_input)

        return attention_output


class BertFeedForward(nn.Module):
    hidden_dim: int
    dropout: float

    @nn.compact
    def __call__(self, x, train=True):
        residual = x
        x = nn.Dense(int(4 * self.hidden_dim), kernel_init=truncated_normal(0.02))(x)
        x = jax.nn.gelu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=truncated_normal(0.02))(x)
        x = nn.Dropout(self.dropout)(x, deterministic=not train)
        x = nn.LayerNorm(epsilon=LAYERNORM_EPSILON, name='MLP_ln')(x + residual)
        return x


class TransformerEncoder(nn.Module):
    hidden_dim: int
    depth: int
    dropout: float
    num_heads: int

    @nn.compact
    def __call__(self, x, train=True):
        for _ in range(self.depth):
            x = BertAttention(
                hidden_dim=self.hidden_dim,
                dropout=self.dropout,
                num_heads=self.num_heads,
            )(x, input_mask=jnp.ones(x.shape[:2], dtype=jnp.int32), train=train)
            x = BertFeedForward(
                hidden_dim=self.hidden_dim,
                dropout=self.dropout,
            )(x, train=train)

        return x


class LFQBert(nn.Module):
    hidden_dim: int
    depth: int
    dropout: float
    num_heads: int
    mask_token: int
    drop_label: int
    num_classes: int
    effective_codebook_size: int
    splits: int

    @nn.compact
    def __call__(self, img_tokens, class_labels, drop_label_mask, train=True):
        N, L = img_tokens.shape[:2]

        img_tokens = self.preprocess_tokens(img_tokens)

        cls_token = class_labels.reshape(N, -1)
        drop_label_mask = drop_label_mask.reshape(N, -1)

        class_token = drop_label_mask * self.drop_label + (~drop_label_mask) * cls_token
        cls_embedding = nn.Embed(self.num_classes + 1, self.hidden_dim,
                                 embedding_init=truncated_normal(0.02))(class_token)

        projected_bit_tokens = nn.Dense(self.hidden_dim, kernel_init=truncated_normal(0.02))(img_tokens)

        projected_bit_tokens = jnp.concatenate([projected_bit_tokens, cls_embedding], axis=1)

        pos_ids = jnp.expand_dims(jnp.arange(L + 1), axis=0)
        pos_embed = nn.Embed(L + 1, self.hidden_dim, embedding_init=truncated_normal(0.02))(pos_ids)
        x = projected_bit_tokens + pos_embed

        x = nn.LayerNorm(epsilon=LAYERNORM_EPSILON)(x)
        x = nn.Dropout(self.dropout)(x, deterministic=not train)

        x = TransformerEncoder(
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            dropout=self.dropout,
            num_heads=self.num_heads,
        )(x, train=train)

        x = nn.Dense(self.hidden_dim, kernel_init=truncated_normal(0.02))(x)
        x = jax.nn.gelu(x)
        x = nn.LayerNorm(epsilon=LAYERNORM_EPSILON)(x)

        x = nn.Dense(self.splits * self.effective_codebook_size, kernel_init=truncated_normal(0.02))(x)

        logits = jnp.reshape(x, (N, L + 1, self.splits, self.effective_codebook_size))
        logits = logits[:, :L, ...]

        return logits

    def preprocess_tokens(self, img_tokens):
        N, L = img_tokens.shape[:2]
        mask = img_tokens == self.mask_token
        bits_to_indices = jnp.pow(2.0, jnp.arange(0, math.log2(self.effective_codebook_size)))
        token_as_bits = ((img_tokens[..., None].astype(jnp.int32) & bits_to_indices.astype(jnp.int32)) != 0).astype(jnp.float32)
        token_as_bits = token_as_bits * 2.0 - 1.0

        mask_expanded = jnp.expand_dims(mask, axis=-1)
        token_as_bits = jnp.where(mask_expanded, 0.0, token_as_bits)
        token_as_bits = jnp.reshape(token_as_bits, (N, L, -1))
        return token_as_bits
