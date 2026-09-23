"""Bitwise-preserving packed Metal recurrence for Kev-4B's DeltaNet.

Adapted from Apple MLX-LM PR #1559, merge e9308d7b0854105386a587db52c2f1a635196165,
under MIT. Eight value rows share one SIMD group. The old mlx-lm
kernel remains the path for masks and unsupported shapes.
"""

import mlx.core as mx
import mlx.nn as nn

from mlx.nn.layers.distributed import sum_gradients
from mlx_lm.models.gated_delta import compute_g, gated_delta_update as original_update
from mlx_lm.models.qwen3_5 import GatedDeltaNet


_kernel = mx.fast.metal_kernel(
    name="nerqova_packed_gated_delta_128",
    input_names=["q", "k", "v", "g", "beta", "state_in", "T", "prefix_tokens"],
    output_names=["y", "state_out"],
    source=r"""
        constexpr int lanes_per_row = 4;
        constexpr int rows_per_simdgroup = 32 / lanes_per_row;
        constexpr int values_per_lane = Dk / lanes_per_row;
        constexpr int partials_per_lane = values_per_lane / 4;

        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);

        auto lane = thread_index_in_simdgroup;
        auto row_in_simdgroup = lane / lanes_per_row;
        auto lane_in_row = lane & (lanes_per_row - 1);
        auto row_group = thread_position_in_grid.y;
        auto dv_idx = row_group * rows_per_simdgroup + row_in_simdgroup;

        auto q_ = q + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
        auto k_ = k + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
        auto v_ = v + (b_idx * T * Hv + hv_idx) * Dv;
        y += (b_idx * T * Hv + hv_idx) * Dv;

        auto i_state = state_in + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;
        float state[values_per_lane];
        for (int i = 0; i < values_per_lane; ++i) {
          state[i] = static_cast<float>(i_state[i]);
        }

        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          float gt = static_cast<float>(g_[hv_idx]);
          float part[partials_per_lane];
          for (int pb = 0; pb < partials_per_lane; ++pb) {
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
              int e = pb * 4 + i;
              state[e] = state[e] * gt;
              acc += state[e] * static_cast<float>(k_[e]);
            }
            part[pb] = acc;
          }
          float kv_mem =
              ((part[0] + part[1]) + (part[2] + part[3])) +
              ((part[4] + part[5]) + (part[6] + part[7]));
          kv_mem += simd_shuffle_xor(kv_mem, 1);
          kv_mem += simd_shuffle_xor(kv_mem, 2);

          auto delta =
              (static_cast<float>(v_[dv_idx]) - kv_mem) *
              static_cast<float>(beta_[hv_idx]);

          for (int pb = 0; pb < partials_per_lane; ++pb) {
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
              int e = pb * 4 + i;
              state[e] = state[e] + static_cast<float>(k_[e]) * delta;
              acc += state[e] * static_cast<float>(q_[e]);
            }
            part[pb] = acc;
          }
          float out =
              ((part[0] + part[1]) + (part[2] + part[3])) +
              ((part[4] + part[5]) + (part[6] + part[7]));
          out += simd_shuffle_xor(out, 1);
          out += simd_shuffle_xor(out, 2);
          if (lane_in_row == 0) {
            y[dv_idx] = static_cast<InT>(out);
          }

          if (CapturePrefix && t + 1 == prefix_tokens) {
            for (int i = 0; i < values_per_lane; ++i) {
              o_state[i] = static_cast<StT>(state[i]);
            }
          }

          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }

        if (!CapturePrefix) {
          for (int i = 0; i < values_per_lane; ++i) {
            o_state[i] = static_cast<StT>(state[i]);
          }
        }
    """,
)


def packed_kernel(q, k, v, g, beta, state, prefix_tokens=0):
    batch, tokens, key_heads, key_width = q.shape
    value_heads, value_width = v.shape[2:]
    if not 0 <= prefix_tokens <= tokens:
        raise ValueError("prefix snapshot must lie within the token sequence")
    if ((key_heads, value_heads, key_width, value_width) != (16, 32, 128, 128)
            or k.shape != q.shape or v.shape[:2] != (batch, tokens)
            or g.shape != (batch, tokens, value_heads)
            or beta.shape != g.shape
            or state.shape != (batch, value_heads, value_width, key_width)
            or k.dtype != q.dtype or v.dtype != q.dtype or beta.dtype != q.dtype
            or g.dtype != mx.float32 or state.dtype != mx.float32):
        raise ValueError("packed DeltaNet needs Kev-4B's unmasked scalar-gate shapes")
    return _kernel(
        inputs=[q, k, v, g, beta, state, tokens, prefix_tokens],
        template=[("InT", q.dtype), ("StT", state.dtype), ("Dk", key_width),
                  ("Dv", value_width), ("Hk", key_heads), ("Hv", value_heads),
                  ("CapturePrefix", bool(prefix_tokens))],
        grid=(32, value_width // 8, batch * value_heads),
        threadgroup=(32, 2, 1),
        output_shapes=[v.shape, state.shape],
        output_dtypes=[q.dtype, state.dtype],
    )


def packed_update(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True, prefix_tokens=0):
    if prefix_tokens and (mask is not None or not use_kernel):
        raise ValueError("prefix snapshot requires the unmasked Metal kernel")
    if mask is not None or not use_kernel:
        return original_update(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel)
    beta = mx.sigmoid(b)
    gamma = compute_g(A_log, a, dt_bias)
    if state is None:
        batch, _, _, key_width = q.shape
        value_heads, value_width = v.shape[2:]
        state = mx.zeros((batch, value_heads, value_width, key_width), dtype=mx.float32)
    return packed_kernel(q, k, v, gamma, beta, state, prefix_tokens)


class PackedGatedDeltaNet(GatedDeltaNet):
    """Qwen3.5 DeltaNet with the packed recurrence and unchanged projections."""

    def __call__(self, inputs, mask=None, cache=None):
        batch, tokens, _ = inputs.shape
        prefix_tokens = getattr(cache, "capture_prefix_tokens", 0)
        if self.sharding_group is not None:
            inputs = sum_gradients(self.sharding_group)(inputs)

        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(
            batch, tokens, self.num_v_heads, self.head_v_dim
        )
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (batch, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype
            )
        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, tokens)
                positions = (ends[:, None] + mx.arange(keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(
                    conv_input[:, prefix_tokens:prefix_tokens + keep, :]
                    if prefix_tokens else conv_input[:, -keep:, :]
                )
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            value.reshape(batch, tokens, heads, width)
            for value, heads, width in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]
        state = cache[1] if cache else None
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        out, state = packed_update(
            q, k, v, a, b, self.A_log, self.dt_bias, state, mask,
            use_kernel=not self.training, prefix_tokens=prefix_tokens,
        )
        if cache is not None:
            cache[1] = state
            cache.advance(tokens)
        out = self.norm(out, z)
        out = self.out_proj(out.reshape(batch, tokens, -1))
        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)
        return out
