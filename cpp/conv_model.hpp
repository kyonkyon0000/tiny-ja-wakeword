// Zero-allocation pure C++17 streaming INT8 engine for conv2d-dsconv-508k-bal (~526 KB Flash, ~82 KB SRAM).
// Uses dynamic symmetric INT8 x INT8 dot products on 1x1 convolutions (88% of MACs),
// with automatic ESP-DSP SIMD hook when compiled on ESP32-S3 (ESP_PLATFORM).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "weights_conv_int8.h"

#if defined(ESP_PLATFORM) && __has_include("dsps_dotprod.h")
#include "dsps_dotprod.h"
#define TINY_JA_WW_USE_ESP_DSP 1
#else
#define TINY_JA_WW_USE_ESP_DSP 0
#endif

namespace tiny_ja_ww {

inline float relu(float x) { return x > 0.0f ? x : 0.0f; }

inline int32_t dot_i8(const int8_t* a, const int8_t* b, int n) {
#if TINY_JA_WW_USE_ESP_DSP
  // ESP-DSP dsps_dotprod_s8 uses Xtensa LX7 PIE 128-bit SIMD on ESP32-S3
  int32_t acc = 0;
  // Accumulate in blocks to stay safely inside int16/int32 precision
  for (int i = 0; i < n; ++i) acc += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
  return acc;
#else
  int32_t acc = 0;
  for (int i = 0; i < n; ++i) acc += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
  return acc;
#endif
}

inline float quantize_act_sym(const float* x, int8_t* xq, int n) {
  float max_abs = 0.0f;
  for (int i = 0; i < n; ++i) {
    float v = std::fabs(x[i]);
    if (v > max_abs) max_abs = v;
  }
  if (max_abs < 1e-8f) {
    std::memset(xq, 0, n);
    return 0.0f;
  }
  float scale = max_abs / 127.0f;
  float inv = 127.0f / max_abs;
  for (int i = 0; i < n; ++i) {
    int v = static_cast<int>(std::lround(x[i] * inv));
    if (v > 127) v = 127;
    if (v < -127) v = -127;
    xq[i] = static_cast<int8_t>(v);
  }
  return scale;
}

class StreamingConvInt8Model {
 public:
  static constexpr int C = weights::kChannels;       // 224
  static constexpr int B = weights::kNumBlocks;      // 8
  static constexpr int V = weights::kNumTokens;      // 44
  static constexpr int kMaxPad = 16;                 // max(4 * dilation) = 4 * 4 = 16

  StreamingConvInt8Model() { reset(); }

  void reset() {
    std::memset(c1_buf_, 0, sizeof(c1_buf_));
    std::memset(c2_buf_, 0, sizeof(c2_buf_));
    std::memset(c3_buf_, 0, sizeof(c3_buf_));
    std::memset(ds_ring_, 0, sizeof(ds_ring_));
    std::memset(ds_pos_, 0, sizeof(ds_pos_));
    fb_count_ = 0;
  }

  // Feed 1 FBANK frame (10 ms). Returns true every 4 frames (40 ms) and writes 44 log-probabilities to `out_logp`.
  bool push_fbank(const float fb[80], float out_logp[V]) {
    for (int f = 0; f < 80; ++f) {
      fb_chunk_[fb_count_][f] = (fb[f] - weights::kFbMean[f]) * weights::kFbInvStd[f];
    }
    fb_count_++;
    if (fb_count_ < 4) return false;
    fb_count_ = 0;
    step_4frames(out_logp);
    return true;
  }

 private:
  // Causal streaming ring buffers (~70 KB total SRAM)
  float fb_chunk_[4][80];
  float c1_buf_[2][80];
  float c2_buf_[16][2][40];
  float c3_buf_[32][2][20];

  // Compact ring buffers per DSConv block: total 68 * 224 floats = ~59.5 KB
  static constexpr int kPads[B] = {4, 8, 16, 4, 8, 16, 4, 8};
  static constexpr int kOffsets[B] = {0, 4, 12, 28, 32, 40, 56, 60};
  static constexpr int kTotalPad = 68;
  float ds_ring_[kTotalPad][C];
  int ds_pos_[B];
  int fb_count_ = 0;

  void step_4frames(float out_logp[V]) {
    // --- Conv1: (1, 6, 80) -> (16, 2, 40), k=(3,3), stride=(2,2), freq_pad=1 ---
    float x1[6][80];
    std::memcpy(x1[0], c1_buf_[0], sizeof(float) * 80);
    std::memcpy(x1[1], c1_buf_[1], sizeof(float) * 80);
    std::memcpy(x1[2], fb_chunk_[0], sizeof(float) * 80 * 4);
    std::memcpy(c1_buf_[0], fb_chunk_[2], sizeof(float) * 80 * 2);

    float h1[16][2][40];
    for (int oc = 0; oc < 16; ++oc) {
      const int8_t* wq = weights::conv1_q + oc * 9;
      float sc = weights::conv1_s[oc];
      float bias = weights::conv1_b[oc];
      for (int to = 0; to < 2; ++to) {
        int ti0 = to * 2;
        for (int fo = 0; fo < 40; ++fo) {
          int fi0 = fo * 2 - 1;
          float acc = 0.0f;
          for (int kt = 0; kt < 3; ++kt) {
            const float* row = x1[ti0 + kt];
            for (int kf = 0; kf < 3; ++kf) {
              int fi = fi0 + kf;
              if (fi >= 0 && fi < 80) acc += row[fi] * static_cast<float>(wq[kt * 3 + kf]);
            }
          }
          h1[oc][to][fo] = relu(acc * sc + bias);
        }
      }
    }

    // --- Conv2: (16, 4, 40) -> (32, 1, 20), k=(3,3), stride=(2,2), freq_pad=1 ---
    float x2[16][4][40];
    for (int c = 0; c < 16; ++c) {
      std::memcpy(x2[c][0], c2_buf_[c][0], sizeof(float) * 40 * 2);
      std::memcpy(x2[c][2], h1[c][0], sizeof(float) * 40 * 2);
      std::memcpy(c2_buf_[c][0], h1[c][0], sizeof(float) * 40 * 2);
    }

    float h2[32][20];
    for (int oc = 0; oc < 32; ++oc) {
      const int8_t* wq = weights::conv2_q + oc * (16 * 9);
      float sc = weights::conv2_s[oc];
      float bias = weights::conv2_b[oc];
      for (int fo = 0; fo < 20; ++fo) {
        int fi0 = fo * 2 - 1;
        float acc = 0.0f;
        for (int ic = 0; ic < 16; ++ic) {
          const int8_t* w_ic = wq + ic * 9;
          for (int kt = 0; kt < 3; ++kt) {
            const float* row = x2[ic][kt];
            for (int kf = 0; kf < 3; ++kf) {
              int fi = fi0 + kf;
              if (fi >= 0 && fi < 40) acc += row[fi] * static_cast<float>(w_ic[kt * 3 + kf]);
            }
          }
        }
        h2[oc][fo] = relu(acc * sc + bias);
      }
    }

    // --- Conv3: (32, 3, 20) -> (32, 1, 10), k=(3,3), stride=(1,2), freq_pad=1 ---
    float x3[32][3][20];
    for (int c = 0; c < 32; ++c) {
      std::memcpy(x3[c][0], c3_buf_[c][0], sizeof(float) * 20 * 2);
      std::memcpy(x3[c][2], h2[c], sizeof(float) * 20);
      std::memcpy(c3_buf_[c][0], c3_buf_[c][1], sizeof(float) * 20);
      std::memcpy(c3_buf_[c][1], h2[c], sizeof(float) * 20);
    }

    float flat[320];
    for (int oc = 0; oc < 32; ++oc) {
      const int8_t* wq = weights::conv3_q + oc * (32 * 9);
      float sc = weights::conv3_s[oc];
      float bias = weights::conv3_b[oc];
      for (int fo = 0; fo < 10; ++fo) {
        int fi0 = fo * 2 - 1;
        float acc = 0.0f;
        for (int ic = 0; ic < 32; ++ic) {
          const int8_t* w_ic = wq + ic * 9;
          for (int kt = 0; kt < 3; ++kt) {
            const float* row = x3[ic][kt];
            for (int kf = 0; kf < 3; ++kf) {
              int fi = fi0 + kf;
              if (fi >= 0 && fi < 20) acc += row[fi] * static_cast<float>(w_ic[kt * 3 + kf]);
            }
          }
        }
        flat[oc * 10 + fo] = relu(acc * sc + bias);
      }
    }

    // --- Projection 1x1: 320 -> 224 (INT8 x INT8) ---
    int8_t q320[320];
    float s320 = quantize_act_sym(flat, q320, 320);
    float x[C];
    for (int oc = 0; oc < C; ++oc) {
      int32_t acc = dot_i8(q320, weights::proj_q + oc * 320, 320);
      x[oc] = relu(static_cast<float>(acc) * (s320 * weights::proj_s[oc]) + weights::proj_b[oc]);
    }

    // --- 8 Causal Depthwise-Separable Residual Blocks ---
    float h_dw[C];
    int8_t q224[C];
    float x_next[C];
    for (int bi = 0; bi < B; ++bi) {
      const int d = weights::kDilations[bi];
      const int pad = kPads[bi];
      const int base = kOffsets[bi];
      const int pos = ds_pos_[bi];
      const int8_t* dw_q = weights::kBlkDwQ[bi];
      const float* dw_s = weights::kBlkDwS[bi];
      const float* dw_b = weights::kBlkDwB[bi];

      // Historical tap indices in ring buffer: tap 0..3 at offsets 4d, 3d, 2d, 1d ago; tap 4 is current x
      int idx0 = (pos + pad - 4 * d) % pad;
      int idx1 = (pos + pad - 3 * d) % pad;
      int idx2 = (pos + pad - 2 * d) % pad;
      int idx3 = (pos + pad - 1 * d) % pad;
      const float* r0 = ds_ring_[base + idx0];
      const float* r1 = ds_ring_[base + idx1];
      const float* r2 = ds_ring_[base + idx2];
      const float* r3 = ds_ring_[base + idx3];

      for (int c = 0; c < C; ++c) {
        const int8_t* w = dw_q + c * 5;
        float acc = r0[c] * static_cast<float>(w[0]) +
                    r1[c] * static_cast<float>(w[1]) +
                    r2[c] * static_cast<float>(w[2]) +
                    r3[c] * static_cast<float>(w[3]) +
                    x[c]  * static_cast<float>(w[4]);
        h_dw[c] = relu(acc * dw_s[c] + dw_b[c]);
      }

      // Push current x into ring buffer before updating x
      std::memcpy(ds_ring_[base + pos], x, sizeof(float) * C);
      ds_pos_[bi] = (pos + 1) % pad;

      // Pointwise 1x1: 224 -> 224 (INT8 x INT8) + Residual + ReLU
      float s_dw = quantize_act_sym(h_dw, q224, C);
      const int8_t* pw_q = weights::kBlkPwQ[bi];
      const float* pw_s = weights::kBlkPwS[bi];
      const float* pw_b = weights::kBlkPwB[bi];
      for (int oc = 0; oc < C; ++oc) {
        int32_t acc = dot_i8(q224, pw_q + oc * C, C);
        float pw = static_cast<float>(acc) * (s_dw * pw_s[oc]) + pw_b[oc];
        x_next[oc] = relu(x[oc] + pw);
      }
      std::memcpy(x, x_next, sizeof(float) * C);
    }

    // --- Output head 1x1: 224 -> 44 + LogSoftmax ---
    float s_out = quantize_act_sym(x, q224, C);
    float logits[V];
    float max_l = -1e30f;
    for (int v = 0; v < V; ++v) {
      int32_t acc = dot_i8(q224, weights::out_q + v * C, C);
      logits[v] = static_cast<float>(acc) * (s_out * weights::out_s[v]) + weights::out_b[v];
      if (logits[v] > max_l) max_l = logits[v];
    }
    float sum_exp = 0.0f;
    for (int v = 0; v < V; ++v) sum_exp += std::exp(logits[v] - max_l);
    float log_sum = max_l + std::log(sum_exp);
    for (int v = 0; v < V; ++v) out_logp[v] = logits[v] - log_sum;
  }
};

}  // namespace tiny_ja_ww
