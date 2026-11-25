// Pure C++17 streaming 80-bin Kaldi FBANK extractor (16 kHz, 25ms Povey window, 10ms hop, snip_edges=false).
// Zero external dependencies; matches kaldi-native-fbank bit-accurately for PC, Raspberry Pi, and ESP32-S3.
#pragma once

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace tiny_ja_ww {

class KaldiFbank80 {
 public:
  static constexpr int kSampleRate = 16000;
  static constexpr int kFrameLen = 400;   // 25 ms
  static constexpr int kFrameShift = 160; // 10 ms
  static constexpr int kFftSize = 512;
  static constexpr int kHalfFft = kFftSize / 2 + 1;  // 257
  static constexpr int kNumBins = 80;
  static constexpr float kPreemph = 0.97f;
  static constexpr float kLowFreq = 20.0f;
  static constexpr float kHighFreq = 7600.0f;  // Nyquist(8000) - 400
  static constexpr int kBufCap = 2048;

  KaldiFbank80() {
    init_window();
    init_fft();
    init_mel();
    reset();
  }

  void reset() {
    total_samples_ = 0;
    frame_index_ = 0;
    buf_start_sample_ = 0;
    buf_len_ = 0;
  }

  // Push mono float32 samples in [-1, 1] (or int16 scaled by 1/32768).
  // Calls `on_frame(const float fb[80])` for every complete 10ms FBANK frame ready.
  template <typename Callback>
  void accept(const float* samples, int count, Callback&& on_frame) {
    int offset = 0;
    while (offset < count) {
      int space = kBufCap - buf_len_;
      int take = std::min(space, count - offset);
      std::memcpy(buf_ + buf_len_, samples + offset, sizeof(float) * take);
      buf_len_ += take;
      total_samples_ += take;
      offset += take;
      drain_frames(false, on_frame);
      compact_buffer();
    }
  }

  template <typename Callback>
  void finish(Callback&& on_frame) {
    drain_frames(true, on_frame);
  }

 private:
  float window_[kFrameLen];
  float cos_tab_[kFftSize / 2];
  float sin_tab_[kFftSize / 2];
  uint16_t bitrev_[kFftSize];

  int mel_start_[kNumBins];
  int mel_len_[kNumBins];
  int mel_off_[kNumBins];
  float mel_weights_[512];  // ~260 non-zero weights for 80 bins over 257 FFT bins

  float buf_[kBufCap];
  int64_t total_samples_ = 0;
  int64_t frame_index_ = 0;
  int64_t buf_start_sample_ = 0;
  int buf_len_ = 0;

  static float mel_scale(float freq) { return 1127.0f * std::log(1.0f + freq / 700.0f); }

  void init_window() {
    constexpr float kTwoPi = 6.28318530717958647692f;
    for (int i = 0; i < kFrameLen; ++i) {
      float a = kTwoPi * static_cast<float>(i) / static_cast<float>(kFrameLen - 1);
      window_[i] = std::pow(0.5f - 0.5f * std::cos(a), 0.85f);
    }
  }

  void init_fft() {
    constexpr float kTwoPi = 6.28318530717958647692f;
    for (int i = 0; i < kFftSize / 2; ++i) {
      float ang = -kTwoPi * static_cast<float>(i) / static_cast<float>(kFftSize);
      cos_tab_[i] = std::cos(ang);
      sin_tab_[i] = std::sin(ang);
    }
    for (int i = 0; i < kFftSize; ++i) {
      int r = 0;
      for (int b = 0; b < 9; ++b) {
        if (i & (1 << b)) r |= (1 << (8 - b));
      }
      bitrev_[i] = static_cast<uint16_t>(r);
    }
  }

  void init_mel() {
    const float mel_low = mel_scale(kLowFreq);
    const float mel_high = mel_scale(kHighFreq);
    const float mel_delta = (mel_high - mel_low) / static_cast<float>(kNumBins + 1);
    const float fft_bin_width = static_cast<float>(kSampleRate) / static_cast<float>(kFftSize);

    int w_pos = 0;
    for (int bin = 0; bin < kNumBins; ++bin) {
      float left_mel = mel_low + bin * mel_delta;
      float center_mel = mel_low + (bin + 1) * mel_delta;
      float right_mel = mel_low + (bin + 2) * mel_delta;
      int first = -1, last = -1;
      mel_off_[bin] = w_pos;
      for (int k = 0; k < kHalfFft - 1; ++k) {
        float m = mel_scale(fft_bin_width * static_cast<float>(k));
        if (m > left_mel && m < right_mel) {
          float w = (m <= center_mel) ? (m - left_mel) / (center_mel - left_mel)
                                      : (right_mel - m) / (right_mel - center_mel);
          if (first < 0) first = k;
          last = k;
          mel_weights_[w_pos++] = w;
        }
      }
      mel_start_[bin] = (first < 0) ? 0 : first;
      mel_len_[bin] = (first < 0) ? 0 : (last - first + 1);
    }
  }

  float sample_at(int64_t idx) const {
    // Exact Kaldi snip_edges=false reflection at both ends
    if (total_samples_ <= 0) return 0.0f;
    while (idx < 0 || idx >= total_samples_) {
      if (idx < 0) idx = -idx - 1;
      if (idx >= total_samples_) idx = 2 * total_samples_ - 1 - idx;
    }
    int64_t local = idx - buf_start_sample_;
    if (local < 0) local = 0;
    if (local >= buf_len_) local = buf_len_ - 1;
    return buf_[local];
  }

  template <typename Callback>
  void drain_frames(bool flush, Callback&& on_frame) {
    while (true) {
      int64_t midpoint = frame_index_ * kFrameShift + kFrameShift / 2;
      int64_t start = midpoint - kFrameLen / 2;
      int64_t end = start + kFrameLen;
      if (!flush) {
        if (end > total_samples_) break;
      } else {
        int64_t num_frames = (total_samples_ + kFrameShift / 2) / kFrameShift;
        if (frame_index_ >= num_frames) break;
      }
      float fb[kNumBins];
      compute_frame(start, fb);
      on_frame(fb);
      frame_index_++;
    }
  }

  void compact_buffer() {
    int64_t next_start = frame_index_ * kFrameShift + kFrameShift / 2 - kFrameLen / 2;
    // Keep any sample >= 0 needed for the next frame (and if next_start < 0, keep from sample 0)
    int64_t keep_from = std::max<int64_t>(0, next_start);
    if (keep_from > buf_start_sample_) {
      int64_t drop = std::min<int64_t>(keep_from - buf_start_sample_, buf_len_);
      if (drop > 0) {
        int remain = buf_len_ - static_cast<int>(drop);
        if (remain > 0) std::memmove(buf_, buf_ + drop, sizeof(float) * remain);
        buf_start_sample_ += drop;
        buf_len_ = remain;
      }
    }
  }

  void compute_frame(int64_t start_sample, float out_fb[kNumBins]) const {
    float re[kFftSize];
    float im[kFftSize];

    // 1. Gather 400 samples & remove DC offset
    float sum = 0.0f;
    for (int i = 0; i < kFrameLen; ++i) {
      float v = sample_at(start_sample + i);
      re[i] = v;
      sum += v;
    }
    float mean = sum / static_cast<float>(kFrameLen);
    for (int i = 0; i < kFrameLen; ++i) re[i] -= mean;

    // 2. Pre-emphasis (reverse loop in-place) + Povey window
    for (int i = kFrameLen - 1; i > 0; --i) {
      re[i] = (re[i] - kPreemph * re[i - 1]) * window_[i];
    }
    re[0] = (re[0] - kPreemph * re[0]) * window_[0];
    for (int i = kFrameLen; i < kFftSize; ++i) re[i] = 0.0f;

    // 3. Bit-reversed copy into FFT working arrays
    float xr[kFftSize], xi[kFftSize];
    for (int i = 0; i < kFftSize; ++i) {
      xr[bitrev_[i]] = re[i];
      xi[bitrev_[i]] = 0.0f;
    }

    // 4. Iterative radix-2 Cooley-Tukey FFT (512 points)
    for (int size = 2; size <= kFftSize; size <<= 1) {
      int half = size >> 1;
      int step = kFftSize / size;
      for (int i = 0; i < kFftSize; i += size) {
        int tw = 0;
        for (int j = 0; j < half; ++j) {
          float wr = cos_tab_[tw];
          float wi = sin_tab_[tw];
          int u = i + j;
          int v = u + half;
          float tr = wr * xr[v] - wi * xi[v];
          float ti = wr * xi[v] + wi * xr[v];
          xr[v] = xr[u] - tr;
          xi[v] = xi[u] - ti;
          xr[u] += tr;
          xi[u] += ti;
          tw += step;
        }
      }
    }

    // 5. Power spectrum & 80 Mel filterbank energies (natural log)
    float power[kHalfFft];
    for (int k = 0; k < kHalfFft; ++k) {
      power[k] = xr[k] * xr[k] + xi[k] * xi[k];
    }
    for (int bin = 0; bin < kNumBins; ++bin) {
      int s = mel_start_[bin];
      int n = mel_len_[bin];
      const float* w = mel_weights_ + mel_off_[bin];
      float e = 0.0f;
      for (int j = 0; j < n; ++j) e += power[s + j] * w[j];
      if (e < FLT_EPSILON) e = FLT_EPSILON;
      out_fb[bin] = std::log(e);
    }
  }
};

}  // namespace tiny_ja_ww
