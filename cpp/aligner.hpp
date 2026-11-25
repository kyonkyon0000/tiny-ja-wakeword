// Streaming free-start CTC Viterbi aligner + 4-feature 0-100% calibrated probability + echo-safe trigger in pure C++17.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "weights_conv_int8.h"

namespace tiny_ja_ww {

struct DetectionEvent {
  float probability;
  int start_frame;
  int end_frame;
  float end_seconds;
  int call;
};

class WakeWordAligner {
 public:
  static constexpr int kMaxStates = 128;  // supports up to 64 phonemes (~30+ Japanese morae)

  bool enroll(const std::vector<int>& tokens, float threshold = 0.50f) {
    if (tokens.empty() || static_cast<int>(tokens.size()) * 2 - 1 > kMaxStates) return false;
    n_tokens_ = static_cast<int>(tokens.size());
    n_states_ = 2 * n_tokens_ - 1;
    history_ = std::min(160, std::max(2 * n_tokens_, weights::kStretch * n_tokens_));
    for (int s = 0; s < n_states_; ++s) {
      labels_[s] = (s % 2 == 0) ? tokens[s / 2] : 0;
    }
    threshold_ = threshold;
    log_len_ = std::log(static_cast<float>(n_tokens_));
    reset();
    return true;
  }

  void reset() {
    for (int s = 0; s < n_states_; ++s) {
      prev_[s] = -std::numeric_limits<float>::infinity();
      starts_[s] = 0;
    }
    index_ = 0;
    call_ = 0;
    last_pass_ = -100000;
    sent_ = false;
    consumed_until_ = -1;
  }

  // Push one 40ms CTC log-probability vector[44]. Returns true if wake-word fired at this frame.
  bool push(const float logp[44], DetectionEvent* out_event, float* out_prob = nullptr) {
    const int t = index_++;
    const float neg_inf = -std::numeric_limits<float>::infinity();

    float next_prev[kMaxStates];
    int next_starts[kMaxStates];

    for (int s = 0; s < n_states_; ++s) {
      float p_cur = (t - starts_[s] < history_) ? prev_[s] : neg_inf;
      float best = p_cur;
      int best_step = 0;

      if (s == 0) {
        if (0.0f > best) { best = 0.0f; best_step = 1; }
      } else {
        float p_m1 = (t - starts_[s - 1] < history_) ? prev_[s - 1] : neg_inf;
        if (p_m1 > best) { best = p_m1; best_step = 1; }
      }

      if (s >= 2 && labels_[s] != 0 && labels_[s] != labels_[s - 2]) {
        float p_m2 = (t - starts_[s - 2] < history_) ? prev_[s - 2] : neg_inf;
        if (p_m2 > best) { best = p_m2; best_step = 2; }
      }

      next_prev[s] = best + logp[labels_[s]];
      if (s == 0 && best_step == 1) {
        next_starts[s] = t;
      } else {
        next_starts[s] = starts_[s - best_step];
      }
    }

    for (int s = 0; s < n_states_; ++s) {
      prev_[s] = next_prev[s];
      starts_[s] = next_starts[s];
    }

    float cost = prev_[n_states_ - 1];
    float prob = 0.0f;
    int start_f = starts_[n_states_ - 1];
    int end_f = t + 1;
    if (std::isfinite(cost)) {
      float s_phone = cost / static_cast<float>(n_tokens_);
      if (s_phone >= weights::kCalInternalGate) {
        float dur = static_cast<float>(std::max(1, end_f - start_f));
        float s_frame = cost / dur;
        float dur_per_phone = dur / static_cast<float>(n_tokens_);
        float z = weights::kCalBias +
                  weights::kCalWeights[0] * s_phone +
                  weights::kCalWeights[1] * s_frame +
                  weights::kCalWeights[2] * log_len_ +
                  weights::kCalWeights[3] * dur_per_phone;
        z = std::max(-40.0f, std::min(40.0f, z));
        prob = 1.0f / (1.0f + std::exp(-z));
      }
    }
    if (out_prob) *out_prob = prob;

    bool overlap = (start_f < consumed_until_);
    if (!overlap && prob >= threshold_) {
      if (end_f - last_pass_ > weights::kRearmFrames) {
        call_++;
        sent_ = false;
      }
      last_pass_ = end_f;
      if (!sent_) {
        sent_ = true;
        consumed_until_ = end_f;
        if (out_event) {
          *out_event = {prob, start_f, end_f, end_f * 0.04f, call_};
        }
        return true;
      }
    }
    return false;
  }

 private:
  int n_tokens_ = 0;
  int n_states_ = 0;
  int history_ = 160;
  int labels_[kMaxStates];
  float prev_[kMaxStates];
  int starts_[kMaxStates];
  int index_ = 0;
  float threshold_ = 0.50f;
  float log_len_ = 0.0f;

  int call_ = 0;
  int last_pass_ = -100000;
  bool sent_ = false;
  int consumed_until_ = -1;
};

}  // namespace tiny_ja_ww
