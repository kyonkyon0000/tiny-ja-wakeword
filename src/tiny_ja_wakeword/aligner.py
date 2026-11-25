"""Streaming free-start CTC Viterbi aligner, calibrated 0-100% probability scorer, and echo-safe trigger."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Candidate:
    start: int
    end: int
    cost: float
    tokens_len: int

    @property
    def per_phone(self) -> float:
        return self.cost / self.tokens_len

    @property
    def per_frame(self) -> float:
        return self.cost / max(1, self.end - self.start)

    @property
    def dur_per_phone(self) -> float:
        return (self.end - self.start) / self.tokens_len


class Aligner:
    """Online free-start CTC Viterbi alignment with bounded duration stretch."""

    def __init__(self, tokens: np.ndarray, history: int = 160, stretch: int = 5):
        self.tokens = np.asarray(tokens, dtype=np.int64)
        if not len(self.tokens) or np.any(self.tokens <= 0):
            raise ValueError("音素列は空にできません。")
        if stretch is not None:
            history = min(history, max(2 * len(self.tokens), round(stretch * len(self.tokens))))
        self.labels = np.zeros(2 * len(self.tokens) - 1, dtype=np.int64)
        self.labels[::2] = self.tokens
        if len(self.labels) > history:
            raise ValueError("語句が長すぎます。もう少し短めに入力してください。")
        self.history = history
        self.reset()

    def reset(self) -> None:
        self.previous = np.full(len(self.labels), -np.inf, dtype=np.float64)
        self.starts = np.zeros(len(self.labels), dtype=np.int64)
        self.index = 0

    def push(self, log_probs: np.ndarray) -> Candidate | None:
        t = self.index
        self.index += 1
        states = np.arange(len(self.labels))
        prev = np.where(t - self.starts < self.history, self.previous, -np.inf)
        choices = np.full((3, len(states)), -np.inf, dtype=np.float64)
        choices[0] = prev
        choices[1, 1:] = prev[:-1]
        skip = (self.labels[2:] != 0) & (self.labels[2:] != self.labels[:-2])
        choices[2, 2:] = np.where(skip, prev[:-2], -np.inf)
        choices[1, 0] = 0.0  # free start at every frame
        steps = choices.argmax(axis=0)
        self.starts = self.starts[np.maximum(0, states - steps)]
        if steps[0] == 1:
            self.starts[0] = t
        self.previous = choices[steps, states] + log_probs[self.labels]
        if not np.isfinite(self.previous[-1]):
            return None
        return Candidate(int(self.starts[-1]), t + 1, float(self.previous[-1]), len(self.tokens))


class ProbabilityCalibrator:
    """Converts a CTC alignment candidate to a calibrated 0.0-1.0 probability using ONNX-embedded metadata."""

    def __init__(self, internal_gate: float, weights: list[float], bias: float, features: list[str]):
        self.internal_gate = float(internal_gate)
        self.weights = tuple(float(w) for w in weights)
        self.bias = float(bias)
        self.features = tuple(features)

    def score(self, cand: Candidate | None) -> float:
        if cand is None or cand.per_phone < self.internal_gate:
            return 0.0
        vals = {
            "ctc_per_phone": cand.per_phone,
            "ctc_per_frame": cand.per_frame,
            "log_len": math.log(cand.tokens_len),
            "dur_per_phone": cand.dur_per_phone,
        }
        z = self.bias
        for w, name in zip(self.weights, self.features):
            z += w * vals[name]
        z = max(-40.0, min(40.0, z))
        return 1.0 / (1.0 + math.exp(-z))


class Trigger:
    """Turns per-frame calibrated probabilities into clean, de-duplicated wake-word events."""

    def __init__(self, threshold: float = 0.5, maybe: float | None = None, rearm: int = 25, settle: int = 5):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("しきい値（threshold）は 0.0 より大きく 1.0 以下で指定してください。")
        if maybe is not None and not (0.0 < maybe <= threshold):
            raise ValueError("「かも」のしきい値（maybe）は 0.0 より大きく threshold 以下にしてください。")
        self.threshold = float(threshold)
        self.maybe = None if maybe is None else float(maybe)
        self.rearm = int(rearm)
        self.settle = int(settle)
        self.reset()

    def reset(self) -> None:
        self.call = 0
        self.last: int | None = None
        self.first: int | None = None
        self.best: dict | None = None
        self.sent: str | None = None
        self.consumed_until = -1

    def fire(self, frame: int, prob: float, start_frame: int, end_frame: int) -> dict | None:
        low = self.threshold if self.maybe is None else self.maybe
        overlapping = start_frame < self.consumed_until
        if not overlapping and prob >= low:
            if self.last is None or frame - self.last > self.rearm:
                self.call += 1
                self.first, self.best, self.sent = frame, None, None
            self.last = frame
            if self.sent != "sure" and (self.sent != "maybe" or prob >= self.threshold):
                info = dict(probability=prob, start_frame=start_frame, end_frame=end_frame)
                if self.best is None or prob > self.best["probability"]:
                    self.best = info
                if prob >= self.threshold:
                    self.consumed_until = end_frame
                    self.sent = "sure"
                    return dict(info, level="sure", call=self.call)
        if self.sent is None and self.best is not None and self.first is not None and frame - self.first >= self.settle:
            self.sent = "maybe"
            return dict(self.best, level="maybe", call=self.call)
        return None
