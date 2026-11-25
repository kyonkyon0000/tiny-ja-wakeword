"""Thread-safe streaming listener driving the browser microphone demo with 0-100% calibrated probability."""
from __future__ import annotations

import threading
from pathlib import Path

from ..detector import AVAILABLE_MODELS, Detector

MODEL_LABELS: dict[str, str] = {
    "crnn": "crnn（推奨・629 KB・雑音に強い連続 GRU）",
    "conv": "conv（623 KB・純畳み込み・マイコン向け）",
}


def available_models() -> list[dict[str, str]]:
    return [
        {"id": key, "label": MODEL_LABELS.get(key, key)}
        for key in AVAILABLE_MODELS
        if AVAILABLE_MODELS[key].exists()
    ]


class Listener:
    def __init__(self, model: str = "crnn", phrase: str = "オーケー、グーグル", sure: float = 0.50, maybe: float = 0.30):
        self.lock = threading.Lock()
        self.model = model
        self.sure = float(sure)
        self.maybe = float(maybe)
        self.detector = Detector(phrase, threshold=self.sure, maybe=self.maybe, model=model)

    def configure(
        self,
        phrase: str | None = None,
        model: str | None = None,
        sure: float | None = None,
        maybe: float | None = None,
    ) -> dict:
        with self.lock:
            next_sure = self.sure if sure is None else float(sure)
            next_maybe = self.maybe if maybe is None else float(maybe)
            if not (0.05 <= next_sure <= 0.99):
                raise ValueError("確定しきい値は 5% 〜 99% の間で指定してください。")
            if not (0.05 <= next_maybe <= next_sure):
                raise ValueError("「かも」のしきい値は、確定しきい値以下にしてください。")
            next_phrase = self.detector.phrases[0] if phrase is None else phrase.strip()
            if not next_phrase:
                raise ValueError("ウェイクワードを入力してください。")

            if model is not None and model != self.model:
                if model not in AVAILABLE_MODELS and not Path(model).exists():
                    raise ValueError(f"未対応のモデルです: {model}")
                self.detector = Detector(
                    next_phrase,
                    threshold=next_sure,
                    maybe=next_maybe,
                    model=model,
                    sample_rate=self.detector.sample_rate,
                )
                self.model = model
            else:
                if next_phrase != self.detector.phrases[0]:
                    self.detector.set_phrases(next_phrase)
                if next_sure != self.sure or next_maybe != self.maybe:
                    self.detector.set_thresholds(next_sure, next_maybe)
            self.sure, self.maybe = next_sure, next_maybe
            return self.state()

    def state(self) -> dict:
        info = self.detector.phrase_info()[0]
        return dict(
            phrase=info["phrase"],
            phones=info["phones"],
            longest_seconds=info["longest_seconds"],
            short=info["short"],
            model=self.model,
            models=available_models(),
            sure=round(self.sure, 2),
            maybe=round(self.maybe, 2),
            frame_seconds=self.detector.frame_seconds,
            rearm_seconds=round(self.detector.rearm_frames * self.detector.frame_seconds, 2),
        )

    def restart(self, rate: int) -> None:
        with self.lock:
            self.detector.reset(sample_rate=int(rate))

    def feed(self, samples) -> dict:
        with self.lock:
            res = self.detector.feed_detailed(samples)
            phrase = self.detector.phrases[0]
            probs = [round(p * 100.0, 1) for p in res["probabilities"][phrase]]
            events = [
                dict(
                    level=e.level,
                    call=e.call,
                    prob=round(e.probability * 100.0, 1),
                    at=e.frame_index,
                    length_seconds=round(e.duration_seconds, 2),
                )
                for e in res["events"]
            ]
            return dict(scores=probs, events=events, heard=res["heard"], seconds=res["seconds"])
