"""High-level streaming Japanese wake-word detector powered by ~629 KB INT8 ONNX models."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import soxr

from .aligner import Aligner, ProbabilityCalibrator, Trigger
from .kana import ID_TO_TOKEN, phone_ids, text_to_phones

SAMPLE_RATE = 16000
MODELS_DIR = Path(__file__).resolve().parent / "models"
AVAILABLE_MODELS: dict[str, Path] = {
    "crnn": MODELS_DIR / "ja-ww-crnn-int8.onnx",
    "conv": MODELS_DIR / "ja-ww-conv-int8.onnx",
}


@dataclass(frozen=True)
class Detection:
    phrase: str
    probability: float  # Calibrated 0.0 - 1.0
    level: str          # "sure" or "maybe"
    call: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    frame_index: int    # Index within the current feed() output batch


class Detector:
    """Streaming open-vocabulary Japanese wake-word detector.

    Example:
        >>> from tiny_ja_wakeword import Detector
        >>> det = Detector("オーケー、グーグル", threshold=0.5)
        >>> events = det.feed(pcm_16k_float32)
    """

    def __init__(
        self,
        phrases: str | Iterable[str],
        threshold: float = 0.5,
        maybe: float | None = None,
        model: str | Path = "crnn",
        sample_rate: int = SAMPLE_RATE,
        threads: int = 1,
    ):
        self.model_path = self._resolve_model(model)
        self.model_name = str(model) if str(model) in AVAILABLE_MODELS else self.model_path.name
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.model_path), opts, providers=["CPUExecutionProvider"])

        raw_meta = self.session.get_modelmeta().custom_metadata_map.get("wakeword_config")
        if not raw_meta:
            raise ValueError(f"ONNX モデルに wakeword_config メタデータが見つかりません: {self.model_path}")
        self.config: dict = json.loads(raw_meta)
        self.frame_seconds = float(self.config.get("frame_seconds", 0.04))
        self.chunk_fbank_frames = int(self.config.get("chunk_fbank_frames", 16))
        self.rearm_frames = int(self.config.get("rearm_frames", 25))
        self.stretch = int(self.config.get("stretch", 5))
        self.calibrator = ProbabilityCalibrator(
            internal_gate=self.config["internal_ctc_gate"],
            weights=self.config["weights"],
            bias=self.config["bias"],
            features=self.config["features"],
        )

        self._in_names = [i.name for i in self.session.get_inputs()]
        self._out_names = [o.name for o in self.session.get_outputs()]
        self._state_shapes = {
            i.name: [int(d) for d in i.shape]
            for i in self.session.get_inputs()[1:]
        }

        self.threshold = float(threshold)
        self.maybe = None if maybe is None else float(maybe)
        self.sample_rate = int(sample_rate)
        self.set_phrases(phrases)
        self.reset(sample_rate=self.sample_rate)

    @staticmethod
    def _resolve_model(model: str | Path) -> Path:
        key = str(model)
        if key in AVAILABLE_MODELS:
            path = AVAILABLE_MODELS[key]
        else:
            path = Path(model).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"モデルファイルが見つかりません: {path}")
        return path

    def set_phrases(self, phrases: str | Iterable[str]) -> None:
        if isinstance(phrases, str):
            phrase_list = [phrases]
        else:
            phrase_list = list(phrases)
        if not phrase_list:
            raise ValueError("ウェイクワードを 1 つ以上指定してください。")
        self._entries = []
        for p in phrase_list:
            text = p.strip()
            labels = text_to_phones(text)
            ids = phone_ids(text)
            aligner = Aligner(ids, stretch=self.stretch)
            trigger = Trigger(threshold=self.threshold, maybe=self.maybe, rearm=self.rearm_frames)
            self._entries.append(dict(phrase=text, labels=labels, ids=ids, aligner=aligner, trigger=trigger))

    def set_thresholds(self, threshold: float, maybe: float | None = None) -> None:
        self.threshold = float(threshold)
        self.maybe = None if maybe is None else float(maybe)
        for e in self._entries:
            e["aligner"].reset()
            e["trigger"] = Trigger(threshold=self.threshold, maybe=self.maybe, rearm=self.rearm_frames)

    @property
    def phrases(self) -> list[str]:
        return [e["phrase"] for e in self._entries]

    def phrase_info(self) -> list[dict]:
        return [
            dict(
                phrase=e["phrase"],
                phones=list(e["labels"]),
                longest_seconds=round(e["aligner"].history * self.frame_seconds, 2),
                short=len(e["labels"]) < 6,
            )
            for e in self._entries
        ]

    def reset(self, sample_rate: int | None = None) -> None:
        if sample_rate is not None:
            self.sample_rate = int(sample_rate)
        opts = knf.FbankOptions()
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = False
        opts.mel_opts.num_bins = 80
        opts.mel_opts.high_freq = -400
        self._fbank = knf.OnlineFbank(opts)
        self._fb_offset = 0
        self._resampler = (
            None if self.sample_rate == SAMPLE_RATE
            else soxr.ResampleStream(self.sample_rate, SAMPLE_RATE, 1, dtype="float32")
        )
        self._states = {name: np.zeros(shape, dtype=np.float32) for name, shape in self._state_shapes.items()}
        self._prev_symbol = 0
        self.seconds = 0.0
        for e in self._entries:
            e["aligner"].reset()
            e["trigger"].reset()

    def _run_model(self, samples: np.ndarray, final: bool = False) -> np.ndarray:
        arr = np.asarray(samples, dtype=np.float32)
        if arr.ndim != 1:
            arr = arr.mean(axis=-1).astype(np.float32)
        if self._resampler is not None and len(arr):
            arr = self._resampler.resample_chunk(arr)
        self.seconds += len(arr) / SAMPLE_RATE
        if len(arr):
            self._fbank.accept_waveform(SAMPLE_RATE, arr)
        if final:
            self._fbank.accept_waveform(SAMPLE_RATE, np.zeros(160 * self.chunk_fbank_frames, dtype=np.float32))
            self._fbank.input_finished()

        out_chunks = []
        step = self.chunk_fbank_frames
        while self._fb_offset + step <= self._fbank.num_frames_ready:
            chunk = np.stack([self._fbank.get_frame(i) for i in range(self._fb_offset, self._fb_offset + step)])
            feed = {self._in_names[0]: chunk[None].astype(np.float32), **self._states}
            res = self.session.run(self._out_names, feed)
            out_chunks.append(res[0][0])
            for name, val in zip(self._in_names[1:], res[1:]):
                self._states[name] = val
            self._fb_offset += step
            self._fbank.pop(step)
        return np.concatenate(out_chunks, axis=0) if out_chunks else np.empty((0, 44), dtype=np.float32)

    def feed_detailed(self, samples: np.ndarray, *, final: bool = False) -> dict:
        """Process audio chunk and return calibrated probabilities, Detection events, and greedy heard phonemes."""
        log_probs = self._run_model(samples, final=final)
        heard: list[str] = []
        events: list[Detection] = []
        probs_by_phrase: dict[str, list[float]] = {e["phrase"]: [] for e in self._entries}

        if len(log_probs):
            for sym in log_probs.argmax(axis=1):
                s = int(sym)
                if s != self._prev_symbol and s > 1:
                    heard.append(ID_TO_TOKEN[s])
                self._prev_symbol = s

            for idx, post in enumerate(log_probs):
                for e in self._entries:
                    cand = e["aligner"].push(post)
                    prob = self.calibrator.score(cand)
                    probs_by_phrase[e["phrase"]].append(prob)
                    if cand is not None:
                        fired = e["trigger"].fire(cand.end, prob, cand.start, cand.end)
                        if fired is not None:
                            events.append(
                                Detection(
                                    phrase=e["phrase"],
                                    probability=fired["probability"],
                                    level=fired["level"],
                                    call=fired["call"],
                                    start_seconds=round(fired["start_frame"] * self.frame_seconds, 3),
                                    end_seconds=round(fired["end_frame"] * self.frame_seconds, 3),
                                    duration_seconds=round((fired["end_frame"] - fired["start_frame"]) * self.frame_seconds, 3),
                                    frame_index=idx,
                                )
                            )
        return dict(
            probabilities=probs_by_phrase,
            events=events,
            heard=heard,
            seconds=round(self.seconds, 2),
        )

    def feed(self, samples: np.ndarray, *, final: bool = False) -> list[Detection]:
        """Process a chunk of audio (float32 in [-1, 1]) and return any wake-word Detection events."""
        return self.feed_detailed(samples, final=final)["events"]
