"""Automated tests for tiny-ja-wakeword (kana G2P, streaming ONNX, calibration, trigger, and demo)."""
import unittest

import numpy as np

from tiny_ja_wakeword import AVAILABLE_MODELS, Detector, kana_to_phones, phone_ids, text_to_phones
from tiny_ja_wakeword.aligner import Trigger
from tiny_ja_wakeword.demo.listener import Listener


class KanaTests(unittest.TestCase):
    def test_common_phrases_match_expected_phonemes(self):
        cases = {
            "オーケー、グーグル": ["o", "o", "k", "e", "e", "g", "u", "u", "g", "u", "r", "u"],
            "おーけーぐーぐる": ["o", "o", "k", "e", "e", "g", "u", "u", "g", "u", "r", "u"],
            "ヘイ、シリ": ["h", "e", "i", "sh", "i", "r", "i"],
            "おはよう": ["o", "h", "a", "y", "o", "o"],
            "すみません": ["s", "u", "m", "i", "m", "a", "s", "e", "N"],
            "ちょっとまって": ["ch", "o", "cl", "t", "o", "m", "a", "cl", "t", "e"],
            "きゅうり": ["ky", "u", "u", "r", "i"],
            "しゃしん": ["sh", "a", "sh", "i", "N"],
            "びょういん": ["by", "o", "o", "i", "N"],
            "ふぁいる": ["f", "a", "i", "r", "u"],
            "てぃー": ["t", "i", "i"],
            "でぃすく": ["d", "i", "s", "u", "k", "u"],
            "ヴぁいおりん": ["v", "a", "i", "o", "r", "i", "N"],
            "サッカー": ["s", "a", "cl", "k", "a", "a"],
            "ロボット": ["r", "o", "b", "o", "cl", "t", "o"],
            "スイッチ": ["s", "u", "i", "cl", "ch", "i"],
            "をかし": ["o", "k", "a", "sh", "i"],
        }
        for phrase, expected in cases.items():
            with self.subTest(phrase=phrase):
                self.assertEqual(kana_to_phones(phrase), expected)
                ids = phone_ids(phrase)
                self.assertEqual(len(ids), len(expected))
                self.assertTrue(np.all(ids > 1))

    def test_empty_and_kanji_phrases_rejected(self):
        with self.assertRaises(ValueError):
            text_to_phones("   ")
        with self.assertRaises(ValueError):
            text_to_phones("今日は晴れ")


class TriggerTests(unittest.TestCase):
    def test_maybe_upgrades_to_sure_and_suppresses_overlapping_echo(self):
        tr = Trigger(threshold=0.5, maybe=0.3, rearm=5, settle=2)
        self.assertIsNone(tr.fire(10, 0.35, start_frame=2, end_frame=10))
        self.assertIsNone(tr.fire(11, 0.38, start_frame=2, end_frame=11))
        m = tr.fire(12, 0.40, start_frame=2, end_frame=12)
        self.assertIsNotNone(m)
        self.assertEqual(m["level"], "maybe")
        s = tr.fire(13, 0.65, start_frame=2, end_frame=13)
        self.assertIsNotNone(s)
        self.assertEqual(s["level"], "sure")
        self.assertEqual(s["call"], m["call"])
        self.assertIsNone(tr.fire(25, 0.80, start_frame=10, end_frame=25))
        s2 = tr.fire(30, 0.80, start_frame=20, end_frame=30)
        self.assertIsNotNone(s2)
        self.assertEqual(s2["call"], 2)


class DetectorTests(unittest.TestCase):
    def test_both_bundled_models_load_and_stream_deterministically(self):
        rng = np.random.RandomState(42)
        audio = (rng.randn(16000 * 2) * 0.02).astype(np.float32)
        for name in ("crnn", "conv"):
            with self.subTest(model=name):
                path = AVAILABLE_MODELS[name]
                self.assertTrue(path.exists())
                self.assertLess(path.stat().st_size, 700_000)

                d1 = Detector(["オーケー、グーグル", "ヘイ、シリ"], threshold=0.5, model=name)
                r1 = d1.feed_detailed(audio)
                d2 = Detector(["オーケー、グーグル", "ヘイ、シリ"], threshold=0.5, model=name)
                p_chunks = []
                for start in range(0, len(audio), 1600):
                    r = d2.feed_detailed(audio[start:start + 1600])
                    p_chunks.extend(r["probabilities"]["オーケー、グーグル"])
                self.assertGreater(len(p_chunks), 40)
                self.assertEqual(len(p_chunks), len(r1["probabilities"]["オーケー、グーグル"]))
                max_diff = float(np.max(np.abs(np.array(p_chunks) - np.array(r1["probabilities"]["オーケー、グーグル"]))))
                self.assertLess(max_diff, 1e-5)
                self.assertTrue(all(0.0 <= p <= 1.0 for p in p_chunks))

    def test_demo_listener_configure_and_feed(self):
        listener = Listener(model="crnn", phrase="オーケー、グーグル", sure=0.5, maybe=0.3)
        st = listener.state()
        self.assertEqual(st["phrase"], "オーケー、グーグル")
        self.assertEqual(st["sure"], 0.5)
        st2 = listener.configure(phrase="ヘイ、シリ", sure=0.7, maybe=0.4, model="conv")
        self.assertEqual(st2["model"], "conv")
        self.assertEqual(st2["sure"], 0.7)
        listener.restart(48000)
        out = listener.feed(np.zeros(48000, dtype=np.float32))
        self.assertIn("scores", out)
        self.assertGreater(len(out["scores"]), 10)


if __name__ == "__main__":
    unittest.main()
