"""tiny-ja-wakeword: Ultra-tiny (~629 KB INT8) noise-robust streaming Japanese wake-word detector."""
from .detector import AVAILABLE_MODELS, Detection, Detector, SAMPLE_RATE
from .kana import TOKENS, kana_to_phones, phone_ids, text_to_phones

__all__ = [
    "AVAILABLE_MODELS",
    "Detection",
    "Detector",
    "SAMPLE_RATE",
    "TOKENS",
    "kana_to_phones",
    "phone_ids",
    "text_to_phones",
]
__version__ = "0.1.0"
