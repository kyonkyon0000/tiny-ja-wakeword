"""Zero-dictionary Japanese kana-to-phoneme conversion matching the 44-token CTC model vocabulary.

Hiragana and katakana phrases work directly with zero external dictionaries.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np

TOKENS: tuple[str, ...] = (
    "<blk>", "<unk>", "I", "N", "U", "a", "b", "by", "ch", "cl", "d", "dy",
    "e", "f", "fy", "g", "gw", "gy", "h", "hy", "i", "j", "k", "kw", "ky",
    "m", "my", "n", "ny", "o", "p", "py", "r", "ry", "s", "sh", "t", "ts",
    "ty", "u", "v", "w", "y", "z",
)
TOKEN_TO_ID: dict[str, int] = {tok: i for i, tok in enumerate(TOKENS)}
ID_TO_TOKEN: dict[int, str] = {i: tok for i, tok in enumerate(TOKENS)}
SAME_SOUND: dict[str, str] = {"I": "i", "U": "u"}
VOWELS = {"a", "i", "u", "e", "o"}

# Two-character kana digraphs (checked before single kana)
_DIGRAPHS: dict[str, list[str]] = {
    "きゃ": ["ky", "a"], "きゅ": ["ky", "u"], "きょ": ["ky", "o"], "きぇ": ["ky", "e"],
    "ぎゃ": ["gy", "a"], "ぎゅ": ["gy", "u"], "ぎょ": ["gy", "o"], "ぎぇ": ["gy", "e"],
    "しゃ": ["sh", "a"], "しゅ": ["sh", "u"], "しょ": ["sh", "o"], "しぇ": ["sh", "e"],
    "じゃ": ["j", "a"], "じゅ": ["j", "u"], "じょ": ["j", "o"], "じぇ": ["j", "e"],
    "ちゃ": ["ch", "a"], "ちゅ": ["ch", "u"], "ちょ": ["ch", "o"], "ちぇ": ["ch", "e"],
    "ぢゃ": ["j", "a"], "ぢゅ": ["j", "u"], "ぢょ": ["j", "o"], "ぢぇ": ["j", "e"],
    "にゃ": ["ny", "a"], "にゅ": ["ny", "u"], "にょ": ["ny", "o"], "にぇ": ["ny", "e"],
    "ひゃ": ["hy", "a"], "ひゅ": ["hy", "u"], "ひょ": ["hy", "o"], "ひぇ": ["hy", "e"],
    "びゃ": ["by", "a"], "びゅ": ["by", "u"], "びょ": ["by", "o"], "びぇ": ["by", "e"],
    "ぴゃ": ["py", "a"], "ぴゅ": ["py", "u"], "ぴょ": ["py", "o"], "ぴぇ": ["py", "e"],
    "みゃ": ["my", "a"], "みゅ": ["my", "u"], "みょ": ["my", "o"], "みぇ": ["my", "e"],
    "りゃ": ["ry", "a"], "りゅ": ["ry", "u"], "りょ": ["ry", "o"], "りぇ": ["ry", "e"],
    "ふぁ": ["f", "a"], "ふぃ": ["f", "i"], "ふぇ": ["f", "e"], "ふぉ": ["f", "o"],
    "ふゅ": ["fy", "u"], "ふょ": ["fy", "o"],
    "てぃ": ["t", "i"], "てゅ": ["ty", "u"], "とぅ": ["t", "u"],
    "でぃ": ["d", "i"], "でゅ": ["dy", "u"], "どぅ": ["d", "u"],
    "つぁ": ["ts", "a"], "つぃ": ["ts", "i"], "つぇ": ["ts", "e"], "つぉ": ["ts", "o"],
    "うぃ": ["w", "i"], "うぇ": ["w", "e"], "うぉ": ["w", "o"],
    "ゔぁ": ["v", "a"], "ゔぃ": ["v", "i"], "ゔぇ": ["v", "e"], "ゔぉ": ["v", "o"], "ゔゅ": ["by", "u"],
    "くぁ": ["kw", "a"], "くぃ": ["kw", "i"], "くぇ": ["kw", "e"], "くぉ": ["kw", "o"],
    "ぐぁ": ["gw", "a"], "ぐぃ": ["gw", "i"], "ぐぇ": ["gw", "e"], "ぐぉ": ["gw", "o"],
    "すぃ": ["s", "i"], "ずぃ": ["z", "i"],
}

_MONOS: dict[str, list[str]] = {
    "あ": ["a"], "い": ["i"], "う": ["u"], "え": ["e"], "お": ["o"],
    "ぁ": ["a"], "ぃ": ["i"], "ぅ": ["u"], "ぇ": ["e"], "ぉ": ["o"],
    "か": ["k", "a"], "き": ["k", "i"], "く": ["k", "u"], "け": ["k", "e"], "こ": ["k", "o"],
    "が": ["g", "a"], "ぎ": ["g", "i"], "ぐ": ["g", "u"], "げ": ["g", "e"], "ご": ["g", "o"],
    "さ": ["s", "a"], "し": ["sh", "i"], "す": ["s", "u"], "せ": ["s", "e"], "そ": ["s", "o"],
    "ざ": ["z", "a"], "じ": ["j", "i"], "ず": ["z", "u"], "ぜ": ["z", "e"], "ぞ": ["z", "o"],
    "た": ["t", "a"], "ち": ["ch", "i"], "つ": ["ts", "u"], "て": ["t", "e"], "と": ["t", "o"],
    "だ": ["d", "a"], "ぢ": ["j", "i"], "づ": ["z", "u"], "で": ["d", "e"], "ど": ["d", "o"],
    "な": ["n", "a"], "に": ["n", "i"], "ぬ": ["n", "u"], "ね": ["n", "e"], "の": ["n", "o"],
    "は": ["h", "a"], "ひ": ["h", "i"], "ふ": ["f", "u"], "へ": ["h", "e"], "ほ": ["h", "o"],
    "ば": ["b", "a"], "び": ["b", "i"], "ぶ": ["b", "u"], "べ": ["b", "e"], "ぼ": ["b", "o"],
    "ぱ": ["p", "a"], "ぴ": ["p", "i"], "ぷ": ["p", "u"], "ぺ": ["p", "e"], "ぽ": ["p", "o"],
    "ま": ["m", "a"], "み": ["m", "i"], "む": ["m", "u"], "め": ["m", "e"], "も": ["m", "o"],
    "や": ["y", "a"], "ゆ": ["y", "u"], "よ": ["y", "o"],
    "ゃ": ["y", "a"], "ゅ": ["y", "u"], "ょ": ["y", "o"],
    "ら": ["r", "a"], "り": ["r", "i"], "る": ["r", "u"], "れ": ["r", "e"], "ろ": ["r", "o"],
    "わ": ["w", "a"], "ゐ": ["i"], "ゑ": ["e"], "を": ["o"], "ゎ": ["w", "a"],
    "ん": ["N"], "っ": ["cl"], "ゔ": ["v", "u"],
}

_SKIP_RE = re.compile(r"[\s、。,.!?！？・\-‐―「」『』()（）\[\]【】~〜]+")


def _katakana_to_hiragana(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    out = []
    for ch in text:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:  # ァ..ヶ
            out.append(chr(code - 0x60))
        else:
            out.append(ch)
    return "".join(out)


def _is_pure_kana(hiragana_text: str) -> bool:
    for ch in hiragana_text:
        if ch in _MONOS or ch == "ー" or _SKIP_RE.match(ch):
            continue
        return False
    return True


def kana_to_phones(text: str) -> list[str]:
    """Convert a hiragana/katakana phrase directly to model phoneme labels (no external dictionary)."""
    hira = _katakana_to_hiragana(text.strip())
    if not hira:
        raise ValueError("空の語句は登録できません。ひらがなまたはカタカナで入力してください。")
    if not _is_pure_kana(hira):
        raise ValueError("ひらがな・カタカナ以外の文字が含まれています。")

    out: list[str] = []
    i = 0
    n = len(hira)
    prev_was_break = True
    while i < n:
        ch = hira[i]
        if _SKIP_RE.match(ch):
            prev_was_break = True
            i += 1
            continue
        if ch == "ー":
            if out and out[-1] in VOWELS:
                out.append(out[-1])
            i += 1
            continue
        if i + 1 < n and hira[i:i + 2] in _DIGRAPHS:
            out.extend(_DIGRAPHS[hira[i:i + 2]])
            prev_was_break = False
            i += 2
            continue
        if ch in _MONOS:
            # Spoken Japanese long-o rule: 'o' + 'う' within a word is pronounced 'o o' (e.g. おはよう, びょういん)
            if ch == "う" and not prev_was_break and out and out[-1] == "o":
                out.append("o")
            else:
                out.extend(_MONOS[ch])
            prev_was_break = False
            i += 1
            continue
        raise ValueError(f"未対応の文字です: {ch!r}")

    if not out:
        raise ValueError("音素に変換できる文字がありません。")
    return out


def text_to_phones(text: str) -> list[str]:
    """Convert hiragana/katakana text to phoneme labels (no external dictionary required)."""
    hira = _katakana_to_hiragana(text.strip())
    if not hira:
        raise ValueError("ウェイクワードを入力してください。")
    if not _is_pure_kana(hira):
        raise ValueError(
            "ウェイクワードはひらがな・カタカナで入力してください（例:「オーケー、グーグル」「ヘイ、シリ」）。"
        )
    return kana_to_phones(hira)


def phone_ids(text: str) -> np.ndarray:
    """Convert Japanese text to integer token IDs for CTC alignment."""
    labels = text_to_phones(text)
    return np.array([TOKEN_TO_ID[p] for p in labels], dtype=np.int64)
