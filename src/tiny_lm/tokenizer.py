"""Character-level tokenizer built from a corpus.

Vocabulary = sorted unique characters of the corpus, plus a reserved <unk>
slot at id 0 for characters never seen at build time.
"""

from __future__ import annotations

import json
from pathlib import Path

UNK_CHAR = "�"  # U+FFFD REPLACEMENT CHARACTER, decodes unknown ids
UNK_ID = 0


class CharTokenizer:
    def __init__(self, chars: list[str]) -> None:
        if not chars:
            raise ValueError("vocabulary must contain at least one character")
        self._chars: tuple[str, ...] = tuple(chars)
        self._stoi: dict[str, int] = {c: i + 1 for i, c in enumerate(self._chars)}

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        if not text:
            raise ValueError("corpus text is empty")
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self._chars) + 1  # + <unk>

    @property
    def unk_id(self) -> int:
        return UNK_ID

    def encode(self, text: str) -> list[int]:
        stoi = self._stoi
        return [stoi.get(c, UNK_ID) for c in text]

    def decode(self, ids: list[int]) -> str:
        chars = self._chars
        return "".join(chars[i - 1] if 1 <= i <= len(chars) else UNK_CHAR for i in ids)

    def save(self, path: Path | str) -> None:
        payload = {"version": 1, "chars": list(self._chars)}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> CharTokenizer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload["chars"])
