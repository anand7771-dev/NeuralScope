"""
app/tokenization/service.py
────────────────────────────
High-level tokenization service used by model adapters.

``TokenizationService`` wraps any HuggingFace tokenizer and returns rich
metadata objects that drive the ``TOKENIZATION_COMPLETED`` event payload.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

__all__ = [
    "TokenizationResult",
    "TokenizationService",
]


# ── Result model ──────────────────────────────────────────────────────────────


class SpecialTokenInfo(BaseModel):
    """Metadata for a single special token found in the encoded sequence."""

    token: str = Field(..., description="String representation of the special token")
    id: int = Field(..., description="Vocabulary index")
    position: int = Field(..., description="Index in the ``input_ids`` list")


class TokenizationResult(BaseModel):
    """Complete tokenization metadata returned by ``TokenizationService.tokenize()``.

    All fields are JSON-serialisable and safe to embed directly in an
    ``InferenceEvent.data`` dictionary.
    """

    input_ids: List[int] = Field(..., description="Encoded token IDs")
    tokens: List[str] = Field(..., description="Individual decoded token strings")
    token_count: int = Field(..., description="Total number of tokens")
    special_tokens: List[SpecialTokenInfo] = Field(
        default_factory=list,
        description="Special tokens present in the encoded sequence",
    )
    bos_token: str = Field(..., description="Beginning-of-sequence token string")
    bos_token_id: int = Field(..., description="Beginning-of-sequence token ID")
    eos_token: str = Field(..., description="End-of-sequence token string")
    eos_token_id: int = Field(..., description="End-of-sequence token ID")
    vocab_size: int = Field(..., description="Tokenizer vocabulary size")
    tokenizer_type: str = Field(..., description="Tokenizer algorithm name (e.g. 'BPE')")
    encoding: str = Field(..., description="Human-readable encoding description")

    def to_event_data(self) -> Dict[str, Any]:
        """Return a plain dict suitable for use in ``InferenceEvent.data``."""
        return self.model_dump()


# ── Service ───────────────────────────────────────────────────────────────────


class TokenizationService:
    """Stateless service that wraps a HuggingFace tokenizer.

    Methods are intentionally stateless so that the same instance can be
    reused across concurrent requests (as long as the underlying tokenizer is
    thread-safe, which HuggingFace tokenizers are for inference).
    """

    # ── Public API ────────────────────────────────────────────────────────────

    @staticmethod
    def tokenize(text: str, tokenizer: Any) -> TokenizationResult:
        """Encode *text* and return a ``TokenizationResult`` with full metadata.

        Args:
            text:      Raw input string.
            tokenizer: A HuggingFace tokenizer instance (must implement
                       ``encode``, ``decode``, ``convert_ids_to_tokens``).

        Returns:
            ``TokenizationResult`` instance.
        """
        # Encode without adding special tokens so we control the sequence
        encoding = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids_tensor = encoding["input_ids"][0]  # shape: [seq_len]
        input_ids: List[int] = input_ids_tensor.tolist()

        tokens = TokenizationService.decode_tokens(input_ids, tokenizer)
        special_tokens = TokenizationService._find_special_tokens(input_ids, tokenizer)

        bos_token = getattr(tokenizer, "bos_token", "") or ""
        bos_token_id = getattr(tokenizer, "bos_token_id", 0) or 0
        eos_token = getattr(tokenizer, "eos_token", "") or ""
        eos_token_id = getattr(tokenizer, "eos_token_id", 0) or 0

        # Attempt to read vocab size; fall back to len(vocab)
        vocab_size: int = getattr(tokenizer, "vocab_size", None) or len(tokenizer.get_vocab())

        return TokenizationResult(
            input_ids=input_ids,
            tokens=tokens,
            token_count=len(input_ids),
            special_tokens=special_tokens,
            bos_token=bos_token,
            bos_token_id=bos_token_id,
            eos_token=eos_token,
            eos_token_id=eos_token_id,
            vocab_size=vocab_size,
            tokenizer_type="BPE",
            encoding="GPT-2 BPE (Byte-Pair Encoding)",
        )

    @staticmethod
    def decode_tokens(input_ids: List[int], tokenizer: Any) -> List[str]:
        """Decode each token ID individually to its string representation.

        Unlike ``tokenizer.decode(input_ids)`` which returns the whole string,
        this method returns one string per token, preserving spacing artefacts
        (e.g. ``"Ġworld"`` → ``" world"``).

        Args:
            input_ids: List of integer token IDs.
            tokenizer: HuggingFace tokenizer.

        Returns:
            List of decoded token strings, one per ID.
        """
        return [tokenizer.decode([token_id]) for token_id in input_ids]

    @staticmethod
    def get_special_tokens(tokenizer: Any) -> Dict[str, Any]:
        """Return a mapping of all special tokens defined by the tokenizer.

        Args:
            tokenizer: HuggingFace tokenizer.

        Returns:
            Dict mapping special token role (``"bos"``, ``"eos"``, …) to
            ``{"token": str, "id": int}``.
        """
        result: Dict[str, Any] = {}
        special_token_attrs = {
            "bos": ("bos_token", "bos_token_id"),
            "eos": ("eos_token", "eos_token_id"),
            "pad": ("pad_token", "pad_token_id"),
            "unk": ("unk_token", "unk_token_id"),
            "sep": ("sep_token", "sep_token_id"),
            "cls": ("cls_token", "cls_token_id"),
            "mask": ("mask_token", "mask_token_id"),
        }
        for role, (tok_attr, id_attr) in special_token_attrs.items():
            token_str = getattr(tokenizer, tok_attr, None)
            token_id = getattr(tokenizer, id_attr, None)
            if token_str is not None and token_id is not None:
                result[role] = {"token": token_str, "id": token_id}
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _find_special_tokens(
        input_ids: List[int],
        tokenizer: Any,
    ) -> List[SpecialTokenInfo]:
        """Identify special token positions within *input_ids*.

        Args:
            input_ids: Encoded sequence.
            tokenizer: HuggingFace tokenizer.

        Returns:
            List of ``SpecialTokenInfo`` for each special token found.
        """
        # Build a set of all special token IDs
        special_ids: Dict[int, str] = {}
        for attr in [
            "bos_token",
            "eos_token",
            "pad_token",
            "unk_token",
            "sep_token",
            "cls_token",
            "mask_token",
        ]:
            tok_str = getattr(tokenizer, attr, None)
            tok_id = getattr(tokenizer, f"{attr}_id", None)
            if tok_str is not None and tok_id is not None:
                special_ids[tok_id] = tok_str

        found: List[SpecialTokenInfo] = []
        for position, token_id in enumerate(input_ids):
            if token_id in special_ids:
                found.append(
                    SpecialTokenInfo(
                        token=special_ids[token_id],
                        id=token_id,
                        position=position,
                    )
                )
        return found
