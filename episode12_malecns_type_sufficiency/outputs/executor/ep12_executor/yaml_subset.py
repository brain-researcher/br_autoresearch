"""A deliberately small YAML reader for EP12's dependency-free executor.

The repository policy uses only nested mappings, scalar sequences, and inline
scalar sequences.  Supporting only that declared subset keeps qualification
runnable with the Python standard library on Sherlock.  Unsupported YAML is
rejected rather than guessed.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any


class YamlSubsetError(ValueError):
    """Raised when a document leaves the supported policy subset."""


_INTEGER = re.compile(r"[-+]?\d+")
_FLOAT = re.compile(
    r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"
)


def _split_inline_list(text: str, line_number: int) -> list[str]:
    inner = text[1:-1].strip()
    if not inner:
        return []
    parts: list[str] = []
    start = 0
    quote: str | None = None
    escaped = False
    for index, character in enumerate(inner):
        if escaped:
            escaped = False
            continue
        if character == "\\" and quote:
            escaped = True
            continue
        if character in {"'", '"'}:
            if quote is None:
                quote = character
            elif quote == character:
                quote = None
            continue
        if character == "," and quote is None:
            parts.append(inner[start:index].strip())
            start = index + 1
    if quote is not None:
        raise YamlSubsetError(f"line {line_number}: unterminated quote")
    parts.append(inner[start:].strip())
    if any(not part for part in parts):
        raise YamlSubsetError(f"line {line_number}: empty inline-list item")
    return parts


def _scalar(text: str, line_number: int) -> Any:
    if text.startswith("["):
        if not text.endswith("]"):
            raise YamlSubsetError(f"line {line_number}: malformed inline list")
        return [_scalar(part, line_number) for part in _split_inline_list(text, line_number)]
    if text.startswith(("'", '"')):
        try:
            value = ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            raise YamlSubsetError(f"line {line_number}: malformed quoted scalar") from exc
        if not isinstance(value, str):
            raise YamlSubsetError(f"line {line_number}: quoted policy value must be text")
        return value
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "~"}:
        return None
    if _INTEGER.fullmatch(text):
        return int(text)
    if _FLOAT.fullmatch(text):
        return float(text)
    if any(token in text for token in ("{", "}", "&", "*", "!", "|", ">")):
        raise YamlSubsetError(
            f"line {line_number}: unsupported YAML feature in scalar {text!r}"
        )
    return text


def load_yaml_subset(path: Path) -> dict[str, Any]:
    """Load the mapping subset used by ``SEARCH_POLICY.yaml``.

    Tabs, duplicate keys, sequence-of-mapping syntax, anchors, tags, block
    scalars, and flow mappings are intentionally rejected.
    """

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    tokens: list[tuple[int, str, int]] = []
    for line_number, raw in enumerate(raw_lines, start=1):
        if "\t" in raw:
            raise YamlSubsetError(f"line {line_number}: tabs are not permitted")
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if indent % 2:
            raise YamlSubsetError(f"line {line_number}: indentation must use two spaces")
        tokens.append((indent, raw[indent:], line_number))

    if not tokens:
        raise YamlSubsetError("empty YAML document")
    if tokens[0][0] != 0:
        raise YamlSubsetError("top-level mapping must start at column zero")

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        if index >= len(tokens) or tokens[index][0] != indent:
            line_number = tokens[index][2] if index < len(tokens) else len(raw_lines)
            raise YamlSubsetError(f"line {line_number}: expected indentation {indent}")
        is_sequence = tokens[index][1].startswith("- ")
        container: Any = [] if is_sequence else {}

        while index < len(tokens):
            current_indent, content, line_number = tokens[index]
            if current_indent < indent:
                break
            if current_indent > indent:
                raise YamlSubsetError(f"line {line_number}: unexpected indentation")

            if is_sequence:
                if not content.startswith("- "):
                    raise YamlSubsetError(
                        f"line {line_number}: cannot mix mapping and sequence entries"
                    )
                item = content[2:].strip()
                if not item or ":" in item:
                    raise YamlSubsetError(
                        f"line {line_number}: only scalar sequence items are supported"
                    )
                container.append(_scalar(item, line_number))
                index += 1
                continue

            if content.startswith("- ") or ":" not in content:
                raise YamlSubsetError(f"line {line_number}: expected a mapping entry")
            key, raw_value = content.split(":", 1)
            key = key.strip()
            if not key or any(character.isspace() for character in key):
                raise YamlSubsetError(f"line {line_number}: invalid mapping key")
            if key in container:
                raise YamlSubsetError(f"line {line_number}: duplicate key {key!r}")
            raw_value = raw_value.strip()
            index += 1
            if raw_value:
                container[key] = _scalar(raw_value, line_number)
            else:
                if index >= len(tokens) or tokens[index][0] <= indent:
                    raise YamlSubsetError(f"line {line_number}: empty mapping value")
                if tokens[index][0] != indent + 2:
                    raise YamlSubsetError(
                        f"line {tokens[index][2]}: nested indentation must increase by two"
                    )
                value, index = parse_block(index, indent + 2)
                container[key] = value
        return container, index

    document, next_index = parse_block(0, 0)
    if next_index != len(tokens):
        raise YamlSubsetError(f"line {tokens[next_index][2]}: unparsed content")
    if not isinstance(document, dict):
        raise YamlSubsetError("top-level document must be a mapping")
    return document
