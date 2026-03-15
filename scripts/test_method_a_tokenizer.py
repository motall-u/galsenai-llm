from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def token_to_surface(tokenizer: AutoTokenizer, token: str) -> str:
    try:
        return tokenizer.convert_tokens_to_string([token])
    except Exception:
        return token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test a saved Method A tokenizer by encoding and decoding a sentence "
            "passed on the command line."
        )
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Path to a saved tokenizer directory.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Optional Wolof run directory. If provided, the script loads "
            "`benchmark/method_a/tokenizer` by default."
        ),
    )
    parser.add_argument(
        "--text",
        required=True,
        help="Sentence to test with the tokenizer.",
    )
    parser.add_argument(
        "--show-json",
        action="store_true",
        help="Print the result as JSON instead of a plain-text summary.",
    )
    args = parser.parse_args()
    if args.tokenizer_path is None and args.run_dir is None:
        parser.error("Pass either --tokenizer-path or --run-dir.")
    return args


def resolve_tokenizer_path(args: argparse.Namespace) -> Path:
    if args.tokenizer_path is not None:
        return args.tokenizer_path.expanduser().resolve()
    assert args.run_dir is not None
    run_dir = args.run_dir.expanduser().resolve()
    method_a_path = run_dir / "benchmark" / "method_a" / "tokenizer"
    if method_a_path.exists():
        return method_a_path
    expected = run_dir / "benchmark" / "method_a" / "tokenizer"
    raise FileNotFoundError(
        f"Could not find a Method A tokenizer under {expected}."
    )


def main() -> None:
    args = parse_args()
    tokenizer_path = resolve_tokenizer_path(args)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)

    text = args.text
    input_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    surface_tokens = [token_to_surface(tokenizer, token) for token in tokens]
    decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
    pieces = [
        {
            "id": int(token_id),
            "raw_token": token,
            "surface": surface,
        }
        for token_id, token, surface in zip(
            input_ids,
            tokens,
            surface_tokens,
            strict=True,
        )
    ]

    payload = {
        "tokenizer_path": str(tokenizer_path),
        "text": text,
        "input_ids": input_ids,
        "tokens": tokens,
        "surface_tokens": surface_tokens,
        "pieces": pieces,
        "decoded": decoded,
        "round_trip_ok": decoded == text,
    }

    if args.show_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(f"Tokenizer: {tokenizer_path}")
    print(f"Text: {text}")
    print(f"Input IDs: {input_ids}")
    print(f"Tokens: {tokens}")
    print(f"Surface tokens: {surface_tokens}")
    print(f"Decoded: {decoded}")
    print(f"Round-trip OK: {payload['round_trip_ok']}")


if __name__ == "__main__":
    main()
