from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and verify a structured QA dataset from a raw SEC filing."
    )
    parser.add_argument("input", type=Path, help="Path to the raw HTML or text filing.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--questions-per-chunk", type=int, default=3)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--max-words", type=int, default=900)
    parser.add_argument("--overlap-words", type=int, default=120)
    parser.add_argument("--provider", choices=["openai", "gemini"], default="openai")
    parser.add_argument("--model", default=None)
    parser.add_argument("--verifier-model", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Run without calling an LLM.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    summary = run_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        questions_per_chunk=args.questions_per_chunk,
        max_chunks=args.max_chunks,
        max_words=args.max_words,
        overlap_words=args.overlap_words,
        provider=args.provider,
        model=args.model,
        verifier_model=args.verifier_model,
        dry_run=args.dry_run,
    )
    print(
        "Done: "
        f"{summary['chunks']} chunks, "
        f"{summary['accepted_records']} accepted records, "
        f"{summary['rejected_records']} rejected records."
    )
