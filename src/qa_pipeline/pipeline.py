from __future__ import annotations

from pathlib import Path
import time

from .chunking import chunk_document
from .llm import DryRunClient, GeminiClient, LLMClient, OpenAIClient
from .models import DatasetRecord
from .models import GeneratedQA
from .models import Verification
from .parsing import load_document
from .writers import write_chunks, write_dataset, write_jsonl


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    questions_per_chunk: int,
    chunk_start: int,
    max_chunks: int | None,
    max_words: int,
    overlap_words: int,
    provider: str,
    model: str,
    verifier_model: str | None,
    verification_mode: str,
    request_delay: float,
    continue_on_error: bool,
    dry_run: bool,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    text = load_document(input_path)
    chunks = chunk_document(
        text=text,
        source_path=input_path,
        max_words=max_words,
        overlap_words=overlap_words,
    )
    if chunk_start < 0:
        raise ValueError("--chunk-start must be 0 or greater.")
    if max_chunks is not None:
        chunks = chunks[chunk_start : chunk_start + max_chunks]
    elif chunk_start:
        chunks = chunks[chunk_start:]

    write_chunks(output_dir / "chunks.jsonl", chunks)

    llm: LLMClient
    if dry_run:
        llm = DryRunClient()
        generator_model = "dry-run"
        checker_model = "dry-run"
    elif provider == "gemini":
        generator_model = model or "gemini-2.5-flash"
        checker_model = verifier_model or generator_model
        llm = GeminiClient(model=generator_model, verifier_model=checker_model)
    else:
        generator_model = model or "gpt-4o-mini"
        checker_model = verifier_model or generator_model
        llm = OpenAIClient(model=generator_model, verifier_model=checker_model)

    accepted: list[DatasetRecord] = []
    rejected: list[dict] = []

    for chunk in chunks:
        try:
            generated_items = llm.generate_questions(chunk, questions_per_chunk)
        except Exception as exc:
            if not continue_on_error:
                raise
            rejected.append(error_row(chunk, "generation_error", exc))
            sleep_between_requests(request_delay)
            continue

        sleep_between_requests(request_delay)
        for qa_index, qa in enumerate(generated_items):
            try:
                verification = verify_generated_answer(llm, chunk, qa, verification_mode)
            except Exception as exc:
                if not continue_on_error:
                    raise
                rejected.append(
                    {
                        **error_row(chunk, "verification_error", exc),
                        "question": qa.question,
                        "answer": qa.answer,
                        "evidence": qa.evidence,
                        "question_type": qa.question_type,
                    }
                )
                sleep_between_requests(request_delay)
                continue

            if verification_mode == "llm":
                sleep_between_requests(request_delay)
            row = DatasetRecord(
                record_id=f"{chunk.chunk_id}-qa-{qa_index:02d}",
                source_file=chunk.source_file,
                section_title=chunk.section_title,
                topic=chunk.topic,
                chunk_id=chunk.chunk_id,
                question=qa.question,
                answer=qa.answer,
                evidence=qa.evidence,
                question_type=qa.question_type,
                verification_label=verification.label,
                verification_rationale=verification.rationale,
                model=generator_model,
                verifier_model=checker_model,
            )
            if verification.label == "supported":
                accepted.append(row)
            else:
                rejected.append(
                    {
                        **row.__dict__,
                        "source_excerpt": chunk.text[:1200],
                    }
                )

    write_dataset(output_dir / "dataset.jsonl", output_dir / "dataset.csv", accepted)
    write_jsonl(output_dir / "rejected.jsonl", rejected)

    return {
        "chunks": len(chunks),
        "accepted_records": len(accepted),
        "rejected_records": len(rejected),
    }


def sleep_between_requests(request_delay: float) -> None:
    if request_delay > 0:
        time.sleep(request_delay)


def error_row(chunk, error_type: str, exc: Exception) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "source_file": chunk.source_file,
        "section_title": chunk.section_title,
        "topic": chunk.topic,
        "error_type": error_type,
        "error": str(exc),
        "source_excerpt": chunk.text[:1200],
    }


def verify_generated_answer(llm: LLMClient, chunk, qa: GeneratedQA, verification_mode: str) -> Verification:
    if verification_mode == "llm":
        return llm.verify_answer(chunk, qa)
    if verification_mode == "evidence":
        evidence = qa.evidence.strip()
        if evidence and evidence in chunk.text:
            return Verification("supported", "The evidence appears verbatim in the source chunk.")
        return Verification(
            "partial",
            "The answer was generated from the chunk, but the evidence was not found verbatim.",
        )
    return Verification("not_verified", "Verification was skipped for this run.")
