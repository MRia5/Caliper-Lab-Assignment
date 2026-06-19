from __future__ import annotations

from pathlib import Path

from .chunking import chunk_document
from .llm import DryRunClient, LLMClient, OpenAIClient
from .models import DatasetRecord
from .parsing import load_document
from .writers import write_chunks, write_dataset, write_jsonl


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    questions_per_chunk: int,
    max_chunks: int | None,
    max_words: int,
    overlap_words: int,
    model: str,
    verifier_model: str | None,
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
    if max_chunks is not None:
        chunks = chunks[:max_chunks]

    write_chunks(output_dir / "chunks.jsonl", chunks)

    llm: LLMClient
    if dry_run:
        llm = DryRunClient()
        generator_model = "dry-run"
        checker_model = "dry-run"
    else:
        llm = OpenAIClient(model=model, verifier_model=verifier_model)
        generator_model = model
        checker_model = verifier_model or model

    accepted: list[DatasetRecord] = []
    rejected: list[dict] = []

    for chunk in chunks:
        generated_items = llm.generate_questions(chunk, questions_per_chunk)
        for qa_index, qa in enumerate(generated_items):
            verification = llm.verify_answer(chunk, qa)
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
