from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_file: str
    section_title: str
    section_index: int
    text: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class GeneratedQA:
    question: str
    answer: str
    evidence: str


@dataclass(frozen=True)
class Verification:
    label: str
    rationale: str


@dataclass(frozen=True)
class DatasetRecord:
    record_id: str
    source_file: str
    section_title: str
    chunk_id: str
    question: str
    answer: str
    evidence: str
    verification_label: str
    verification_rationale: str
    model: str
    verifier_model: str

