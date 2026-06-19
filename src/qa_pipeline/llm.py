from __future__ import annotations

import json
import re
from typing import Any

from .models import Chunk, GeneratedQA, Verification


class LLMClient:
    def generate_questions(self, chunk: Chunk, questions_per_chunk: int) -> list[GeneratedQA]:
        raise NotImplementedError

    def verify_answer(self, chunk: Chunk, qa: GeneratedQA) -> Verification:
        raise NotImplementedError


class OpenAIClient(LLMClient):
    def __init__(self, model: str, verifier_model: str | None = None) -> None:
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model
        self.verifier_model = verifier_model or model

    def generate_questions(self, chunk: Chunk, questions_per_chunk: int) -> list[GeneratedQA]:
        prompt = f"""
You are creating a high-quality QA dataset from an SEC filing passage.

Generate {questions_per_chunk} questions that can be answered ONLY from the passage.
Prefer concrete financial, business, risk, segment, legal, accounting, or governance facts.
Use a mix of question types when the passage supports them:
- factual: direct facts, definitions, amounts, dates, product names, or entities
- comparison: changes across years, segments, products, regions, or categories
- causal: reasons the filing gives for an increase, decrease, risk, or business outcome
- risk: risk-factor questions answerable from the passage

Rules:
- The answer must be fully supported by the passage.
- Do not use outside knowledge.
- Do not ask vague questions such as "What does this section discuss?"
- Prefer questions that a financial analyst would actually ask.
- Evidence must be a short excerpt copied or closely paraphrased from the passage.

Return strict JSON with this shape:
{{"items":[{{"question":"...","answer":"...","evidence":"short exact or near-exact supporting passage","question_type":"factual|comparison|causal|risk"}}]}}

Section: {chunk.section_title}
Topic: {chunk.topic}

Passage:
{chunk.text}
"""
        data = self._json_completion(self.model, prompt)
        generated: list[GeneratedQA] = []
        for item in data.get("items", []):
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            evidence = str(item.get("evidence", "")).strip()
            question_type = normalize_question_type(str(item.get("question_type", "")).strip())
            if question and answer and evidence:
                generated.append(
                    GeneratedQA(
                        question=question,
                        answer=answer,
                        evidence=evidence,
                        question_type=question_type,
                    )
                )
        return generated[:questions_per_chunk]

    def verify_answer(self, chunk: Chunk, qa: GeneratedQA) -> Verification:
        prompt = f"""
You are a strict verifier. Decide whether the answer is fully supported by the source passage.

Labels:
- supported: the passage directly supports the answer.
- partial: the passage supports part of the answer, but not all of it.
- unsupported: the answer is not supported by the passage.

Return strict JSON:
{{"label":"supported|partial|unsupported","rationale":"one concise reason"}}

Question: {qa.question}
Answer: {qa.answer}
Claimed evidence: {qa.evidence}

Source passage:
{chunk.text}
"""
        data = self._json_completion(self.verifier_model, prompt)
        label = str(data.get("label", "unsupported")).strip().lower()
        if label not in {"supported", "partial", "unsupported"}:
            label = "unsupported"
        return Verification(label=label, rationale=str(data.get("rationale", "")).strip())

    def _json_completion(self, model: str, prompt: str) -> dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "Return valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception as exc:
            if exc.__class__.__name__ == "RateLimitError" and "insufficient_quota" in str(exc):
                raise RuntimeError(
                    "OpenAI API quota is unavailable for this key/project. "
                    "Check billing, credits, project limits, or use another API key."
                ) from exc
            raise
        content = response.choices[0].message.content or "{}"
        return json.loads(content)


class DryRunClient(LLMClient):
    def __init__(self) -> None:
        self.model = "dry-run"
        self.verifier_model = "dry-run"

    def generate_questions(self, chunk: Chunk, questions_per_chunk: int) -> list[GeneratedQA]:
        sentences = split_sentences(chunk.text)
        items: list[GeneratedQA] = []
        for sentence in sentences[:questions_per_chunk]:
            subject = chunk.section_title.replace("Item ", "item ")
            items.append(
                GeneratedQA(
                    question=f"What does the filing state in {subject}?",
                    answer=sentence,
                    evidence=sentence,
                    question_type="factual",
                )
            )
        return items

    def verify_answer(self, chunk: Chunk, qa: GeneratedQA) -> Verification:
        if qa.evidence and qa.evidence in chunk.text:
            return Verification(label="supported", rationale="The evidence appears verbatim in the chunk.")
        return Verification(label="unsupported", rationale="The evidence was not found verbatim in the chunk.")


def split_sentences(text: str) -> list[str]:
    candidates = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in candidates if 40 <= len(s.strip()) <= 350]


def normalize_question_type(value: str) -> str:
    normalized = value.lower()
    if normalized in {"factual", "comparison", "causal", "risk"}:
        return normalized
    return "factual"
