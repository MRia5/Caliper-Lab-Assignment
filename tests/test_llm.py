from qa_pipeline.llm import DryRunClient, normalize_question_type
from qa_pipeline.models import Chunk


def test_normalize_question_type_defaults_unknown_values() -> None:
    assert normalize_question_type("comparison") == "comparison"
    assert normalize_question_type("made-up") == "factual"


def test_dry_run_generation_includes_question_type() -> None:
    chunk = Chunk(
        chunk_id="chunk-001",
        source_file="sample.htm",
        section_title="Item 7. Management Discussion > Net Sales",
        topic="Net Sales",
        section_index=1,
        text="Net sales increased because iPhone and Services revenue grew. " * 10,
        start_char=0,
        end_char=600,
    )
    qa = DryRunClient().generate_questions(chunk, questions_per_chunk=1)[0]
    assert qa.question_type == "factual"
    assert qa.answer
