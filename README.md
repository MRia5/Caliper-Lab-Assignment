# Caliper Lab Assignment

Build a clean question-answer dataset from a raw SEC filing.

The pipeline:

1. Parses a raw HTML or text filing.
2. Splits it into section-aware chunks.
3. Uses an LLM to generate questions and answers from each chunk.
4. Uses a separate LLM verification pass to check whether each answer is actually supported by the source passage.
5. Writes a structured dataset as JSONL and CSV.

## Setup

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3 -m pip install -r requirements.txt
```

Set your API key:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

## Input

Save the raw SEC HTML file here:

```text
raw_inputs/aapl-20250927.htm
```

The pipeline checks for SEC automated-tool block pages and stops if the saved input is not the real filing.

## Run

```powershell
py -3 -m qa_pipeline raw_inputs/aapl-20250927.htm --output-dir outputs --questions-per-chunk 3
```

For a no-API smoke test:

```powershell
py -3 -m qa_pipeline raw_inputs/aapl-20250927.htm --output-dir outputs --dry-run
```

## Outputs

- `outputs/chunks.jsonl`: parsed section chunks with source metadata.
- `outputs/dataset.jsonl`: verified question-answer records.
- `outputs/dataset.csv`: spreadsheet-friendly version of the dataset.
- `outputs/rejected.jsonl`: generated records that failed verification.

Each accepted dataset row contains:

- `record_id`
- `source_file`
- `section_title`
- `chunk_id`
- `question`
- `answer`
- `evidence`
- `verification_label`
- `verification_rationale`
- `model`
- `verifier_model`
