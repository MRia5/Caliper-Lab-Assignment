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
py -3 -m pip install -e .
```

Set your API key:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

For Gemini instead:

```powershell
$env:GEMINI_API_KEY="your_gemini_api_key_here"
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

Run with Gemini:

```powershell
py -3 -m qa_pipeline raw_inputs/aapl-20250927.htm --provider gemini --model gemini-2.5-flash-lite --output-dir outputs_gemini --questions-per-chunk 3 --max-chunks 1
```

After the one-chunk smoke test works, run the full filing:

```powershell
py -3 -m qa_pipeline raw_inputs/aapl-20250927.htm --provider gemini --model gemini-2.5-flash-lite --output-dir outputs_gemini --questions-per-chunk 3
```

For Gemini free-tier stability, throttle requests and keep going if one request fails:

```powershell
py -3 -m qa_pipeline raw_inputs/aapl-20250927.htm --provider gemini --model gemini-2.5-flash-lite --output-dir outputs_gemini --questions-per-chunk 1 --request-delay 8 --continue-on-error
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
- `topic`
- `chunk_id`
- `question_type`
- `question`
- `answer`
- `evidence`
- `verification_label`
- `verification_rationale`
- `model`
- `verifier_model`
