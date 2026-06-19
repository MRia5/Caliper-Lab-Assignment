from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from .models import Chunk, DatasetRecord


def write_chunks(path: Path, chunks: list[Chunk]) -> None:
    write_jsonl(path, [asdict(chunk) for chunk in chunks])


def write_dataset(jsonl_path: Path, csv_path: Path, records: list[DatasetRecord]) -> None:
    rows = [asdict(record) for record in records]
    write_jsonl(jsonl_path, rows)
    write_csv(csv_path, rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
