"""Import a Hugging Face dataset into an Agri Talk knowledge-base file.

The knowledge base is loaded at runtime from every ``*.json`` file in
``knowledge_base/``, so an imported file is picked up on the next app restart.

Only import datasets that carry a clear licence and whose content you are
allowed to redistribute. Always pass the dataset's own page as ``--source-url``
so each retrieved answer can be attributed.

Example (the licensed dataset currently shipped as hf_agri_qa.json):

    python scripts/import_hf_dataset.py \
        --dataset KisanVaani/agriculture-qa-english-only \
        --hf-file data/train-00000-of-00001.parquet \
        --answer-field answers \
        --source-title "KisanVaani agriculture Q&A dataset (Apache-2.0; community FAQ aggregation, not an official advisory)" \
        --source-url https://huggingface.co/datasets/KisanVaani/agriculture-qa-english-only \
        --out knowledge_base/hf_agri_qa.json

Requires ``pyarrow`` for parquet input (see requirements-dev.txt).
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

AGRI_TERMS = re.compile(
    r"(soil|crop|plant|seed|sow|irrigat|fertili|fertil|pest|disease|weed|harvest|"
    r"stor|compost|manure|rotat|drain|water|farm|grain|livestock|dairy|poultry)",
    re.IGNORECASE,
)
# Numeric agrochemical rates. The app deliberately avoids dose advice, so rows
# that state one are dropped instead of being surfaced as grounded fact.
DOSE_PATTERN = re.compile(
    r"\d+\s*(?:ml|l\b|litre|liter|kg|g\b|gm|gram|acre|ha\b|hectare|ppm|%)",
    re.IGNORECASE,
)
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "why", "how", "when",
    "which", "who", "do", "does", "did", "can", "should", "i", "my", "we", "our",
    "to", "of", "in", "on", "for", "and", "or", "it", "this", "that", "be",
}


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def slugify(text: str, limit: int = 40) -> str:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    slug = "-".join(words)[:limit].strip("-")
    return slug or "item"


def topics_from_question(question: str, extra=()) -> list:
    words = [
        word
        for word in re.findall(r"[a-z]{3,}", (question or "").lower())
        if word not in STOP_WORDS
    ]
    topics = []
    for item in list(extra) + words:
        if item and item not in topics:
            topics.append(item)
    return topics[:10]


def is_usable_row(question: str, answer: str, min_answer: int, max_answer: int) -> bool:
    question = (question or "").strip()
    answer = (answer or "").strip()
    if not (15 <= len(question) <= 200):
        return False
    if not (min_answer <= len(answer) <= max_answer):
        return False
    if not question.isascii() or not answer.isascii():
        return False
    if DOSE_PATTERN.search(answer):
        return False
    return bool(AGRI_TERMS.search(question))


def read_rows(path: Path):
    """Yield dict rows from a .jsonl, .csv or .parquet file."""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    elif suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle)
    elif suffix in {".parquet", ".pq"}:
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        for batch in table.to_batches(max_chunksize=2048):
            yield from batch.to_pylist()
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def build_entries(rows, source_title, source_url, min_answer, max_answer, answer_field="answers"):
    seen = set()
    used_ids = set()
    entries = []
    for row in rows:
        question = str(row.get("question") or "").strip()
        answer = str(row.get(answer_field) or row.get("answer") or "").strip()
        if not is_usable_row(question, answer, min_answer, max_answer):
            continue
        key = normalise(question)
        if key in seen:
            continue
        seen.add(key)
        entry_id = f"hf_{slugify(question)}"
        suffix = 2
        while entry_id in used_ids:
            entry_id = f"hf_{slugify(question)}-{suffix}"
            suffix += 1
        used_ids.add(entry_id)
        entries.append(
            {
                "id": entry_id,
                "title": question.rstrip("?").strip() + "?",
                "topics": topics_from_question(question),
                "content": answer,
                "source_title": source_title,
                "source_url": source_url,
            }
        )
    return entries


def write_kb(entries, out_path: Path):
    lines = ["["]
    for index, entry in enumerate(entries):
        lines.append("  {")
        lines.append(f'    "id": {json.dumps(entry["id"])},')
        lines.append(f'    "title": {json.dumps(entry["title"], ensure_ascii=False)},')
        lines.append(f'    "topics": {json.dumps(entry["topics"], ensure_ascii=False)},')
        lines.append(f'    "content": {json.dumps(entry["content"], ensure_ascii=False)},')
        lines.append(f'    "source_title": {json.dumps(entry["source_title"], ensure_ascii=False)},')
        lines.append(f'    "source_url": {json.dumps(entry["source_url"])}')
        lines.append("  }" + ("," if index < len(entries) - 1 else ""))
    lines.append("]")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Import a dataset into a knowledge-base JSON file.")
    parser.add_argument("--dataset", help="Hugging Face dataset repo id")
    parser.add_argument("--hf-file", help="File inside the dataset repository")
    parser.add_argument("--input", help="Local .parquet/.jsonl/.csv file instead of a download")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--answer-field", default="answers")
    parser.add_argument("--source-title", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-answer-chars", type=int, default=60)
    parser.add_argument("--max-answer-chars", type=int, default=700)
    parser.add_argument("--max-entries", type=int, default=0, help="0 keeps every gated row")
    args = parser.parse_args(argv)

    if args.input:
        path = Path(args.input)
    elif args.dataset and args.hf_file:
        from huggingface_hub import hf_hub_download

        path = Path(
            hf_hub_download(repo_id=args.dataset, filename=args.hf_file, repo_type="dataset")
        )
    else:
        parser.error("provide --input, or both --dataset and --hf-file")

    rows = list(read_rows(path))
    entries = build_entries(
        rows,
        args.source_title,
        args.source_url,
        args.min_answer_chars,
        args.max_answer_chars,
        args.answer_field,
    )
    entries.sort(key=lambda entry: entry["title"].lower())
    if args.max_entries:
        entries = entries[: args.max_entries]

    if not entries:
        print("No usable rows survived the filters; nothing written.", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    write_kb(entries, out_path)
    print(f"Read {len(rows)} rows, wrote {len(entries)} entries to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
