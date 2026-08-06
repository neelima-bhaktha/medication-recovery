# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research/experimentation pipeline for recovering the medicine name a patient actually took from a noisy call-center transcript (English, Hindi, Kannada). Flow: audio → speech-to-text → candidate medicine matching → evaluation against ground truth.

No `requirements.txt`/`pyproject.toml` exists yet — dependencies are inferred from imports: `pandas`, `numpy`, `rapidfuzz`, `sentence-transformers`, `scikit-learn` (cosine_similarity), `requests`, `python-dotenv`, `tqdm`. Install these manually before running scripts.

## Running the pipeline

All commands run from the repo root (scripts append `src/` or repo root to `sys.path` internally):

```bash
# 1. Transcribe audio (data/audio/{hindi,english,kannada}/*.m4a) via Sarvam STT API
#    Requires .env with SARVAM_API_KEY, and ffmpeg on PATH (used to truncate/convert audio to 30s WAV)
python src/transcription/transcribe.py

# 2. Baseline: fuzzy string-match transcripts against the medicine dataset
python src/run_baseline.py

# 3. Evaluate predictions (Top-1/3/5 accuracy, per-language breakdown)
python src/evaluations/evaluate.py

# 4. Dense retrieval: build sentence-transformer embeddings for medicine context text
python src/retrieval/context_matching.py
```

There is no test suite, linter, or build step configured.

## Architecture

**Data flow is CSV-in / CSV-out, joined by `sample_id`:**
- `data/transcripts/metadata.csv` — the working table: `sample_id, audio_file, language, medicine_name (ground truth), stt_transcript, recovered_medicine, notes`. Scripts read and progressively fill columns onto this file (or write a copy to `results/`).
- `data/raw/updated_indian_medicine_data.csv` — the medicine catalog to match against (`name`, `salt_composition`, `medicine_desc`, etc.).
- `results/baseline_predictions.csv` — `run_baseline.py` output; adds `recovered_medicine`, `similarity_score`, `matched_phrase`, `phrase_length`, `top_k_predictions` (JSON list) to a copy of metadata.
- `results/error_analysis.csv` — `evaluate.py` output; rows where the Top-1 prediction didn't match ground truth.

**Two independent matching strategies against the same medicine catalog:**
1. **Baseline (`src/baseline/string_match.py`)** — `StringMatcher` builds an n-gram (1–4 word) candidate set from the cleaned transcript, fuzzy-matches each n-gram against medicine names via `rapidfuzz`, and aggregates the best score per medicine. This is what `run_baseline.py` runs today.
2. **Dense retrieval (`src/retrieval/`)** — embeds a `context_text` (composition + description, built in `context_matching.py`) or `embedding_text` (name + composition + description, prototyped in `notebook/dense_retrieval.ipynb`) per medicine using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, then ranks by cosine similarity to the transcript's embedding. This is exploratory (notebook) and the embedding-generation half is scripted in `context_matching.py`, but there's no corresponding retrieval/eval script yet — that logic currently only exists in the notebook.

**Ground-truth matching convention:** `medicine_name` in metadata can hold multiple correct answers separated by `/` (e.g. `"Aneudox M Injection / Urigo 40mg Tablet"`). `Evaluator.is_match` splits on `/` and cleans both sides via `clean_medicine_name` before comparing — always use this convention rather than exact string equality.

**Shared normalization (`src/utils/preprocessing.py`):** `normalize_text` → `clean_medicine_name`/`clean_transcript` (lowercase, strip punctuation, collapse whitespace) is the single text-cleaning path used by both matching strategies and evaluation. Any new matching logic should route through these rather than reimplementing cleaning.

**Path-append pattern:** each entry-point script under `src/` manually appends a directory to `sys.path` at the top (either `src/` itself or the repo root) before importing sibling modules — there's no installed package. Follow the existing pattern in a given file (`transcription/transcribe.py` imports as `src.utils...` from repo root; other scripts import as `utils...` from `src/`) rather than mixing styles within one file.

## Known rough edges (visible in code, not fixed)

- `src/transcription/transcribe.py` has a hardcoded Windows ffmpeg fallback path; `shutil.which("ffmpeg")` is tried first.
- `data/sampled/sample_medicines.py` hardcodes an absolute Windows path (`c:\medication-recovery`) — not portable, treat as a one-off script, not a reusable utility.
- Dense retrieval has no end-to-end script mirroring `run_baseline.py` + `evaluate.py`; only embedding generation is scripted, ranking/eval logic lives only in the notebook.
