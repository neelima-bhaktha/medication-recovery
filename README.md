# Medication Recovery from Mistranscribed Speech

Recovering medication names from noisy speech-to-text transcripts using fuzzy string matching, dense retrieval, and LLM-based reranking.

## Overview

This project addresses a critical problem in Indian healthcare: when patients speak medication names over the phone, speech-to-text models often garble them due to accent variation and unfamiliar phonology. The system recovers the intended medication name using:

1. **Baseline**: Phonetic & fuzzy string matching against a catalog of ~200,000 Indian medications
2. **Pipeline**: Dense retrieval (sentence embeddings) + LLM reranking (Groq/LLaMA 3.3)

The baseline fuzzy matcher remains the strongest performer after repair, while the LLM reranker adds +1.7 points overall and is particularly valuable for non-English scripts.

---

## Setup

### Prerequisites
- Python 3.9+
- ~2 GB disk space (medicine catalog + embeddings cache)

### 1. Clone & Install Dependencies

```bash
git clone <your-repo-url>
cd medication-recovery
pip install -r requirements.txt
```

**Key dependencies:**
- `sentence-transformers` — multilingual embeddings (paraphrase-multilingual-MiniLM-L12-v2)
- `rapidfuzz` — fast fuzzy string matching
- `groq` — LLM API client (for reranking)
- `pandas`, `numpy` — data handling
- `pydub`, `librosa` — audio processing (for transcription)

### 2. Download Medicine Dataset

The catalog of ~200,000 Indian medications is required:

```bash
mkdir -p data/raw
cd data/raw
wget "https://github.com/junioralive/Indian-Medicine-Dataset/raw/main/DATA/updated_indian_medicine_data.csv"
# Or download manually from: 
# https://github.com/junioralive/Indian-Medicine-Dataset/blob/main/DATA/updated_indian_medicine_data.csv
cd ../..
```

Expected location: `data/raw/updated_indian_medicine_data.csv`

### 3. Set Up API Keys

Create a `.env` file in the project root:

```bash
cat > .env << EOF
GROQ_API_KEY=your_groq_api_key_here
SARVAM_API_KEY=your_sarvam_api_key_here
EOF
```

**Get your keys:**
- **Groq**: Sign up at [console.groq.com](https://console.groq.com) → API keys
- **Sarvam AI**: Sign up at [sarvam.ai](https://sarvam.ai) → API dashboard

### 4. Prepare Transcripts (Optional)

If you have audio files and need transcripts:

```bash
python src/transcription/transcribe.py
```

This:
- Reads `.wav` or `.mp3` files from `data/audio/`
- Uses Sarvam STT API to generate transcripts
- Saves results to `data/transcripts/metadata.csv`

**Note:** Audio files are in `.gitignore` — you must provide your own or use the included metadata if transcripts already exist.

---

## Execution Order

### Quick Start: Run Both Methods on Existing Data

If you already have `data/transcripts/metadata.csv` with `stt_transcript` populated:

```bash
# 1. Run baseline (fuzzy matching)
python src/run_baseline.py

# 2. Evaluate baseline
python src/evaluations/evaluate.py

# 3. Run full pipeline (dense retrieval + LLM reranking)
python src/run_pipeline.py
```

### Step-by-Step: Complete End-to-End Pipeline

| Order | Script | Purpose | Output |
|-------|--------|---------|--------|
| 0 | `python src/transcription/transcribe.py` | **Optional**: STT transcription from audio | `data/transcripts/metadata.csv` |
| 1 | `python src/run_baseline.py` | Fuzzy string matching baseline | `results/baseline_predictions.csv` |
| 2 | `python src/evaluations/evaluate.py` | Evaluate baseline (Top-1, Top-3, Top-5 accuracy) | Console output + metrics |
| 3 | `python src/retrieval/context_matching.py` | **Optional**: Generate context embeddings (composition + description) | `data/context_embeddings.npy` |
| 4 | `python src/run_pipeline.py` | Dense retrieval + LLM reranking | `results/pipeline_predictions.csv` |

### Detailed Method Descriptions

#### Baseline: Fuzzy String Matching (`src/run_baseline.py`)

Uses [StringMatcher](src/baseline/string_match.py) to:
- Generate n-grams (1–7 words) from the transcript
- Score each against the medicine catalog using:
  - Literal string similarity (Levenshtein / fuzzy.ratio)
  - Phonetic matching (Metaphone key comparison)
- Weight phonetic vs. literal based on source script (English vs. romanized Indic text)
- Return top-K matches

**Performance:**
- Top-1 accuracy: **28.3%** overall
- English: 54.2% | Hindi: 11.8% | Kannada: 10.5%
- Latency: ~0.2s per sample

---

#### Context Matching: Composition + Description Embeddings (`src/retrieval/context_matching.py`)

Uses [context_matching.py](src/retrieval/context_matching.py) to:
- Build context text from medicine composition (salt composition) and description fields
- Encode context using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Embed the patient's transcript and find similar medicines by semantic/therapeutic content
- Return top-K matches based on composition similarity

**Performance:**
- Top-1 accuracy: **No end-to-end evaluation completed** (embedding generation only; ranking and evaluation stayed in notebook)
- Reason for 0% effective accuracy: Method was never fully implemented as a standalone pipeline
- Latency: Similar to dense retrieval (~0.3s per sample, after embeddings cached)

**Why it failed:**
Context matching encodes what a medicine *does* (composition, therapeutic use) rather than what it's *called* (brand name). The method works for queries like "find a medicine for cough" but not for "recover the exact brand name spoken". Since medicine names are arbitrary proper nouns with no semantic content, embeddings that capture therapeutic meaning cannot recover them. This is fundamentally misaligned with the task.

**Lesson learned:**
Semantic embeddings are the wrong tool for phonetic recovery of arbitrary brand names. The composition/description fields never contain the drug name itself, so this approach was doomed from the start.

#### Pipeline: Dense Retrieval + LLM Reranking (`src/run_pipeline.py`)

1. **Dense Retrieval** ([src/retrieval/dense_retrieval.py](src/retrieval/dense_retrieval.py)):
   - Encodes medicine names + composition + description using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
   - Computes sliding windows (2–6 words) from the transcript
   - Returns top-30 candidates by cosine similarity
   - Embeddings cached to `data/catalog_embeddings.npy` (one-time cost: ~30 seconds)

2. **LLM Reranking** ([src/generative/rerank.py](src/generative/rerank.py)):
   - Sends transcript + top-30 candidates to Groq (LLaMA 3.3-70B)
   - LLM picks the most likely medicine or returns "no match"
   - Result placed first in the final ranked list

**Performance:**
- Top-1 accuracy: **30.0%** (baseline 28.3% + LLM +1.7%)
- Kannada improves with LLM: 10.5% → 21.1%
- English slightly regresses: 54.2% → 50.0% (LLM second-guesses correct matches)
- Latency: ~0.8s per sample (including Groq API call)

---

## Project Structure

```
medication-recovery/
├── data/
│   ├── raw/
│   │   └── updated_indian_medicine_data.csv       # Medicine catalog (~200k entries)
│   ├── audio/                                      # Input audio files (user-provided, .gitignore)
│   ├── sampled/
│   │   └── sample_medicines.csv                   # 20–30 medicines sampled for annotation
│   ├── transcripts/
│   │   ├── metadata.csv                           # Ground truth: sample_id, medicine_name, stt_transcript, language
│   │   ├── context_embeddings.npy                 # Cached embeddings for composition+description
│   │   └── context_index.faiss                    # FAISS index (if using FAISS retrieval)
│   └── catalog_embeddings.npy                     # Cached embeddings for full medicine catalog
│
├── src/
│   ├── baseline/
│   │   ├── __pycache__/
│   │   └── string_match.py                        # StringMatcher class: fuzzy + phonetic matching
│   ├── evaluations/
│   │   ├── __pycache__/
│   │   └── evaluate.py                            # Top-1, Top-3, Top-5 accuracy + per-language breakdown
│   ├── generative/
│   │   ├── __pycache__/
│   │   └── rerank.py                              # LLM reranking via Groq API
│   ├── retrieval/
│   │   ├── __pycache__/
│   │   ├── context_matching.py                    # Context embeddings (composition + description)
│   │   └── dense_retrieval.py                     # DenseRetriever: sentence-transformers + cosine similarity
│   ├── tests/
│   │   └── test_matching.py                       # Unit tests for matching logic
│   ├── transcription/
│   │   └── transcribe.py                          # Sarvam STT pipeline
│   ├── utils/
│   │   ├── __pycache__/
│   │   ├── constants.py                           # Configuration constants
│   │   ├── data_loader.py                         # Load medicine dataset + transcripts
│   │   └── preprocessing.py                       # Text cleaning, romanization (ITRANS), phonetic keys
│   ├── run_baseline.py                            # Main entry point: baseline fuzzy matching
│   └── run_pipeline.py                            # Main entry point: dense retrieval + LLM reranking
│
├── results/
│   ├── baseline_predictions.csv                   # Baseline fuzzy matching output
│   ├── pipeline_predictions.csv                   # Dense retrieval + LLM reranking output
│   ├── pipeline_error_analysis.csv                # Error breakdown for pipeline predictions
│   ├── context_matching_predictions.csv           # Context matching embeddings output
│   ├── context_matching_error_analysis.csv        # Error breakdown for context matching
│   └── error_analysis.csv                         # General error analysis
│
├── docs/
│   └── method-review.html                         # Full method comparison & results
│
├── notebook/
│   └── dense_retrieval.ipynb                      # Jupyter notebook for exploration
│
├── .venv/                                         # Virtual environment (git ignored)
├── .vscode/                                       # VS Code settings
├── .env                                           # API keys (GROQ_API_KEY, SARVAM_API_KEY) — DO NOT COMMIT
├── .gitattributes                                 # Git attributes configuration
├── .gitignore                                     # Excludes audio/, .env, .venv
├── requirements.txt                               # Python dependencies
├── README.md                                      # Complete setup & execution guide
├── QUICKSTART.md                                  # 5-minute getting started guide
├── DECISIONS_AND_DOUBTS.md                        # Design decisions & reasoning                        
└── Raw prompts.md                                # Master checklist of outputs
```

---

## Key Configuration & Tuning

### Fuzzy Matching ([src/baseline/string_match.py](src/baseline/string_match.py))

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `MIN_CANDIDATE_CHARS` | 4 | Drop n-grams shorter than this (e.g., "ham", "the") |
| `SCORE_CUTOFF` | 55 | Discard matches scoring below this |
| `MAX_NGRAM_WORDS` | 7 | Longest n-gram to generate |
| `PHONETIC_WEIGHT` | 0.80 | Weight of phonetic matching vs. literal |
| `ROMANIZED_LITERAL_WEIGHT` | 0.35 | For Indic scripts: lower literal weight, rely on phonetics |
| `GENERIC_TOKEN_MIN_FREQ` | 0.02 | Drop tokens appearing in 2%+ of catalog (dosage words) |
| `FILLER_TOKEN_MIN_FREQ` | 0.25 | Drop tokens appearing in 25%+ of transcripts (filler words) |

### Dense Retrieval ([src/retrieval/dense_retrieval.py](src/retrieval/dense_retrieval.py))

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `MODEL_NAME` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Multilingual embeddings model |
| `WINDOW_SIZES` | (2, 3, 4, 5, 6) | Word lengths for sliding windows |
| `k` (in search) | 30 | Number of candidates to retrieve (k=5 was too tight; recall plateaus at k=30) |

### LLM Reranking ([src/generative/rerank.py](src/generative/rerank.py))

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `MODEL_NAME` | `llama-3.3-70b-versatile` | Groq-hosted LLaMA 3.3 70B |
| `max_tokens` | 10 | Limit output (just a number) |

---

## Hardware & Performance

### Tested On
- **Python**: 3.14.2
- **CPU**: Standard CPU (no GPU required)
- **RAM**: 8 GB minimum (embeddings + catalog fit comfortably)
- **Disk**: ~2 GB (medicine catalog + cached embeddings)
- **OS**: Windows / Linux / macOS (cross-platform)

**Note:** All runs performed on CPU. Dense retrieval was experimentally tested once on Google Colab GPU, but GPU acceleration is not required or optimized for. CPU performance is acceptable for this dataset size (200k medicines, 60 samples).

### Latency

| Stage | Cold Start | Warm Run | Per Sample | Notes |
|-------|-----------|----------|-----------|-------|
| **Baseline (`run_baseline.py`)** | 12–15s | 12–15s | ~0.2s | RapidFuzz + ITRANS + Metaphone |
| **Dense Retrieval (part of pipeline)** | — | ~0.1s | ~0.1s | Sliding windows + cosine similarity |
| **LLM Reranking (part of pipeline)** | — | ~0.4–0.6s | ~0.4–0.6s | Groq LLaMA 3.3-70B API call |
| **Full Pipeline (`run_pipeline.py`)** | 5–10 min | 30–45s | ~0.6s | Includes cached embeddings load + LLM |
| **Embeddings cache (one-time)** | 5–10 min | — | — | First run only: encode 250k medicines |
| **STT Transcription (`transcribe.py`)** | — | 1–2 min | ~1–2s per clip | Optional; Sarvam STT API |
| **Evaluation (`evaluate.py`)** | <1s | <1s | — | Compute Top-1, Top-3, Top-5 metrics |

---

## Evaluation & Results

### Reproduce the Results Table

After running both methods:

```bash
python src/evaluations/evaluate.py
```

This computes:
- **Top-1 accuracy**: Correct medicine is ranked first
- **Top-3 / Top-5 accuracy**: Correct medicine in top-K results
- **Per-language breakdown**: English, Hindi, Kannada separately
- **Error analysis**: Breakdown of which types of predictions fail

### Results Summary

| Method | Top-1 | Top-3 | EN | HI | KN | Status |
|--------|-------|-------|----|----|-----|--------|
| **Dense Retrieval (embeddings)** | 3.3% | 3.3% | 8.3% | 0.0% | 0.0% | Failed (semantic mismatch) |
| **Context Matching (composition)** | — | — | — | — | — | No end-to-end evaluation |
| **Baseline (Fuzzy matching)** | 28.3% | 35.0% | 54.2% | 11.8% | 10.5% | ✅ Best performer |
| **Pipeline (Fuzzy + LLM rerank)** | 30.0% | — | 50.0% | 11.8% | 21.1% | +1.7% improvement |

**Key findings:**
- **Fuzzy matching remains the backbone** — Simple, fast, effective. Careful repair (scorer choice, script handling, phonetic weighting) was more valuable than replacing it with neural methods.
- **Dense retrieval fails** — Embeddings encode symptoms/topics, not brand names. Five times worse than fuzzy matching.
- **Context matching never completed** — Composition/description fields don't contain the drug name itself. Fundamentally misaligned with the task.
- **LLM helps selectively** — Adds +1.7 points overall, helps romanized transcripts (+10.6 on Kannada), hurts English (−4.2 points). Works near its ceiling (88% extraction rate).
- **English has large headroom** — 54.2% vs. 96% potential; Hindi/Kannada are near ceiling due to transcription truncation (30s limit).
- **Highest-value next step** — Removing the 30-second audio truncation and re-transcribing would improve Hindi/Kannada ceilings, not just chase existing ones.

---

## Troubleshooting

### Issue: `FileNotFoundError: data/raw/updated_indian_medicine_data.csv`
**Solution**: Download the medicine dataset (see Setup section, step 2).

### Issue: `KeyError: GROQ_API_KEY` or `SARVAM_API_KEY`
**Solution**: Ensure `.env` file exists with correct keys (see Setup section, step 3).

### Issue: `ModuleNotFoundError: No module named 'sentence_transformers'`
**Solution**: Run `pip install -r requirements.txt`.

### Issue: Dense retrieval very slow on first run
**Solution**: Expected—it's building and caching embeddings for 200k medicines (~30s). Subsequent runs will load the cache.

### Issue: LLM reranking returns `None` for all samples
**Solution**: Check Groq API key and rate limits. Ensure `.env` contains valid `GROQ_API_KEY`.

### Issue: Low accuracy on non-English languages
**Solution**: This is a data constraint, not a model issue. The 30-second audio truncation often cuts off the medicine name in Hindi/Kannada samples. Re-transcribe without truncation to improve.

---

## Testing & Validation

### 1. Fast Unit & Regression Test Suite

Run the self-check suite to validate core matching and preprocessing logic:

```bash
python src/tests/test_matching.py
```

Or via pytest:
```bash
pytest src/tests/test_matching.py -v
```

**Runtime:** < 1 second (uses in-memory test catalog, no I/O)

**Tests (8 total, all passing):**
- `test_romanizes_indic_script` — Devanagari/Kannada → ITRANS romanization
- `test_collapses_spaced_dosage` — "40 mg" normalizes to "40mg"
- `test_phonetic_key_bridges_romanization` — Transliterated & English share Metaphone keys
- `test_digits_survive_phonetic_key` — Dosage numbers aren't dropped during phonetic reduction
- `test_filler_does_not_outrank_real_name` — Filler words ("ham", "hai") don't beat real drug names
- `test_generic_phrase_is_not_a_candidate` — Generic tokens ("tablet", "mg") filtered
- `test_full_length_name_is_reachable` — 6+ token drug names are formable by n-gram windows
- `test_combination_dosage_is_not_split_as_two_answers` — Dosages like "10mg/30mg" preserved

**What it validates:**
- Text preprocessing (cleaning, romanization, collapsing dosages)
- N-gram generation and filtering
- Phonetic key generation and matching
- Edge cases (empty input, very short candidates, malformed ground truth)

---

### 2. Accuracy & Evaluation Validation

Evaluate predictions against ground truth and generate error analysis:

```bash
python src/evaluations/evaluate.py
```

**Outputs:**
- Overall Top-1, Top-3, Top-5 accuracy across all 60 samples
- Per-language breakdown (English, Hindi, Kannada)
- Error analysis exported to `results/error_analysis.csv` for debugging

**To evaluate the pipeline predictions instead of baseline:**

```python
from src.evaluations.evaluate import Evaluator

Evaluator(
    predictions_path="results/pipeline_predictions.csv",
    errors_path="results/pipeline_error_analysis.csv"
).evaluate()
```

---

### 3. Embeddings Verification Script

Verify Sentence Transformer model and embedding generation:

```bash
python src/retrieval/context_matching.py
```

**Outputs:**
- Builds and validates `data/context_embeddings.npy`
- Prints embedding array dimensions, data types, and sample vector slices
- Verifies encoder integrity

---

### Full Validation Pipeline

To run all checks in sequence:

```bash
# 1. Unit tests (< 1 second)
python src/tests/test_matching.py

# 2. Baseline predictions & evaluation (15 seconds)
python src/run_baseline.py
python src/evaluations/evaluate.py

# 3. Full pipeline (30–45 seconds)
python src/run_pipeline.py

# 4. Pipeline evaluation
python -c "from src.evaluations.evaluate import Evaluator; Evaluator(predictions_path='results/pipeline_predictions.csv', errors_path='results/pipeline_error_analysis.csv').evaluate()"

# 5. Embeddings check (optional)
python src/retrieval/context_matching.py
```

**Total runtime:** ~1–2 minutes (mostly I/O and API calls)

---

## Development & Testing

### Code Quality Practices

The codebase follows these conventions:
- Type hints throughout for clarity
- Logging with `logging` module for debugging
- Vectorized NumPy operations (no slow Python loops)
- 8 unit tests in `src/tests/test_matching.py` (all passing)
- Docstrings for all major functions and classes

### Code Style

The project uses:
- Type hints throughout
- Logging with `logging` module
- NumPy for vectorized operations

---

## Reproducibility Notes

### Known Limitations
1. **Fitted weights**: Combination weights (phonetic vs. literal blend) are fitted on the same 60 samples they're evaluated on. A held-out validation split would strengthen generalization claims.
2. **30-second truncation**: Audio is truncated to 30 seconds before transcription, often cutting off medicine names in longer calls. This is the binding constraint for Hindi/Kannada performance.
3. **No reproducible seed**: Random medicine sampling and train/test splits are not seeded, so exact runs may vary slightly.

### To Improve Reproducibility
- Add `random.seed(42)` and `np.random.seed(42)` to entry points
- Create a validation split before hyperparameter tuning
- Document exact versions of all dependencies in `requirements.txt`

---

## License & Attribution

- Medicine dataset: [junioralive/Indian-Medicine-Dataset](https://github.com/junioralive/Indian-Medicine-Dataset)
- Sentence embeddings: [sentence-transformers](https://www.sbert.net/)
- Fuzzy matching: [rapidfuzz](https://github.com/maxbachmann/RapidFuzz)
- LLM: Groq + LLaMA 3.3

---

## Contact & Questions

For issues or questions:
1. Check the troubleshooting section above
2. Review logs in `results/` or console output
3. Check that all dependencies and data files are in place

---

**Last updated:** August 2026  
**Status:** Complete
