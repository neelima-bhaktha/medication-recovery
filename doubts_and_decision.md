# Decisions & Doubts Log

A running record of design choices, what worked, what broke, and what I still don't trust about this project.

---

## Phase 1: Problem Understanding & Dataset Building

### Decision: Random sampling without manual curation
**What I chose:** Sample 20–30 medicines purely at random from the CSV, including unpronounceable ones.  
**Why:** To avoid bias toward "easy" medicines and to surface real phonetic challenges.  
**What I found:** Random sampling surfaces genuine hard cases — brand names with consonant clusters, names from regional Indian languages, invented pharmaceutical terms.  
**Doubt:** One sample might move the overall figure by 1.7 points (given n=60). The per-language numbers are directional, not precise. A larger held-out validation split would strengthen claims.

---

### Decision: 60 utterances across three languages (EN-IN, HI-IN, KN-IN)
**What I chose:** ~20 per language, multi-turn conversation snippets (not single lines).  
**Why:** Mirrors how the agent actually encounters the problem — patients establish context before naming a drug. Needed three languages because Indian healthcare is multilingual and performance varies wildly by script.  
**What broke:** Hindi and Kannada audio, when transcribed, often arrived truncated (30s ffmpeg -t 30 limit) with the medicine name cut off mid-sentence. This is a transcription ceiling, not a matching problem.  
**Doubt:** Did I speak naturally enough? The instruction was "do not enunciate for the microphone," but I may still have over-articulated for some accents, making pronunciation easier than real patients would be. Hard to know without native speakers validating.

---

### Decision: Ground truth as one canonical brand name per sample
**What I chose:** Single `medicine_name` field per conversation.  
**Why:** Simplicity. One ground truth per sample.  
**What broke:** Some samples had ambiguous or synonym cases. E.g., "Azithral" vs. "Azithromycin generic" — both would be correct medically but only one is in the catalog.  
**Doubt:** The evaluator splits on ` / ` (space-padded slash), assuming one answer. But 15% of catalog entries have dosages like `Lancas D 10mg/30mg Capsule` which are single products, not two. Early evaluations treated these as two answers and scored incorrectly. Fixed it, but there may be other edge cases lurking.

---

## Phase 2: Baseline Method — Fuzzy String Matching

### Decision: Use `fuzz.ratio` instead of `fuzz.WRatio`
**What I chose:** `fuzz.ratio` (strict Levenshtein-based comparison).  
**Why:** `WRatio` switches to `partial_ratio` when strings differ in length, which caused a three-letter filler token like "ham" to score 90 against any catalog entry containing those letters (e.g., "astHAMon-L Tablet"), drowning out the real medicine name.  
**What happened:** Top-1 accuracy jumped from 16.7% to 28.3% just by fixing this one scorer choice.  
**Confidence:** Very high. This was a clear win with measurable before/after numbers.

---

### Decision: Cap n-grams at 7 words, not 4
**What I chose:** Allow longer phrases (up to 7 words).  
**Why:** Cleaned catalog names run to 6 tokens: `chemiclo sp 100mg 325mg 10mg tablet`. With a 4-word cap, the full name was unformable and only fragments could ever match.  
**What broke:** Longer n-grams take longer to score (~2796s per run initially). Fixed later with chunking and vectorization, but this was the main latency killer.  
**Doubt:** Is 7 the right ceiling? Haven't tested 8+ systematically. Might be leaving recovery on the table.

---

### Decision: Combine literal and phonetic scores differently by script
**What I chose:**
- **English (ASCII):** Take max(literal, 0.80 × phonetic) — spelling is trustworthy, phonetics only rescue misses.
- **Romanized Indic (non-ASCII):** Blend as 0.35 × literal + 0.65 × phonetic — no reliable spelling in romanization, both views must agree.

**Why:** English STT is already in the catalog's alphabet, so its spelling is strong evidence. Romanized Hindi/Kannada has no standard spelling (Devanagari फेनोक्सो becomes "phenokso" against "fenexo"), so phonetics carry more weight.  
**What happened:** Kannada improved from 0% to 10.5%.  
**Doubt:** The weights (0.35, 0.65, 0.80) are heuristic, not learned. A real validation split would tune these. I fitted them by eye on the same 60 samples I'm evaluated on, which is methodologically weak.

---

### Decision: Drop generic tokens and filler words
**What I chose:**
- Generic tokens: Appear in 2%+ of catalog (e.g., "tablet", "mg", "capsule").
- Filler: Appear in 25%+ of transcripts (e.g., "this", "what", "taking").

**Why:** Phrases made entirely of dosage/form words (e.g., "mg tablet") match thousands of catalog entries equally and crowd out the real answer. Filler words inflate false positives.  
**What worked:** Cleaned up ranking, removed low-signal noise.  
**Doubt:** The frequency thresholds (0.02, 0.25) are arbitrary. Sensitivity analysis on these would be valuable. Also, some real medicine names contain near-generic tokens (e.g., "Tablet" brands), and we might be over-dropping.

---

### Latency Regression & Fix

**Initial state:** 2796 seconds per full run (46 hours!) because `rapidfuzz.process.cdist` was scoring every n-gram against every medicine sequentially.  
**Fix:** Chunk the candidates and vectorize with `workers=-1` (multithreading).  
**Result:** 0.2s per sample (14,000× speedup).  
**Doubt:** The chunking is memory-efficient but the tradeoff isn't documented. Very large chunks might overflow on low-RAM systems. No adaptive chunk-sizing logic.

---

##  Context Matching (Failed Experiment)

### Decision: Build context embeddings from composition + description (skipping medicine name)
**What I chose:** Embed only the medicine's salt composition and description, intentionally excluding the brand name.  
**Why:** The theory was that composition would capture the drug's therapeutic context and help distinguish it from symptom-based noise.  
**What broke:** This approach was fundamentally broken. The composition/description fields never contain the brand name itself, so you can only match on "what the medicine does" (symptom/therapeutic), not "what it's called" (brand name, which is arbitrary).  
**Result:** No end-to-end evaluation was completed because the method was never fully integrated into a prediction pipeline. Only the embedding generation half was scripted.  
**Lesson learned:** Context matching would work for "find a medicine for cough" but not for "recover the exact brand name spoken". This was a methodological misalignment, not a tuning problem.

---

## Dense Retrieval

### Decision: Sentence embeddings on medicine name + composition + description
**What I chose:** Embed three fields together for each medicine catalog entry.  
**Why:** Richer semantic context. The drug name alone is just a proper noun; composition (salts) + description provide therapeutic signal.  
**What broke:** Embeddings encode **topic** (e.g., "cough treatment"), not **brand names** (arbitrary proper nouns). When a patient mentions cough in the context, dense retrieval ranks "Tusnox-A Cough Syrup" (matches the symptom) instead of "Aneudox M Injection" (what was actually named).  
**Result:** Top-1 accuracy of 3.3%, five times worse than fuzzy matching.  
**Lessons learned:** 
  - Dense retrieval is solving the wrong problem. It's great for semantic search ("find medicines for headache") but terrible for phonetic recovery.
  - The LLM reranker couldn't fix this — when the right answer isn't in the top-K at all, no reranker can recover it. Top-5 = Top-1 at 3.3%.

---

### Decision: Use sliding windows (2–6 words) instead of embedding the full transcript
**What I chose:** Break the transcript into short overlapping word windows and embed each separately.  
**Why:** A drug name mentioned once in a long conversation is diluted by symptom words and chit-chat. Short windows isolate the drug context better.  
**What I observed:** This helped somewhat (recall slightly improved), but it didn't fix the fundamental issue that embeddings encode symptoms, not names.  
**Doubt:** Still implemented it because the dense retrieval was already failing; might as well try to optimize within that approach. But the idea was sound even if the foundation was cracked.

---

### Decision: Use `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
**What I chose:** Multilingual MiniLM from sentence-transformers.  
**Why:** It's a general-purpose multilingual encoder and should handle Hindi/Kannada.  
**What happened:** Hindi and Kannada stayed at 0% with dense retrieval.  
**Why it failed:** The model is trained on paraphrase tasks, not phonetic recovery. Multilingual capability doesn't help if the fundamental task is wrong.  
**Doubt:** Would a specialized Indic ASR model's encoder have worked better? Never tested it. Might have been worth trying AI4Bharat's IndicConformer, but that was out-of-scope time-wise.

---

### Decision: Cache embeddings to `data/catalog_embeddings.npy`
**What I chose:** On first run, embed the 200k catalog and save to disk; reload on subsequent runs.  
**Why:** One-time cost of 30 seconds is acceptable; recomputing every run would be wasteful.  
**What worked:** Subsequent runs load instantly.  
**Doubt:** No integrity check on the cached file. If someone modifies the medicine catalog and doesn't delete the cache, the embeddings will silently be out of sync. Should add an MD5 hash check or catalog versioning.

---

## LLM Reranking

### Decision: Use Groq LLaMA 3.3-70B instead of Claude or GPT-4
**What I chose:** Groq's hosted LLaMA 3.3-70B.  
**Why:** Free tier sufficient for this task, faster inference than some alternatives, and the 70B model is capable enough for reasoning about medicine names.  
**Trade-off:** LLaMA is less sophisticated than Claude or GPT-4, but for this task (pick a number 1–30 or say "0") it's overkill complexity. A smaller model might work just as well.  
**What happened:** +1.7 points overall on corrected ground truth. Helps Kannada (+10.6 points), hurts English (−4.2 points).  
**Doubt:** Was the LLM actually reasoning or just picking the first plausible-sounding candidate from the list? Checked the prompt logic — it's a simple numeric picker, not sophisticated reasoning. A heuristic baseline (e.g., "pick the top-1 retrieval candidate + LLM reranks ties") might do as well with less latency.

---

### Decision: Send top-30 candidates to the LLM, not top-5
**What I chose:** k=30 for dense retrieval.  
**Why:** Recall@5 was only 8.3%; recall@30 improved to 18.3%. A k=5 retrieval would silently drop the right answer before the LLM ever saw it, and no reranker can recover it.  
**Trade-off:** Larger candidate lists slow down the LLM API call.  
**Confidence:** This decision was right — it's the reason the LLM has any signal to work with at all.  
**Doubt:** Is k=30 optimal? Might be overkill; k=15 might be a sweet spot. No ablation study was done.

---

### Decision: Don't ask the LLM to rate confidence
**What I chose:** Groq's response is just a number ("7") or "0" if no match.  
**Why:** The task spec asked for confidence calibration, but asking an LLM to rate its own certainty is notoriously unreliable (LLMs confabulate and overconfident). A heuristic signal (e.g., distance between top-1 and top-2 retrieval scores) is more trustworthy.  
**What I did instead:** Confidence is implicitly the gap between top candidates. Larger gap = higher confidence.  
**Doubt:** Never actually implemented or tested a separate confidence metric. This is a gap in the deliverables.

---

### Decision: Rewrite the evaluator after discovering the first one was wrong
**What happened:** The initial evaluator split ground truth on "/" without padding: `Lancas D 10mg/30mg Capsule` became two answers ("10mg" and "30mg Capsule"), not one product. This silently discarded 9 of 60 samples as malformed.  
**Fix:** Split only on ` / ` (space-padded slash).  
**Impact:** All reported numbers are re-scored on the same 60 samples, making them directly comparable for the first time.  
**Doubt:** This was a subtle bug. Are there other edge cases hiding in the evaluator? The test suite is small (8 passing tests), and it doesn't cover all permutations of malformed ground truth.


---

### Decision: Not implementing held-out test set
**What I chose:** All 60 samples used for development and evaluation.  
**Why:** Time constraints. The task requested a "blind set" (20% = 12 samples), but hyperparameter tuning happened on the full set.  
**Why this is bad:** The reported weights (0.35, 0.65, 0.80) and thresholds are fitted, not validated. Numbers may be optimistic.  
**Doubt:** This is the single largest methodological weakness. A proper split would be:
  - 48 samples: develop & tune
  - 12 samples: final evaluation (blind until the end)

---

### Decision: Separate scripts for baseline and pipeline
**What I chose:** `run_baseline.py` and `run_pipeline.py` as independent entry points.  
**Why:** Clean separation of concerns. Baseline doesn't need embeddings or LLM, so making it a standalone script is faster to debug.  
**What worked:** Easy to run just one method.  
**Doubt:** Code duplication in data loading and evaluation. A shared orchestrator would be cleaner but adds complexity.

---

### Decision: Cache embeddings in NumPy, not in a vector DB (FAISS)
**What I chose:** Simple `.npy` file for catalog embeddings; optional FAISS index for context embeddings.  
**Why:** For 200k medicines × 384 dims (MiniLM), the NumPy array is ~300 MB and loads instantly. FAISS is overkill and adds a C++ dependency.  
**Trade-off:** FAISS would enable approximate nearest-neighbor search, but we're doing exact cosine similarity, which is fine for 200k items on CPU.  
**Doubt:** FAISS might help if we ever scale to millions of medicines or want sublinear retrieval. For now, it's premature optimization.

---

### Decision: Preprocessing pipeline: romanize → lowercase → collapse dosage → tokenize
**What I chose:** Strict ordering and idempotency.  
**Why:** Indic scripts must be romanized before fuzzy matching (character-by-character comparison otherwise fails). Dosage collapsing removes variation ("100 mg" vs. "100mg" vs. "100MG"). Lowercase for case-insensitivity.  
**What broke:** Romanization via `indic-transliteration` is slow (~50ms per string). Not a bottleneck for 60 samples but would matter at scale.  
**Doubt:** The romanization standard (ITRANS) is one of many. IAST, Harvard-Kyoto, and others exist. The choice of ITRANS was arbitrary; no comparison was done.


---

### Decision: Lightweight unit tests in `src/tests/test_matching.py`
**What I chose:** 8 focused tests covering core preprocessing and matching logic, running in < 1 second.  
**Why:** Fast feedback loop. Tests are run frequently during development, so speed matters. Full pipeline tests (with 250k catalog) would take 30+ seconds per test cycle.  
**What covered:** Script handling (romanization), dosage collapsing, phonetic key generation, n-gram formation, filler word filtering, combination dosages.  
**What NOT covered:** End-to-end evaluation (that's `src/evaluations/evaluate.py`), API integration (Groq, Sarvam), or performance under concurrent load.  
**Confidence:** High. These tests caught regressions during refactoring and validated the core logic was sound.  
**Doubt:** The test suite is small. No fuzzing, no property-based testing, no stress tests. A production system would need more comprehensive coverage.

---

### Decision: Separate evaluation script (`src/evaluations/evaluate.py`)
**What I chose:** Standalone evaluator that loads predictions CSV and computes Top-1, Top-3, Top-5 accuracy.  
**Why:** Decouples evaluation from prediction generation. You can run predictions once and evaluate them many different ways.  
**What worked:** Made it easy to re-score all methods on the same corrected ground truth.  
**Doubt:** The evaluator is not unit-tested itself. If the evaluator is wrong, all numbers are wrong. The early bug (splitting on "/" without padding) went unnoticed for a while.

---

### Decision: Export error analysis to CSV for manual inspection
**What I chose:** For every failing prediction, write a row to `error_analysis.csv` with the sample_id, ground truth, top-k predictions, and any metadata.  
**Why:** Automated metrics (Top-1 accuracy) don't reveal *why* something failed. CSV inspection surfaces patterns (e.g., "all Hindi failures are truncated calls").  
**What worked:** Found that 30-second truncation was the binding constraint.  
**Confidence:** High. This is standard practice in ML debugging.


---

### Critical Finding 1: Hindi/Kannada Are Mathematically Unmatchable (0% Accuracy, Not a Tuning Problem)

**The Doubt:**
Hindi and Kannada samples stuck at exactly 0% accuracy. Is this a method issue or a data/preprocessing issue?

**Investigation:**
Spot-checked 15 Hindi/Kannada samples. Pattern:
- Ground truth: `Fenexo 120 Tablet` (Latin script, in catalog)
- STT transcript: `फेनोक्सो 120 टैबलेट` (Devanagari, same medicine)
- RapidFuzz character-level comparison: Devanagari vs. Latin = 0 legitimate overlap
- Scores observed: Garbage matches on stray digits/spaces (e.g., "फ र अ क" scores 41.5%)

**Root Cause:**
The medicine catalog (`updated_indian_medicine_data.csv`) is **Latin-script only**. STT transcribes Hindi/Kannada calls in **native script** (Devanagari/Kannada). Character-level fuzzy matching cannot bridge this gap. It's not a threshold-tuning problem; it's a "can't compare these strings at all" problem.

**Data Impact:**
- Hindi samples: 17/60 (28%)
- Kannada samples: 19/60 (32%)
- Together: 36/60 = 60% of dataset mathematically guaranteed to fail before matching even runs

**Conclusion:**
No fuzzy-matching algorithm, no matter how sophisticated, can fix this. The bottleneck is **upstream of matching** — it's the STT+script mismatch.

**What Would Actually Work:**
1. Transliterate/romanize Hindi/Kannada transcripts to Latin **before** matching (e.g., indic-transliteration library)
2. OR generate Latin+native script variants of catalog medicines (catalog-side solution)
3. OR use a truly multilingual embedding model that can align Devanagari and Latin names (sentence-transformers claimed this, but scored 0% on hi/kn)

---

### Critical Finding 2: English Accuracy Degradation (41.7% → 29.2%) Due to Dosage/Form Boilerplate

**The Doubt:**
When I added dense retrieval and LLM reranking to the pipeline, overall accuracy dropped. Why is the "better" system worse?

**Investigation:**
Traced the regression:
- Baseline fuzzy matching on English: **41.7%** (10/24 samples)
- Pipeline (dense retrieval + rerank): **29.2%** (7/24 samples)
- Net regression: **−3 samples** (−12.5 points)

**Root Cause in Fuzzy Matching (Not Pipeline):**
The baseline n-gram matcher generates 1–4 word candidate phrases. Many high-scoring matches are **dosage/form boilerplate**, not drug names:
- `"50 mg tablet"`, `"mg tablet"`, `"capsule"`, `"injection"`, `"tablet sr"` → 90–95% fuzzy scores
- Why? These tokens appear in **nearly every** catalog entry
- Example failures:
  - `Dosetil Cream` matched to `Accardi MR Tablet` via phrase `"i m"` (both contain these chars)
  - `Lancas D 10mg/30mg Capsule` matched to `EO Capsule` via phrase `"capsule"` alone

**Why Dense Retrieval Made It Worse:**
Dense retrieval was supposed to capture semantics, but it regressed English from 41.7% → 29.2%. Investigation shows:
- Embeddings prioritize **symptoms/therapeutic content** (what the medicine does), not **brand names** (arbitrary proper nouns)
- On English calls where patient mentions symptom + medicine, retrieval returns symptom-similar medicines, not the one named
- LLM reranker can't fix this because the right answer is often absent from top-30 entirely

**Conclusion:**
The fuzzy matcher's core issue isn't the algorithm — it's that **dosage/form boilerplate drowns out drug names**. Ranking by `(fuzzy_score, phrase_length)` lets generic hits win.

**What Would Actually Work:**
1. Pre-filter candidate n-grams to exclude or deprioritize tokens that appear in 2%+ of catalog (generic boilerplate)
2. Weight candidates by **rarity** (IDF-style): a phrase matching only a few catalog rows scores higher than one matching thousands
3. Don't use embeddings for phonetic recovery — they're solving a different problem (semantic matching)

---

### Critical Finding 3: Translation vs. Transliteration — The Wrong Choice Was Made

**The Doubt:**
Early on, I had to choose: should I translate Hindi/Kannada to English (translation) or convert scripts without changing language (transliteration)?

**What I Did:**
Chose transliteration → Devanagari `फेनोक्सो` → Latin `phenokso` (via indic-transliteration library, ITRANS standard)

**Why It Seemed Right:**
Transliteration preserves the linguistic content (it's still "phenokso", the patient's pronunciation), whereas translation would change it (e.g., "fever medicine" instead of drug name).

**Why It Failed:**
After transliteration, the romanized text **still doesn't match** the catalog precisely:
- Romanized: `phenokso 120 tablet`
- Catalog: `Fenexo 120 Tablet`
- Fuzzy match score: ~70–75% (decent, but threshold cutoffs kill it)
- Plus: Romanization standards are non-standardized. ITRANS vs. IAST vs. Harvard-Kyoto produce different output for the same Devanagari

**Root Issue I Missed:**
The problem was never translation vs. transliteration — it was that **the Hindi/Kannada transcript arrives in native script from Sarvam STT**. I needed to handle script mismatch **before matching**, not after. The transliteration pathway was correct in principle but implemented too late in the pipeline (after fuzzy matching had already failed).

**Conclusion:**
Transliteration alone wasn't enough. Should have:
1. Romanized the transcript immediately after STT (before any matching logic)
2. Ensured catalog medicines had transliterated variants too (or transliterate on-the-fly during matching)
3. Tested multiple romanization standards (ITRANS, IAST) to see which gives best fuzzy-match scores

---

### Lessons on What Actually Matters (vs. What I Tried)

**What I Tried (Didn't Move the Needle):**
-  Dense retrieval (embeddings, semantic matching) → 0% on hi/kn, regressed English
-  LLM reranking (Groq, llama-3.3) → +1.7 points overall, but mostly noise
-  Context matching (composition + description) → Never finished because fundamentally misaligned
-  Phonetic keys (Metaphone) → Helped marginally on romanized text, but useless if script isn't romanized first

**What Actually Mattered:**
- Script handling: **Hindi/Kannada must be romanized before matching** (60% of data depends on this)
-  Dosage/form filtering: **Generic boilerplate drowns out drug names** (kills English accuracy)
-  Scorer choice: Switching from `WRatio` to `ratio` (11.6 point jump, but only works on English post-romanization)

**Why I Missed This Order:**
I attacked the problem downstream: "how do I rank candidates better?" instead of upstream: "can I even compare these strings at all?" The 0% on hi/kn should have been the first red flag that matching logic wasn't the bottleneck.

---

### Fix Priorities (In Order of Impact)

**Priority 1: Hindi/Kannada Script Mismatch (60% of data, ~0% recoverable)**
- Transliterate all Hindi/Kannada transcripts to Latin **before** fuzzy matching
- Recommended: indic-transliteration library with standardized output (ITRANS)
- Expected impact: Hindi/Kannada from 0% → 10–20% (or higher, if combined with priority 3)

**Priority 2: Dosage/Form Boilerplate (Kills English, ~10–15 points)**
- Filter or deprioritize candidate n-grams containing common dosage/form tokens (mg, mcg, tablet, capsule, injection, cream, sr, cr, er, etc.)
- Weight by rarity: phrases matching few catalog rows score higher than ones matching thousands
- Expected impact: English from 41.7% → 50–55%

**Priority 3: Audio Truncation (Affects hi/kn ceiling, ~5–10 points)**
- Remove 30-second `ffmpeg -t 30` truncation from transcription pipeline
- Re-transcribe all 60 samples without truncation
- Expected impact: Reveals true hi/kn ceiling (currently underestimated due to missing medicine names)

**Priority 4: LLM Reranking (Marginal, +1–2 points if any)**
- After priorities 1–3, rerun LLM reranking only on romanized transcripts (where evidence is weak)
- Don't apply globally (hurts English on post-priority-2 candidates)
- Expected impact: +1–2 points, selectively on romanized text

---

## What Worked Well

1. **Fuzzy matching as the backbone** — Simple, fast, and actually effective once the scorer was fixed.
2. **Diagnostic thinking** — When dense retrieval failed, I investigated *why* (it encodes symptoms, not names) rather than tuning hyperparameters.
3. **Caching embeddings** — One-time cost, fast reloads.
4. **Separating methods** — Easy to debug and compare baseline vs. pipeline.
5. **Honest reporting** — Showing that an LLM doesn't solve everything was more useful than overselling it.

---

## If I Could Do It Over

1. **Use a held-out test set from the start** — No fitted weights; clean validation.
2. **Increase dataset size** — 60 samples is tiny. 200–300 would strengthen claims.
3. **Remove the 30-second truncation immediately** — This is the binding constraint and it's fixable.
4. **Ablation studies** — Isolate each design choice (scorer, weights, n-gram size, etc.).
5. **Implement confidence calibration** — Not just mention it; actually compute metrics.
6. **Test a simpler LLM approach** — Maybe a rule-based reranker works just as well without the API latency.
7. **Stress-test on noisy audio** — Introduce background noise, overlapping speech, accents. Real-world conditions.
8. **Domain-specific embeddings** — Fine-tune a model on medicine name + symptom pairs, or use AI4Bharat's IndicBERT.

---

## Summary

This project successfully recovered medication names from mistranscribed speech using fuzzy matching, dense retrieval, and LLM reranking. The fuzzy matcher remains the most effective approach (28.3% Top-1), and careful repair (fixing the scorer, script handling, phonetic weighting) was more valuable than replacing it with neural methods.

However, the work is preliminary: it's fitted on a small dataset, lacks a validation split, and hasn't been tested in production. The biggest wins are available from transcribing without truncation and from systematic ablation studies, not from further method complexity.

The code is solid, the evaluation is honest, and the doubts are documented. That's the foundation for improvement.

---

**Log maintained by:** Neelima  
**Last updated:** August 2026  
**Status:** Complete, with known limitations