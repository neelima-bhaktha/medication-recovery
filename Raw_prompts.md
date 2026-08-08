# Raw Prompts Log

---

## 1. Data Loader (`src/utils/data_loader.py`)

Build src/utils/data_loader.py for my medication recovery project.

### Requirements:

- Implement only this file.
- Create three functions:
  - `load_metadata(path: str) -> pd.DataFrame`
  - `load_medicine_dataset(path: str) -> pd.DataFrame`
  - `load_sampled_medicines(path: str) -> pd.DataFrame`
- Each function should:
  - Use `pandas.read_csv()`
  - Return a `pandas.DataFrame`
  - Check if the file exists and raise `FileNotFoundError` with a meaningful message if it doesn't.
  - Include type hints and concise docstrings.
  - Follow clean code principles (PEP 8, readable, modular, no duplicate code).
  - Do not perform preprocessing, validation, cleaning, logging, or business logic. This module is solely responsible for loading CSV files.
- At the end, briefly explain the design decisions and how this module will be used by the rest of the project.

---

## 2. Preprocessing Utilities (`src/utils/preprocessing.py`)

Build src/utils/preprocessing.py for my medication recovery project.

### Requirements:

- Implement only this file.
- Create reusable preprocessing functions for medicine names and transcripts:
  - `normalize_text(text: str) -> str`
  - `clean_medicine_name(name: str) -> str`
  - `clean_transcript(transcript: str) -> str`
- Functions should:
  - convert to lowercase
  - strip leading/trailing whitespace
  - normalize multiple spaces
  - remove unnecessary punctuation while preserving meaningful characters (e.g., numbers and dosage units like 500mg)
  - safely handle None or empty strings
- Use type hints, concise docstrings, and follow PEP 8.
- Keep the module generic and reusable. Do not include any matching, retrieval, file I/O, logging, or project-specific business logic.
- Briefly explain the purpose of each function and why this preprocessing is useful for later string matching and retrieval.

> ok done

---

## 3. Baseline String Matcher (`src/baseline/string_match.py`)

Build src/baseline/string_match.py for my medication recovery project.

### Requirements:

- Implement only this file.
- Use RapidFuzz for fuzzy string matching.
- The module should expose a class `StringMatcher`.
- The constructor should accept the medicine dataset (DataFrame or list of medicine names).
- Implement:
  - `find_best_match(transcript: str) -> tuple[str, float]`
  - Returns the predicted medicine name and its similarity score.
- Reuse preprocessing utilities from `utils/preprocessing.py`.
- Keep matching logic modular so different scorers (e.g., WRatio, token_sort_ratio) can be swapped later.
- Handle empty or invalid transcripts gracefully.
- Use type hints, concise docstrings, and follow PEP 8.
- Do not include evaluation, CLI code, file I/O, or transcription logic.
- Briefly explain the algorithm, why RapidFuzz was chosen as the baseline, its time complexity, and its limitations compared to embedding-based retrieval.

---

## 4. Evaluator Module (`src/evaluation/evaluate.py`)

Build src/evaluation/evaluate.py for my medication recovery project.

### Requirements:

- Implement only this file.
- Create a class `Evaluator`.
- The evaluator should compare the ground-truth medicine (`medicine_name`) with the predicted medicine (`recovered_medicine`).
- Implement methods to:
  - calculate overall accuracy
  - identify incorrect predictions
  - generate a summary report (total samples, correct, incorrect, accuracy)
- Return results as dictionaries/DataFrames rather than printing directly.
- Use type hints, concise docstrings, and follow PEP 8.
- Keep the evaluator independent of any specific recovery algorithm so it can evaluate `StringMatcher`, `DenseRetriever`, and `ContextMatcher` without modification.
- Do not include visualization, CLI code, or file I/O.
- Briefly explain the design and how this evaluator will support comparing multiple recovery methods later.

---

## 5. Dense Retrieval Iteration & Module (`src/retrieval/dense_retrieval.py`)

> doesnt dense retrival need sarvam api key?
> 
> refine the prompt for dense retrieval 

Build src/retrieval/dense_retrieval.py for my medication recovery project.

### Requirements:

- Implement only this file.
- Create a class `DenseRetriever`.
- Use the `sentence-transformers` library to generate embeddings.
- The constructor should accept the medicine dataset and preload embeddings for all medicine names.
- Use a lightweight model suitable for semantic retrieval (e.g., `all-MiniLM-L6-v2`), but keep the model name configurable.
- Implement:
  - `build_index()` – preprocess medicine names and generate embeddings.
  - `find_best_match(transcript: str) -> tuple[str, float]` – preprocess the transcript, encode it, compute cosine similarity against the medicine embeddings, and return the best matching medicine and similarity score.
- Reuse preprocessing utilities from `utils/preprocessing.py`.
- Use cosine similarity for now. Do not implement FAISS yet.
- Store embeddings in memory so they are generated only once.
- Handle empty or invalid transcripts gracefully.
- Use type hints, concise docstrings, and follow PEP 8.
- Do not include evaluation, CLI code, file I/O, or transcription logic.
- Briefly explain the architecture, why Sentence Transformers were chosen, why cosine similarity is sufficient for the first version, and how FAISS can later replace only the similarity search layer without changing the rest of the code.

---

## 6. Audio Transcription Pipeline (`src/transcription/transcribe.py`)

> now what do i do then, btw i got the api key

Build src/transcription/transcribe.py for my medication recovery project.

### Requirements:

- Read the Sarvam API key from a `.env` file using `python-dotenv`.
- Read `metadata.csv` using the existing `data_loader.py`.
- Iterate through the metadata rows and locate each audio file.
- Transcribe audio using the Sarvam Speech-to-Text API.
- For now, only process rows where `language == "Hindi"`.
- Include clearly commented code blocks for processing English and Kannada, but keep them disabled so they can be enabled later with minimal changes.
- Write the returned transcript into the `stt_transcript` column for the corresponding row.
- Skip rows that already contain a transcript.
- Handle API errors gracefully and continue with the next file.
- Save the updated metadata back to CSV after each successful transcription to avoid losing progress if credits run out or execution stops.
- Use type hints, concise docstrings, logging, and clean modular functions.
- Do not implement any medicine recovery, evaluation, or retrieval logic.
- At the end, briefly explain how to enable English and Kannada transcription by uncommenting the relevant lines.

---

## 7. Metadata Debugging & Validation Notes

> check the .csv file for yourself.
> 
> there are three things i noticed in english section, i bymistakenly added 1 kn and 1 hn audio into the english section and they have been transcribed to their respective language. S017,Nbenoquin and linzid en.m4a is missing the transcription. 
> 
> what you can also do is, give a prompt for gemini with the rules to check the metadata and ill make gemini check for any errors

---

## 8. Run Baseline Script (`src/run_baseline.py`)

Build src/run_baseline.py for my medication recovery project.

### Requirements:

- Use the existing `load_metadata()` and `load_medicine_dataset()` functions from `utils/data_loader.py`.
- Import and use the existing `StringMatcher` class from `baseline/string_match.py`.
- Load `metadata.csv` and the medicine dataset.
- Initialize `StringMatcher` once.
- Iterate through every row in `metadata.csv`.
- For each non-empty `stt_transcript`, call `find_best_match()` and store:
  - the predicted medicine in `recovered_medicine`
  - the similarity score in a new column called `similarity_score`.
- Skip rows with empty transcripts.
- Save the results to `results/baseline_predictions.csv`.
- Create the `results/` directory if it does not exist.
- Do not overwrite the original `metadata.csv`.
- Display a progress bar using `tqdm`.
- Print a short summary at the end:
  - total samples processed,
  - successful predictions,
  - skipped samples,
  - output file location.
- Use type hints, logging, clean functions, and PEP 8.
- Do not compute evaluation metrics or accuracy in this script. Its sole purpose is to generate baseline predictions.

---

## 9. Baseline Evaluation Script (`src/evaluation/evaluate_baseline.py`)

Build src/evaluation/evaluate_baseline.py for my medication recovery project.

### Objective:
Evaluate the performance of the baseline RapidFuzz retrieval using the generated `baseline_predictions.csv`.

### Requirements:

- Read `results/baseline_predictions.csv`.
- Compare the ground-truth `medicine_name` with:
  - the Top-1 prediction (`recovered_medicine`)
  - the Top-3 predictions
  - the Top-5 predictions from the `top_k_predictions` column.
- Compute:
  - Top-1 Accuracy
  - Top-3 Accuracy
  - Top-5 Accuracy
- Also compute accuracy broken down by language (en, hi, kn).
- Produce a confusion summary containing:
  - total samples
  - correct predictions
  - incorrect predictions
  - skipped samples (if any)
- Generate a CSV named `results/error_analysis.csv` containing only incorrect predictions with the following columns:
  - `sample_id`
  - `language`
  - `medicine_name`
  - `recovered_medicine`
  - `similarity_score`
  - `matched_phrase`
  - `phrase_length`
  - `top_k_predictions`

---

## 10. Pipeline Architecture Concept i told chat gpt then it gave me this outline

```
Transcript
      │
      ▼
split_transcript()
      │
 ┌────┴────┐
 ▼         ▼
Context   Medicine phrase
 │            │
 ▼            ▼
FAISS     StringMatcher
 │            │
 └────┬───────┘
      ▼
 Final prediction
```

> what do i do now

---

## 11. Additional Notes

LLM part was done in another laptop so no record of the prompt found, dense retreival was alot more arguing with the AI (Chat gpt - always confused for some reason)