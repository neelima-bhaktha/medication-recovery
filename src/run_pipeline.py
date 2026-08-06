import os
import sys
import json
import logging

import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils.data_loader import load_metadata
from retrieval.dense_retrieval import DenseRetriever
from generative.rerank import rerank

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main() -> None:
    """
    Retrieval + LLM rerank pipeline: dense-retrieve top-k candidates per transcript
    (conditioned on the full call, not isolated n-grams), then ask Claude to pick/correct
    the actual medicine among them using the transcript + language as context.
    """
    metadata_path = os.path.join("data", "transcripts", "metadata.csv")
    results_dir = "results"
    output_path = os.path.join(results_dir, "pipeline_predictions.csv")

    os.makedirs(results_dir, exist_ok=True)

    metadata_df = load_metadata(metadata_path)
    for col in ["recovered_medicine", "top_k_predictions"]:
        metadata_df[col] = metadata_df.get(col, pd.Series(dtype="object")).astype(object)

    logger.info("Loading dense retriever (embeds catalog once, cached to data/catalog_embeddings.npy)...")
    retriever = DenseRetriever()

    total_processed = 0
    llm_matched = 0

    for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Pipeline predictions"):
        transcript = row.get("stt_transcript")
        if pd.isna(transcript) or not str(transcript).strip():
            continue

        total_processed += 1
        transcript = str(transcript)

        # k=30: recall@k measured on this dataset plateaus around here (8.3%@5 -> 18.3%@30),
        # so a tight k=5 was silently dropping the right medicine before the LLM ever saw it.
        candidates = retriever.search(transcript, k=30)
        best = rerank(transcript, candidates, language=str(row.get("language", "")))

        # Put the LLM's pick first so downstream Top-1 evaluation reflects the final answer,
        # keep the rest of the retrieval ranking behind it for Top-3/Top-5 metrics.
        if best:
            candidates = sorted(candidates, key=lambda c: c["medicine"] != best)
            llm_matched += 1

        metadata_df.at[idx, "top_k_predictions"] = json.dumps(candidates)
        metadata_df.at[idx, "recovered_medicine"] = best

    metadata_df.to_csv(output_path, index=False)

    print("\n" + "=" * 45)
    print("Pipeline Prediction Summary")
    print("=" * 45)
    print(f"Total samples processed: {total_processed}")
    print(f"LLM found a match:       {llm_matched}")
    print(f"Output file location:    {output_path}")
    print("=" * 45 + "\n")


if __name__ == "__main__":
    main()
