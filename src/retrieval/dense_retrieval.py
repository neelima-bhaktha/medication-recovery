import os
import sys
import logging
from typing import List, Dict

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_loader import load_medicine_dataset
from utils.preprocessing import clean_transcript

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Matches EMBEDDING_MODEL_NAME in context_matching.py
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDINGS_CACHE = os.path.join("data", "catalog_embeddings.npy")


WINDOW_SIZES = (2, 3, 4, 5, 6)

# Generic dosage/form words that appear in nearly every catalog entry. A window made
# entirely of these (e.g. "mg tablet") embeds close to ~all medicines and drowns out
# windows that actually contain the drug name, so such windows are dropped outright.
DOSAGE_FORM_STOPWORDS = {
    "mg", "mcg", "ml", "gm", "g", "iu", "tablet", "tablets", "capsule", "capsules",
    "injection", "syrup", "cream", "ointment", "gel", "drop", "drops", "suspension",
    "powder", "lotion", "sr", "cr", "er", "md", "mr", "dr", "xr", "od", "sachet",
    "strip", "of", "and", "the", "a", "an",
}


def _sliding_windows(transcript: str) -> List[str]:
    """
    Breaks a multi-turn call transcript into short word windows. Even single-sentence
    embeddings dilute a drug name once other words (symptoms, a second drug, chit-chat)
    share the sentence, so retrieval scores short windows and keeps each medicine's
    best-matching one instead of embedding the transcript as a whole.
    """
    words = clean_transcript(transcript).split()
    if not words:
        return [transcript]

    windows = set()
    for size in WINDOW_SIZES:
        if size > len(words):
            continue
        for i in range(len(words) - size + 1):
            chunk = words[i:i + size]
            # Require at least one word that isn't a number or dosage/form boilerplate,
            # otherwise the window carries no drug-identifying signal.
            if any(w not in DOSAGE_FORM_STOPWORDS and not w.isdigit() for w in chunk):
                windows.add(" ".join(chunk))

    return list(windows) or [transcript]


def _catalog_text(df: pd.DataFrame) -> pd.Series:
    """Short embedding text per medicine: name + composition + truncated description."""
    return (
        df["name"].fillna("") + ". " +
        df["salt_composition"].fillna("") + ". " +
        df["medicine_desc"].fillna("").str.slice(0, 300)
    )


class DenseRetriever:
    """Embeds the medicine catalog once (cached to disk) and ranks it against a query transcript."""

    def __init__(self, medicine_path: str = os.path.join("data", "raw", "updated_indian_medicine_data.csv")):
        self.df = load_medicine_dataset(medicine_path)
        self.model = SentenceTransformer(MODEL_NAME)
        self.embeddings = self._load_or_build_embeddings()

    def _load_or_build_embeddings(self) -> np.ndarray:
        if os.path.exists(EMBEDDINGS_CACHE):
            logger.info(f"Loading cached catalog embeddings from {EMBEDDINGS_CACHE}")
            return np.load(EMBEDDINGS_CACHE)

        logger.info(f"Embedding {len(self.df)} catalog entries (one-time cost, cached after)...")
        texts = _catalog_text(self.df).tolist()
        embeddings = self.model.encode(
            texts, batch_size=128, convert_to_numpy=True, show_progress_bar=True
        )
        os.makedirs(os.path.dirname(EMBEDDINGS_CACHE), exist_ok=True)
        np.save(EMBEDDINGS_CACHE, embeddings)
        return embeddings

    def search(self, transcript: str, k: int = 5) -> List[Dict]:
        """
        Returns the top-k catalog medicines ranked by embedding similarity to the transcript.
        Scores short sliding word-windows and keeps each medicine's best-matching window, so
        a drug name mentioned once in a long call isn't diluted by the rest of the conversation.
        """
        if not transcript or not isinstance(transcript, str) or not transcript.strip():
            return []

        windows = _sliding_windows(transcript)
        window_embeddings = self.model.encode(windows, convert_to_numpy=True)
        sims = cosine_similarity(window_embeddings, self.embeddings).max(axis=0)
        top_idx = np.argsort(sims)[-k:][::-1]

        return [
            {
                "medicine": str(self.df.iloc[i]["name"]),
                "salt_composition": str(self.df.iloc[i].get("salt_composition", "")),
                "score": float(sims[i]),
            }
            for i in top_idx
        ]
