import os
import sys
import logging
from typing import List, Union, Optional
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer

# Add the src directory to sys.path to enable imports when run directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_loader import load_medicine_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Consistent SentenceTransformer model matching Dense Retrieval
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def build_context_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs a 'context_text' column from 'salt_composition' and 'medicine_desc'.

    Args:
        df (pd.DataFrame): The medicine DataFrame containing 'salt_composition'
                           and 'medicine_desc' columns.

    Returns:
        pd.DataFrame: The DataFrame with the new 'context_text' column added.
    """
    df["context_text"] = (
        "Composition: " + df["salt_composition"].fillna("") +
        "\n\nDescription: " + df["medicine_desc"].fillna("")
    )
    return df


def generate_context_embeddings(
    texts: Union[List[str], pd.Series],
    model_name: str = EMBEDDING_MODEL_NAME,
    batch_size: int = 32,
    show_progress_bar: bool = True
) -> np.ndarray:
    """
    Generates sentence embeddings for given text sequences using SentenceTransformer.

    Args:
        texts (Union[List[str], pd.Series]): Texts to encode.
        model_name (str): SentenceTransformer model name.
        batch_size (int): Batch size for embedding computation.
        show_progress_bar (bool): Whether to display progress bar during encoding.

    Returns:
        np.ndarray: NumPy array containing computed embeddings.
    """
    logger.info(f"Loading SentenceTransformer model: '{model_name}'...")
    model = SentenceTransformer(model_name)

    if isinstance(texts, pd.Series):
        text_list = texts.tolist()
    else:
        text_list = list(texts)

    logger.info(f"Encoding {len(text_list)} text sequences (batch_size={batch_size})...")
    embeddings = model.encode(
        text_list,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=show_progress_bar
    )

    return embeddings


def save_embeddings(embeddings: np.ndarray, output_path: str) -> None:
    """
    Saves a NumPy embedding array to disk.

    Args:
        embeddings (np.ndarray): The embedding array to persist.
        output_path (str): File destination path (.npy).
    """
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        logger.info(f"Created directory: {out_dir}")

    np.save(output_path, embeddings)
    logger.info(f"Embeddings saved to: {output_path}")


def main(output_path: Optional[str] = None) -> np.ndarray:
    """
    Main pipeline to load medicine data, build context texts, generate
    sentence embeddings, print verification statistics, and save to disk.

    Args:
        output_path (Optional[str]): Custom path to save context_embeddings.npy.
                                     Defaults to data/context_embeddings.npy.

    Returns:
        np.ndarray: Computed embeddings array.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    medicine_path = os.path.join(base_dir, "data", "raw", "updated_indian_medicine_data.csv")

    if output_path is None:
        output_path = os.path.join(base_dir, "data", "context_embeddings.npy")

    # 1. Load dataset
    logger.info(f"Loading medicine dataset from: {medicine_path}")
    df = load_medicine_dataset(medicine_path)

    # 2. Build context text
    logger.info("Building 'context_text' column...")
    df = build_context_text(df)

    # 3. Generate embeddings
    embeddings = generate_context_embeddings(
        texts=df["context_text"],
        model_name=EMBEDDING_MODEL_NAME,
        batch_size=32,
        show_progress_bar=True
    )

    # 4. Print verification metrics
    print("\n" + "=" * 60)
    print("Embedding Verification Summary")
    print("=" * 60)
    print(f"Embedding Array Shape: {embeddings.shape}")
    print(f"Embedding Data Type:   {embeddings.dtype}")
    print(f"First 5 values of 1st embedding: {embeddings[0][:5]}")
    print("=" * 60 + "\n")

    # 5. Save embeddings
    save_embeddings(embeddings, output_path)

    return embeddings


if __name__ == "__main__":
    main()
