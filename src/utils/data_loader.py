import os
import pandas as pd

def _load_csv(path: str) -> pd.DataFrame:
    """
    Helper function to load a CSV file into a pandas DataFrame.

    Args:
        path (str): The file path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.

    Raises:
        FileNotFoundError: If the specified file does not exist at the path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found at the specified path: {path}")
    return pd.read_csv(path)

def load_metadata(path: str) -> pd.DataFrame:
    """
    Loads the metadata CSV file.

    Args:
        path (str): The file path to the metadata CSV.

    Returns:
        pd.DataFrame: The loaded metadata as a DataFrame.
    """
    return _load_csv(path)

def load_medicine_dataset(path: str) -> pd.DataFrame:
    """
    Loads the full medicine dataset CSV file.

    Args:
        path (str): The file path to the medicine dataset CSV.

    Returns:
        pd.DataFrame: The loaded medicine dataset as a DataFrame.
    """
    return _load_csv(path)

def load_sampled_medicines(path: str) -> pd.DataFrame:
    """
    Loads the sampled medicines CSV file.

    Args:
        path (str): The file path to the sampled medicines CSV.

    Returns:
        pd.DataFrame: The loaded sampled medicines dataset as a DataFrame.
    """
    return _load_csv(path)
