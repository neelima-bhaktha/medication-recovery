import os
import sys

# Add the project root to sys.path so the 'src' module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import logging
import pandas as pd
import requests
import tempfile
import subprocess
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

from src.utils.data_loader import load_metadata

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SARVAM_STT_API_URL = "https://api.sarvam.ai/speech-to-text"

def get_audio_file_path(base_dir: str, language: str, audio_filename: str) -> str | None:
    """
    Constructs the path to the audio file based on the language.
    
    Args:
        base_dir (str): Base directory containing audio files.
        language (str): Language code (e.g., 'hi', 'en', 'kn').
        audio_filename (str): Name of the audio file.
        
    Returns:
        str | None: The constructed file path, or None if the language is unsupported.
    """
    # NOTE: To enable English and Kannada transcription, uncomment the respective blocks below
    if language == 'hi':
        return os.path.join(base_dir, 'hindi', audio_filename)
    elif language == 'en':
        return os.path.join(base_dir, 'english', audio_filename)
    elif language == 'kn':
        return os.path.join(base_dir, 'kannada', audio_filename)
    else:
        return None

def transcribe_audio(audio_path: str, api_key: str) -> str | None:
    """
    Transcribes an audio file using the Sarvam Speech-to-Text API.
    
    Args:
        audio_path (str): The absolute or relative path to the audio file.
        api_key (str): The Sarvam API key.
        
    Returns:
        str | None: The transcribed text if successful, None otherwise.
    """
    headers = {
        "api-subscription-key": api_key
    }
    
    try:
        # Force conversion to WAV. This ensures exact truncation without residual metadata
        # from the original container that might trick the API's duration check.
        content_type = 'audio/wav'
        temp_suffix = ".wav"
        
        with tempfile.NamedTemporaryFile(suffix=temp_suffix, delete=False) as temp_audio:
            temp_path = temp_audio.name
            
        try:
            import shutil
            # Try to find ffmpeg in PATH, fallback to the absolute path where winget installed it
            ffmpeg_path = shutil.which("ffmpeg") or r"C:\Users\HP\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin\ffmpeg.exe"
            
            # Use ffmpeg directly to truncate to 30s and convert to WAV.
            subprocess.run(
                [ffmpeg_path, "-y", "-i", audio_path, "-t", "30", temp_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            with open(temp_path, 'rb') as audio_file:
                files = {'file': (os.path.basename(audio_path), audio_file, content_type)}
                response = requests.post(SARVAM_STT_API_URL, headers=headers, files=files)
                
            if response.status_code == 200:
                data = response.json()
                return data.get("transcript", "")
            else:
                logger.error(f"API error for {os.path.basename(audio_path)}: {response.status_code} - {response.text}")
                return None
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        logger.error(f"Exception during transcription for {os.path.basename(audio_path)}: {e}")
        return None

def main() -> None:
    """
    Main function to read metadata, process audio files for transcription,
    and save the results back to the CSV.
    """
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        logger.error("SARVAM_API_KEY not found in environment variables. Please check your .env file.")
        return

    # Define paths
    # Using relative paths from the root of the project
    metadata_path = os.path.join("data", "transcripts", "metadata.csv")
    audio_base_dir = os.path.join("data", "audio")
    
    # Load metadata
    try:
        df = load_metadata(metadata_path)
        # Force the column to be of object type to avoid TypeError when assigning string transcripts
        df['stt_transcript'] = df['stt_transcript'].astype(object)
    except FileNotFoundError:
        logger.error(f"Metadata file not found at {metadata_path}")
        return

    # Iterate through metadata rows
    for index, row in df.iterrows():
        # Skip rows that already have a transcript
        if pd.notna(row['stt_transcript']) and str(row['stt_transcript']).strip() != "":
            logger.info(f"Skipping {row['sample_id']}, transcript already exists.")
            continue

        language = row['language']
        audio_filename = row['audio_file']
        
        # Process 'hi' (Hindi), 'kn' (Kannada), and 'en' (English)
        if language not in ['hi', 'kn', 'en']:
            continue

        audio_path = get_audio_file_path(audio_base_dir, language, audio_filename)
        
        if not audio_path or not os.path.exists(audio_path):
            logger.warning(f"Audio file not found: {audio_path}")
            continue

        logger.info(f"Transcribing {audio_filename} ({language})...")
        transcript = transcribe_audio(audio_path, api_key)

        if transcript is not None:
            df.at[index, 'stt_transcript'] = transcript
            # Save progress after each successful transcription to avoid data loss
            try:
                df.to_csv(metadata_path, index=False)
                logger.info(f"Successfully transcribed and saved {audio_filename}.")
            except Exception as e:
                logger.error(f"Failed to save metadata after transcribing {audio_filename}: {e}")
        else:
            logger.warning(f"Failed to transcribe {audio_filename}.")

if __name__ == "__main__":
    main()
