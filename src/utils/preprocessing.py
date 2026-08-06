import re
import unicodedata

import jellyfish

from indic_transliteration import sanscript
from indic_transliteration.detect import detect
from indic_transliteration.sanscript import transliterate


def _romanize(text: str) -> str:
    """
    Phonetically romanizes non-Latin script (Devanagari/Kannada/etc.) to ASCII so it
    can be fuzzy-matched against the Latin-only medicine catalog, e.g.
    "फेनोक्सो" -> "phenokso" (matchable against catalog name "Fenexo").
    Only text with non-ASCII characters is passed to script detection, since
    `detect` misclassifies plain ASCII text as the SLP1 scheme.
    """
    if text.isascii():
        return text

    scheme = detect(text)
    if scheme == sanscript.ITRANS or scheme not in sanscript.SCHEMES:
        return text

    return transliterate(text, scheme, sanscript.ITRANS)


def _strip_punctuation(text: str) -> str:
    """
    Replaces punctuation with spaces while keeping letters, digits, and combining
    marks intact. Devanagari/Kannada (and other Indic scripts) spell syllables with
    combining vowel signs (Unicode category Mn/Mc) that `\\w` does NOT match, so a
    plain `[^\\w\\s]` regex strips them as if they were punctuation and shreds words
    down to bare consonants (e.g. "टैबलेट" -> "ट बल ट"). Category-based filtering
    keeps those marks attached to their base character instead.
    """
    return "".join(
        ch if ch.isalnum() or ch.isspace() or unicodedata.category(ch)[0] == "M" else " "
        for ch in text
    )


def normalize_text(text: str) -> str:
    """
    Normalizes text by converting it to lowercase, stripping leading and 
    trailing whitespace, and reducing multiple spaces to a single space.

    Args:
        text (str): The input text to normalize.

    Returns:
        str: The normalized string, or an empty string if the input is None or empty.
    """
    if not text:
        return ""
    
    # Convert to string (in case of unexpected types), lowercase, and strip
    text = str(text).lower().strip()
    
    # Replace multiple whitespace characters with a single space
    return re.sub(r'\s+', ' ', text)


def clean_medicine_name(name: str) -> str:
    """
    Cleans a medicine name by normalizing it and removing unnecessary 
    punctuation, while preserving meaningful alphanumeric characters 
    (like numbers and dosage units, e.g., '500mg').

    Args:
        name (str): The raw medicine name.

    Returns:
        str: The cleaned medicine name.
    """
    if not name:
        return ""
    
    # Convert to lowercase first
    name = str(name).lower()
    
    # Replace non-alphanumeric and non-whitespace characters with a space
    # This removes punctuation but keeps words separate (e.g., "Tylenol, 500mg" -> "tylenol  500mg")
    name = _strip_punctuation(name)
    
    # Delegate to normalize_text to handle stripping and multiple spaces
    return normalize_text(name)


def clean_transcript(transcript: str) -> str:
    """
    Cleans a transcript by normalizing it and stripping out extraneous 
    punctuation. This provides a clean alphanumeric baseline for 
    downstream processing.

    Args:
        transcript (str): The raw transcript text.

    Returns:
        str: The cleaned transcript.
    """
    if not transcript:
        return ""

    # Romanize non-Latin script (Devanagari/Kannada/...) before lowercasing, since
    # `detect`/`transliterate` need the original script characters to work.
    transcript = _romanize(str(transcript))

    # Convert to lowercase
    transcript = transcript.lower()

    # Replace punctuation with a space to prevent words from merging
    transcript = _strip_punctuation(transcript)
    
    transcript = normalize_text(transcript)

    # Catalog names write dosages unspaced ("500mg"), transcripts say them spaced
    # ("500 mg"). Collapse to the catalog convention so the two are comparable.
    return re.sub(r'\b(\d+(?:\.\d+)?)\s+(mg|ml|mcg|gm|iu)\b', r'\1\2', transcript)


def phonetic_key(text: str) -> str:
    """
    Reduces text to a Metaphone consonant skeleton so that phonetically-equivalent
    spellings collapse together. Romanized Indic speech and the Latin catalog spell the
    same drug very differently ("phenokso" vs "fenexo", "taibaleta" vs "tablet"), but
    both reduce to the same key (FNKS, TBLT).

    Tokens containing digits are kept verbatim: Metaphone discards digits entirely, and
    the dosage ("120", "500mg") is one of the few reliably-transcribed discriminators.
    """
    if not text:
        return ""

    keys = []
    for token in str(text).split():
        if any(ch.isdigit() for ch in token):
            keys.append(token)
        else:
            keys.append(jellyfish.metaphone(token) or token)

    return " ".join(k for k in keys if k)
