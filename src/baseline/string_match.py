from typing import Union, List, Tuple, Callable
from collections import Counter
import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz  # type: ignore

from utils.preprocessing import clean_medicine_name, clean_transcript, phonetic_key  # noqa: F401

# Candidate phrases shorter than this are filler ("ham", "hai", "i am") that cannot
# plausibly be a medicine name, but still score highly under substring-based scorers.
MIN_CANDIDATE_CHARS = 4

# Scores below this are noise; dropping them keeps the score matrix sparse-ish and cheap.
SCORE_CUTOFF = 55

# Longest candidate phrase, in words. Catalog names run to 6 tokens once cleaned
# ("chemiclo sp 100mg 325mg 10mg tablet"), so a shorter cap makes the full name
# impossible to form and only ever scores a fragment of it.
MAX_NGRAM_WORDS = 7

# Literal and phonetic similarity are combined differently depending on the source script,
# because the two inputs carry different noise:
#
#   English STT is already in the catalog's alphabet, so its spelling is trustworthy.
#   Phonetics only need to rescue the cases it gets wrong -> take the better of the two,
#   discounting phonetics so it cannot override a clean spelling match.
#
#   Romanized Hindi/Kannada has no reliable spelling at all (Devanagari "फेनोक्सो" becomes
#   "phenokso" against a catalog spelling of "fenexo"), so the literal view is weak
#   evidence on its own. Blending makes agreement across both views the deciding factor.
PHONETIC_WEIGHT = 0.80
ROMANIZED_LITERAL_WEIGHT = 0.35

# A token appearing in at least this fraction of catalog names is a dosage/form word
# ("tablet" is in 60% of them), not brand information. Derived from the catalog rather
# than hand-listed so it stays correct if the catalog changes.
GENERIC_TOKEN_MIN_FREQ = 0.02

# A token appearing in at least this fraction of transcripts is conversational filler.
# Kept high enough that a medicine mentioned across a handful of calls is never dropped.
FILLER_TOKEN_MIN_FREQ = 0.25


class StringMatcher:
    """
    A baseline model for finding the closest medicine name from a dataset
    using fuzzy string matching.
    """

    def __init__(
        self,
        medicines: Union[pd.DataFrame, List[str]],
        scorer: Callable = fuzz.ratio,
        corpus: Union[List[str], None] = None,
    ) -> None:
        """
        Initializes the StringMatcher with a dataset of medicines.

        Args:
            medicines (Union[pd.DataFrame, List[str]]): The medicine dataset.
                If a DataFrame is provided, it is assumed to have a column 
                representing the medicine names (defaults to the first string column if unnamed, 
                or explicitly 'medicine_name'). Here we simply take the first column for robustness, 
                or you can pass a list of strings directly.
            scorer (Callable, optional): The rapidfuzz scoring function to use.
                Defaults to fuzz.ratio. NOT WRatio: WRatio switches to partial_ratio
                when the two strings differ a lot in length, so a 3-letter filler token
                ("ham") scores ~90 against every catalog entry containing those letters
                ("astHAMon-l tablet") and drowns out the real medicine name.
        """
        self.scorer = scorer
        
        # Extract medicine names into a list
        if isinstance(medicines, pd.DataFrame):
            # Attempt to find a 'medicine_name' column, then 'name', otherwise fallback to the first column
            if 'medicine_name' in medicines.columns:
                raw_medicines = medicines['medicine_name'].dropna().astype(str).tolist()
            elif 'name' in medicines.columns:
                raw_medicines = medicines['name'].dropna().astype(str).tolist()
            else:
                raw_medicines = medicines.iloc[:, 0].dropna().astype(str).tolist()
        else:
            raw_medicines = [str(m) for m in medicines if m]
            
        # Store a mapping of cleaned name -> original raw name
        # We match against the cleaned name, but return the original name to the user.
        self._cleaned_to_raw = {}
        for raw_name in raw_medicines:
            cleaned = clean_medicine_name(raw_name)
            if cleaned:
                # If multiple raw names map to the same cleaned name, the last one overwrites.
                self._cleaned_to_raw[cleaned] = raw_name
                
        self._cleaned_medicines = list(self._cleaned_to_raw.keys())
        self._phonetic_medicines = [phonetic_key(n) for n in self._cleaned_medicines]

        # Tokens common enough across the catalog to carry no brand information.
        document_freq = Counter()
        for cleaned in self._cleaned_medicines:
            document_freq.update(set(cleaned.split()))
        min_count = GENERIC_TOKEN_MIN_FREQ * max(1, len(self._cleaned_medicines))
        self._generic_tokens = {
            token for token, count in document_freq.items() if count >= min_count
        }

        # Conversational filler ("this", "what", "taking", and their romanized Hindi /
        # Kannada equivalents) is absent from the catalog, so catalog frequency alone
        # rates it as brand-like. Transcript frequency catches it instead: a brand name
        # appears in one or two calls, filler appears in most of them.
        if corpus:
            cleaned_corpus = [clean_transcript(text) for text in corpus if text]
            corpus_freq = Counter()
            for text in cleaned_corpus:
                corpus_freq.update(set(text.split()))
            filler_min = FILLER_TOKEN_MIN_FREQ * max(1, len(cleaned_corpus))
            self._generic_tokens |= {
                token for token, count in corpus_freq.items() if count >= filler_min
            }

    def _is_informative(self, phrase: str) -> bool:
        """
        A candidate phrase is only worth scoring if at least one of its tokens could be
        brand information. Phrases made entirely of dosage/form words ("mg tablet") match
        thousands of catalog entries equally well and crowd out the real answer.
        """
        return any(token not in self._generic_tokens for token in phrase.split())

    def find_best_match(self, transcript: str, k: int = 5) -> List[dict]:
        """
        Finds the top-K matching medicine names for a given raw transcript using 
        candidate phrase matching (n-grams).

        Args:
            transcript (str): The input transcript containing a spoken medication.
            k (int): The number of top matches to return. Default is 5.

        Returns:
            List[dict]: A ranked list of dictionaries, each containing:
                - medicine (str): The best matched original medicine name
                - score (float): The similarity score (0.0 to 100.0)
                - candidate_phrase (str): The candidate phrase that matched best
                - phrase_length (int): The number of words in the candidate phrase
        """
        if not transcript or not isinstance(transcript, str):
            return []

        # Checked before cleaning, since cleaning is what romanizes the text.
        is_romanized = not transcript.isascii()

        cleaned_query = clean_transcript(transcript)
        
        # Gracefully handle cases where preprocessing strips the string entirely
        if not cleaned_query:
            return []

        words = cleaned_query.split()
        if not words:
            return []

        # Generate unique candidate phrases (n-grams from 1 to 4 words), dropping
        # phrases too short to be a medicine name (see MIN_CANDIDATE_CHARS).
        candidate_phrases = set()
        max_n = min(MAX_NGRAM_WORDS, len(words))

        for n in range(1, max_n + 1):
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i:i+n])
                if len(phrase) >= MIN_CANDIDATE_CHARS and self._is_informative(phrase):
                    candidate_phrases.add(phrase)

        if not candidate_phrases:
            return []

        candidates = sorted(candidate_phrases)
        phrase_lengths = np.array([len(c.split()) for c in candidates])

        # Score every candidate against every medicine with one vectorised, multithreaded
        # pass per chunk. Chunking caps peak memory at CHUNK x len(catalog) bytes.
        n_medicines = len(self._cleaned_medicines)
        best_scores = np.zeros(n_medicines, dtype=np.uint8)
        best_candidate_idx = np.zeros(n_medicines, dtype=np.int32)

        CHUNK = 64
        for start in range(0, len(candidates), CHUNK):
            chunk = candidates[start:start + CHUNK]
            scores = process.cdist(
                chunk,
                self._cleaned_medicines,
                scorer=self.scorer,
                dtype=np.uint8,
                workers=-1,
            )

            # Second pass on Metaphone keys, so romanized speech can still reach the
            # catalog spelling ("phenokso 120 taibaleta" and "fenexo 120 tablet" share
            # the key "FNKS 120 TBLT"). No score_cutoff on either pass: zeroing a view
            # would corrupt the blend rather than just discarding a weak match.
            phonetic_scores = process.cdist(
                [phonetic_key(c) for c in chunk],
                self._phonetic_medicines,
                scorer=self.scorer,
                dtype=np.uint8,
                workers=-1,
            )

            if is_romanized:
                combined = (
                    ROMANIZED_LITERAL_WEIGHT * scores
                    + (1.0 - ROMANIZED_LITERAL_WEIGHT) * phonetic_scores
                )
            else:
                combined = np.maximum(scores, PHONETIC_WEIGHT * phonetic_scores)

            scores = combined.astype(np.uint8)
            scores[scores < SCORE_CUTOFF] = 0

            chunk_best = scores.max(axis=0)
            improved = chunk_best > best_scores
            best_scores[improved] = chunk_best[improved]
            best_candidate_idx[improved] = scores.argmax(axis=0)[improved] + start

        matched = np.nonzero(best_scores)[0]
        if matched.size == 0:
            return []

        # Rank by score desc, then by phrase length desc (lexsort's last key is primary).
        matched_lengths = phrase_lengths[best_candidate_idx[matched]]
        order = np.lexsort((-matched_lengths, -best_scores[matched].astype(np.int16)))

        results_list = []
        for idx in matched[order[:k]]:
            candidate = candidates[best_candidate_idx[idx]]
            results_list.append({
                "medicine": self._cleaned_to_raw[self._cleaned_medicines[idx]],
                "score": float(best_scores[idx]),
                "candidate_phrase": candidate,
                "phrase_length": int(phrase_lengths[best_candidate_idx[idx]]),
            })

        return results_list
