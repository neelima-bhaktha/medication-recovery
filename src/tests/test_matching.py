"""
Self-check for the matching path. Run directly: `python src/tests/test_matching.py`

Each assertion pins a bug that actually cost accuracy on the 60-sample eval set, so a
regression here shows up as a metric drop. Uses a tiny inline catalog, not the 249k-row
CSV, so it runs in under a second.
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from baseline.string_match import StringMatcher
from evaluations.evaluate import Evaluator
from utils.preprocessing import clean_transcript, phonetic_key

REAL_ENTRIES = [
    "Fenexo 120 Tablet",
    "Aneudox M Injection",
    "Urigo 40mg Tablet",
    "Lancas D 10mg/30mg Capsule",
    "MDG Tablet",
    "Asthamon-L Tablet",
    "Relicef CV 200mg/125mg Tablet",
]

# The generic-token filter is frequency-based, so the catalog has to be big enough for
# "common" to mean something: in a 7-row catalog every token looks common. Padding with
# distinct brands keeps "tablet" above the 2% generic threshold while each brand stays
# well below it, which is the ratio the real 249k-row catalog has.
CATALOG = REAL_ENTRIES + [f"Zylpha{i:03d} Tablet" for i in range(200)]


def test_romanizes_indic_script():
    # Devanagari must reach the Latin catalog at all; without this it scores 0 forever.
    assert "phenokso" in clean_transcript("मैंने फेनोक्सो 120 टैबलेट ले रही हूं")


def test_collapses_spaced_dosage():
    # Transcripts say "40 mg", the catalog writes "40mg".
    assert "40mg" in clean_transcript("I am taking Urigo 40 mg tablet")


def test_phonetic_key_bridges_romanization():
    # The whole reason Hindi/Kannada can match at all.
    assert phonetic_key("phenokso 120 taibaleta") == phonetic_key("fenexo 120 tablet")


def test_digits_survive_phonetic_key():
    # Metaphone drops digits; dosage is one of the few reliably transcribed parts.
    assert "120" in phonetic_key("fenexo 120 tablet")


def test_filler_does_not_outrank_real_name():
    # The original bug: fuzz.WRatio scored the filler token "ham" ~90 against
    # "astHAMon-l tablet", burying the true answer under generic catalog entries.
    matcher = StringMatcher(CATALOG)
    top = matcher.find_best_match("haan maine phenokso 120 taibaleta liya hai ham", k=3)
    assert top, "expected at least one match"
    assert top[0]["medicine"] == "Fenexo 120 Tablet", top[0]


def test_generic_phrase_is_not_a_candidate():
    # Form/dosage words carry no brand information; "mg tablet" was matching thousands
    # of entries at 95 and winning. The generic set is derived from catalog frequency,
    # so it adapts to whatever this catalog considers common ("tablet" here).
    matcher = StringMatcher(CATALOG)
    assert "tablet" in matcher._generic_tokens
    assert not matcher._is_informative("tablet")
    assert matcher._is_informative("fenexo 120")


def test_full_length_name_is_reachable():
    # Cleaned catalog names run to 6 tokens; a 4-word n-gram cap made them unformable.
    matcher = StringMatcher(CATALOG)
    top = matcher.find_best_match("she takes relicef cv 200mg 125mg tablet daily", k=1)
    assert top[0]["medicine"] == "Relicef CV 200mg/125mg Tablet", top[0]


def test_combination_dosage_is_not_split_as_two_answers():
    # "10mg/30mg" is one dosage. Splitting on a bare "/" made 9 of 60 samples
    # impossible to score correctly, regardless of what the matcher predicted.
    assert Evaluator.is_match("Lancas D 10mg/30mg Capsule", "Lancas D 10mg/30mg Capsule")
    # A space-padded slash really does separate two acceptable answers.
    assert Evaluator.is_match("Cmerx Eye Drop / Exam 50mg Tablet ER", "Cmerx Eye Drop")
    assert not Evaluator.is_match("Lancas D 10mg/30mg Capsule", "Lancas D 10mg")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} passed")
