import os
from typing import List, Dict, Optional

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL_NAME = "llama-3.3-70b-versatile"

_client: Optional[Groq] = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


def rerank(transcript: str, candidates: List[Dict], language: str = "") -> Optional[str]:
    """
    Asks the LLM which candidate medicine (if any) is the one referenced in the transcript.
    Candidates come from dense retrieval; the transcript may be in a different script or
    contain STT phonetic errors, which an LLM is better positioned to reason through than
    embedding similarity alone. Returns the matched medicine name verbatim, or None.
    """
    if not candidates:
        return None

    numbered = "\n".join(
        f"{i + 1}. {c['medicine']} ({c.get('salt_composition', '')})"
        for i, c in enumerate(candidates)
    )
    prompt = (
        f"Call transcript (language: {language or 'unknown'}):\n\"{transcript}\"\n\n"
        f"Candidate medicines (from retrieval, may include wrong matches):\n{numbered}\n\n"
        "Which candidate number is the medicine actually mentioned in the transcript? "
        "The transcript may be in a different script than the candidate names, or contain "
        "speech-to-text phonetic errors. Reply with ONLY the number, or 0 if none match."
    )

    response = _get_client().chat.completions.create(
        model=MODEL_NAME,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    reply = response.choices[0].message.content.strip()

    digits = "".join(ch for ch in reply if ch.isdigit())
    if not digits:
        return None

    choice = int(digits)
    if choice < 1 or choice > len(candidates):
        return None

    return candidates[choice - 1]["medicine"]
