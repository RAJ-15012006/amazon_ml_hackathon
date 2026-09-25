"""
Text preprocessing and tokenization module for Entity Resolution.
"""
import re
from unidecode import unidecode
try:
    from .config import LEGAL_SUFFIXES, COMMON_ADDR_WORDS
except (ImportError, ValueError):
    from src.config import LEGAL_SUFFIXES, COMMON_ADDR_WORDS

URL_PATTERN = re.compile(
    r"(https?://|www\.|\.com|\.org|\.net|\.c0m|\.co\.in|\.co|\.in|\.fr|\.io|@)",
    re.IGNORECASE
)
NON_ALPHANUM = re.compile(r"[^a-z0-9\s]")
ONLY_DIGITS = re.compile(r"\b\d+\b")


def clean_text(text: str) -> str:
    """Normalize unicode, strip accents, and lowercase."""
    if not text:
        return ""
    text = unidecode(str(text)).lower()
    return text


def parse_business_name(name: str):
    """
    Clean business name:
    - Strip URLs, domains, @ symbols
    - Remove legal suffixes (LLC, Inc, Pvt Ltd, SARL, etc.)
    - Return (cleaned_string, token_set, alphanumeric_slug)
    """
    if not name:
        return "", set(), ""
    cleaned = clean_text(name)
    cleaned = URL_PATTERN.sub(" ", cleaned)
    cleaned = NON_ALPHANUM.sub(" ", cleaned)
    
    tokens = [
        w for w in cleaned.split()
        if len(w) >= 2 and w not in LEGAL_SUFFIXES
    ]
    slug = "".join(tokens)
    return " ".join(tokens), set(tokens), slug


def parse_business_address(address: str):
    """
    Clean business address:
    - Extract numeric sequences (house numbers, PIN/zip codes, plot numbers)
    - Remove common address stopwords
    - Return (cleaned_string, token_set, number_set)
    """
    if not address:
        return "", set(), set()
    cleaned = clean_text(address)
    nums = set(ONLY_DIGITS.findall(cleaned))
    
    cleaned_words = NON_ALPHANUM.sub(" ", cleaned)
    tokens = {
        w for w in cleaned_words.split()
        if len(w) >= 3 and w not in COMMON_ADDR_WORDS
    }
    return cleaned, tokens, nums


def parse_entity_record(name: str, address: str):
    """
    Unified record parser returning processed representations for blocking and features.
    """
    clean_n, n_tokens, n_slug = parse_business_name(name)
    clean_a, a_tokens, nums = parse_business_address(address)
    combined_tokens = n_tokens | a_tokens
    return {
        "raw_name": clean_n,
        "name_tokens": n_tokens,
        "name_slug": n_slug,
        "raw_address": clean_a,
        "address_tokens": a_tokens,
        "nums": nums,
        "combined_tokens": combined_tokens,
        "has_address": bool(clean_a)
    }
