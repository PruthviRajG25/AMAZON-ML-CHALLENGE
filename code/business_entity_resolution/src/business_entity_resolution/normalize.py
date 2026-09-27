import re
import unicodedata
from typing import List, Set, Optional

LEGAL_SUFFIXES: Set[str] = {
    "inc", "incorporated", "corp", "corporation", "llc", "llp", "ltd", "limited",
    "pvt", "private", "co", "company", "enterprises", "enterprise", "services",
    "service", "holdings", "holding", "group", "solutions", "industries", "industry",
    "technologies", "technology", "consultants", "consulting", "builders", "builder",
    "marketing", "associates", "properties", "international", "retail", "bank"
}

DOMAIN_REGEX = re.compile(r"\.(com|org|net|in|co|us|fr|io|info|biz|gov|edu)(\.[a-z]{2})?$", re.IGNORECASE)
NUMBER_REGEX = re.compile(r"\b\d+([a-zA-Z]|\(\d+\))?\b")
ADDR_STOPWORDS: Set[str] = {
    "rd", "road", "st", "street", "ave", "avenue", "dr", "drive", "lane", "ln",
    "blvd", "boulevard", "near", "opp", "opposite", "unit", "apt", "apartment",
    "flat", "floor", "block", "town", "city", "po", "post", "office", "box",
    "hno", "plot", "survey"
}

def normalize_text(text: Optional[str]) -> str:
    """Standardize unicode representation and lowercase."""
    if not text or not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text).lower()
    return text

def clean_name(text: Optional[str]) -> str:
    """Normalize business name by removing web extensions, legal suffixes, and punctuation."""
    norm = normalize_text(text)
    if not norm:
        return ""
    # Strip domain endings like .com
    norm = DOMAIN_REGEX.sub("", norm)
    # Replace non-alphanumeric with spaces, preserving letters and digits
    norm = re.sub(r"[^\w\s]", " ", norm)
    words = [w for w in norm.split() if w not in LEGAL_SUFFIXES]
    return " ".join(words)

def canonical_name_key(text: Optional[str]) -> str:
    """Extract alphanumeric-only representation without spaces or legal suffixes."""
    cleaned = clean_name(text)
    return re.sub(r"[^\w]", "", cleaned)

def token_signature_key(text: Optional[str], min_len: int = 3) -> str:
    """Alphabetically sorted significant tokens joined together."""
    cleaned = clean_name(text)
    tokens = sorted({w for w in cleaned.split() if len(w) >= min_len})
    return "_".join(tokens)

def normalize_address(text: Optional[str]) -> str:
    """Normalize address string with number zero-stripping and punctuation cleaning."""
    norm = normalize_text(text)
    if not norm:
        return ""
    # Standardize abbreviations
    norm = re.sub(r"\b(st|str)\b", "street", norm)
    norm = re.sub(r"\b(rd)\b", "road", norm)
    norm = re.sub(r"\b(ave|av)\b", "avenue", norm)
    norm = re.sub(r"\b(dr)\b", "drive", norm)
    norm = re.sub(r"\b(ln)\b", "lane", norm)
    # Replace punctuation with spaces
    norm = re.sub(r"[^\w\s]", " ", norm)
    # Normalize leading zeros in numbers (e.g. '0337' -> '337')
    tokens = [re.sub(r"^0+(\d+)", r"\1", w) for w in norm.split()]
    return " ".join(tokens)

def extract_address_keys(text: Optional[str]) -> List[str]:
    """Extract candidate blocking keys from address (street number + street keyword)."""
    norm = normalize_address(text)
    if not norm:
        return []
    tokens = norm.split()
    numbers = [w for w in tokens if any(c.isdigit() for c in w)]
    words = [w for w in tokens if not any(c.isdigit() for c in w) and w not in ADDR_STOPWORDS and len(w) >= 3]
    
    keys = []
    if numbers and words:
        # e.g., '337_oakland', '85_wayne'
        for num in numbers[:2]:
            for word in words[:2]:
                keys.append(f"{num}_{word}")
    elif words:
        for word in words[:1]:
            keys.append(f"word_{word}")
    return keys
