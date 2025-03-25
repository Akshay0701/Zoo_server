import spacy
import re

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Define a set of keywords that indicate non-research-field content
BANNED_WORDS = {
    "bachelor", "master", "phd", "gpa", "university", "college", "experience",
    "technical", "skills", "research", "assistant", "degree", "certificate",
    "project", "projects", "education", "contact", "email", "phone"
}

# Define known acronyms and their preferred casing
ACRONYMS = {"ai": "AI", "ml": "ML"}

def normalize_candidate(candidate):
    """Normalize a candidate phrase by lemmatizing and handling acronyms."""
    doc = nlp(candidate)
    lemmas = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
    normalized = " ".join(lemmas)
    # Use uppercase for known acronyms, otherwise title case
    if normalized.lower() in ACRONYMS:
        return ACRONYMS[normalized.lower()]
    else:
        return normalized.title()

def is_valid_field(candidate):
    """Check if a candidate phrase is a valid research topic."""
    candidate = candidate.strip()
    # Skip if empty or contains digits
    if not candidate or any(char.isdigit() for char in candidate):
        return False
    # Use spaCy NER to exclude specific entity types
    doc = nlp(candidate)
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG", "GPE", "DATE", "TIME"}:
            return False
    # Tokenize and check length and acronyms
    tokens = candidate.split()
    if len(tokens) < 2 and candidate.lower() not in {"ai", "ml"}:
        return False
    # Exclude banned words
    for token in tokens:
        if token.lower() in BANNED_WORDS:
            return False
    return True

def extract_research_interests(text):
    """Extract research interests from a text description."""
    interests = []
    # Look for a "Research Interests" section
    pattern = re.compile(
        r"Research Interests\s*[:\-]?\s*(.*?)(?=\n[A-Z][a-z]+:|\n\n|\Z)",
        re.IGNORECASE | re.DOTALL
    )
    match = pattern.search(text)
    if match:
        section_text = match.group(1).strip()
        candidates = re.split(r'[,\n;]+', section_text)
    else:
        # Fallback to noun chunks from the entire text
        doc = nlp(text)
        candidates = [chunk.text for chunk in doc.noun_chunks]
    
    # Normalize and filter candidates
    seen = set()
    for candidate in candidates:
        normalized = normalize_candidate(candidate)
        if is_valid_field(normalized):
            if normalized.lower() not in seen:
                seen.add(normalized.lower())
                interests.append(normalized)
        if len(interests) >= 10:
            break
    return interests