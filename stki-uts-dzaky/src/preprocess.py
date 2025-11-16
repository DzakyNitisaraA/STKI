import re
from typing import List

_ID = re.compile(r"[^a-z0-9\s]")

def clean(text: str) -> str:
    text = text.lower()
    text = _ID.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str) -> List[str]:
    return text.split()

_STOP = {"yang","dan","di","ke","dari","untuk","pada","ini","itu","dengan","atau"}

def remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in _STOP]

def stem(tokens: List[str]) -> List[str]:
    return [re.sub(r"(lah|kah|pun|nya|kan|i|an)$", "", t) for t in tokens]

def preprocess(text: str) -> List[str]:
    return stem(remove_stopwords(tokenize(clean(text))))

if __name__ == "__main__":
    sample_text = "Informasi lengkap tentang Universitas Dian Nuswantoro, kampus unggulan dengan fasilitas terbaik dan program studi berkualitas"
    print(preprocess(sample_text))
