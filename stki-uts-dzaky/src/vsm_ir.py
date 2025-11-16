import math
from collections import Counter, defaultdict
from typing import Dict, List

def tf(tokens: List[str]) -> Dict[str, float]:
    c = Counter(tokens)
    m = max(c.values()) or 1
    return {t: v/m for t, v in c.items()}  # tf normalisasi

def idf(all_docs: Dict[str, List[str]]) -> Dict[str, float]:
    N = len(all_docs)
    df = defaultdict(int)
    for toks in all_docs.values():
        for t in set(toks): df[t] += 1
    return {t: math.log((N+1)/(df_t+1)) + 1 for t, df_t in df.items()}

def tfidf_matrix(all_docs: Dict[str, List[str]]):
    _idf = idf(all_docs)
    mat = {}
    for doc, toks in all_docs.items():
        tfv = tf(toks)
        mat[doc] = {t: tfv[t]*_idf[t] for t in tfv}
    return mat, _idf

def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = set(a) | set(b)
    dot = sum(a.get(k,0)*b.get(k,0) for k in keys)
    na = math.sqrt(sum(v*v for v in a.values()))
    nb = math.sqrt(sum(v*v for v in b.values()))
    return 0.0 if na*nb == 0 else dot/(na*nb)

def rank(query_tokens, tfidf_docs, idf_dict, topk=5):
    # vector query (tf*idf)
    from collections import Counter
    qtf = Counter(query_tokens)
    mq = max(qtf.values()) or 1
    qvec = {t:(qtf[t]/mq)*idf_dict.get(t, 0.0) for t in qtf}
    scores = {doc: cosine(vec, qvec) for doc, vec in tfidf_docs.items()}
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]
