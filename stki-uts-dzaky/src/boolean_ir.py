from collections import defaultdict
from typing import Dict, List, Set

def build_inverted_index(corpus: Dict[str, List[str]]) -> Dict[str, Set[str]]:
    sindex: Dict[str, Set[str]] = defaultdict(set)
    for doc_id, tokens in corpus.items():
        for t in set(tokens):
            index[t].add(doc_id)
    return index


_PRECEDENCE = {"NOT": 3, "AND": 2, "OR": 1}

def _tokenize(q: str) -> List[str]:
    q = q.lower().replace("(", " ( ").replace(")", " ) ")
    return q.split()

def _to_postfix(tokens: List[str]) -> List[str]:
    out: List[str] = []
    ops: List[str] = []
    for tok in tokens:
        up = tok.upper()
        if tok == "(":
            ops.append("(")
        elif tok == ")":
            while ops and ops[-1] != "(":
                out.append(ops.pop())
            if ops and ops[-1] == "(":
                ops.pop()
        elif up in ("AND", "OR", "NOT"):
            while ops and ops[-1] != "(" and (
                (_PRECEDENCE[ops[-1]] > _PRECEDENCE[up]) or
                (_PRECEDENCE[ops[-1]] == _PRECEDENCE[up] and up != "NOT")
            ):
                out.append(ops.pop())
            ops.append(up)
        else:
            out.append(tok)  # term
    while ops:
        out.append(ops.pop())
    return out

def eval_boolean(index: Dict[str, Set[str]], universe: Set[str], query: str) -> List[str]:
    """
    Mengembalikan daftar doc_id yang cocok dengan query Boolean.
    Mendukung AND/OR/NOT dan tanda kurung.
    """
    tokens = _tokenize(query)
    postfix = _to_postfix(tokens)

    stack: List[Set[str]] = []
    for tok in postfix:
        up = tok.upper()
        if up == "AND":
            b, a = stack.pop(), stack.pop()
            stack.append(a & b)
        elif up == "OR":
            b, a = stack.pop(), stack.pop()
            stack.append(a | b)
        elif up == "NOT":
            a = stack.pop()
            stack.append(universe - a)
        else:
            stack.append(index.get(tok, set()))

    return sorted(stack[-1]) if stack else []
