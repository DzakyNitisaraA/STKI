from typing import Set, List

def precision(retrieved: List[str], relevant: Set[str]) -> float:
    return 0.0 if not retrieved else len([d for d in retrieved if d in relevant]) / len(retrieved)

def recall(retrieved: List[str], relevant: Set[str]) -> float:
    return 0.0 if not relevant else len([d for d in retrieved if d in relevant]) / len(relevant)

def f1(p: float, r: float) -> float:
    return 0.0 if (p+r)==0 else 2*p*r/(p+r)
