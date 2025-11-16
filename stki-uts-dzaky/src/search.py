import argparse, json, os
from src.preprocess import preprocess
from src.boolean_ir import build_inverted_index, eval_boolean
from src.vsm_ir import tfidf_matrix, rank

def load_corpus(data_dir="data"):
    docs = {}
    for fn in os.listdir(data_dir):
        if fn.endswith(".txt"):
            with open(os.path.join(data_dir, fn), "r", encoding="utf-8") as f:
                docs[fn] = preprocess(f.read())
    return docs

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["boolean","vsm"], default="vsm")
    p.add_argument("--query", required=True)
    args = p.parse_args()

    corpus = load_corpus()
    if args.model == "boolean":
        index = build_inverted_index(corpus)
        universe = set(corpus.keys())
        results = eval_boolean(index, universe, args.query)
    else:
        tfidf_docs, idf = tfidf_matrix(corpus)
        q_tokens = preprocess(args.query)
        results = [d for d,_ in rank(q_tokens, tfidf_docs, idf, topk=10)]

    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
