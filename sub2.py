import json
import math
import os
import re
import zipfile
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


NGRAM_N = 2


def tokenize(text: str) -> list:
    if not text:
        return []
    return re.findall(r"\w+", text.lower())


def add_ngrams(tokens: list, n: int = NGRAM_N) -> list:
    if n <= 1 or len(tokens) < n:
        return []
    return ["_".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def tokenize_with_ngrams(text: str) -> list:
    tokens = tokenize(text)
    return tokens + add_ngrams(tokens, NGRAM_N)


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def bm25_score(
    query_tokens: list,
    doc_counts: Counter,
    doc_len: int,
    avg_doc_len: float,
    total_docs: int,
    doc_freq: dict,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    score = 0.0
    if not query_tokens or avg_doc_len <= 0:
        return score

    norm = k1 * (1 - b + b * (doc_len / avg_doc_len))
    for token in query_tokens:
        df = doc_freq.get(token, 0)
        if df == 0:
            continue
        idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
        tf = doc_counts.get(token, 0)
        score += idf * (tf * (k1 + 1)) / (tf + norm)
    return score


class BM25Retriever:
    def __init__(self, doc_texts: list, doc_token_lists: list):
        self.doc_texts = doc_texts
        self.doc_token_lists = doc_token_lists
        self.doc_token_sets = [set(tokens) for tokens in doc_token_lists]
        self.doc_token_counts = [Counter(tokens) for tokens in doc_token_lists]
        self.doc_len = [len(tokens) for tokens in doc_token_lists]
        self.total_docs = len(doc_token_lists)
        self.avg_doc_len = sum(self.doc_len) / self.total_docs if self.total_docs else 1.0
        self.doc_freq = defaultdict(int)
        for tokens in self.doc_token_sets:
            for token in tokens:
                self.doc_freq[token] += 1

    def search(self, query_tokens: list, k: int = 5) -> list:
        scores = []
        for i in range(self.total_docs):
            score = bm25_score(
                query_tokens,
                self.doc_token_counts[i],
                self.doc_len[i],
                self.avg_doc_len,
                self.total_docs,
                self.doc_freq,
            )
            if score > 0:
                scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [
            {
                "id": idx,
                "text": self.doc_texts[idx],
                "tokens": self.doc_token_lists[idx],
                "token_set": self.doc_token_sets[idx],
                "score": score,
            }
            for idx, score in scores[:k]
        ]


def load_corpus(corpus_file: str = "dataset.json"):
    documents = []
    doc_token_lists = []
    doc_token_sets = []
    try:
        with open(corpus_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for doc in data:
            text = f"{doc.get('title', '')} {doc.get('content', '')}".strip()
            documents.append(text)
            tokens = tokenize_with_ngrams(text)
            doc_token_lists.append(tokens)
            doc_token_sets.append(set(tokens))
    except (FileNotFoundError, json.JSONDecodeError):
        return [], [], []
    return documents, doc_token_lists, doc_token_sets


def select_answers(item: dict, candidate_docs: list, valid_choices: list, threshold_ratio: float = 0.8) -> list:
    scores = {}
    max_score = 0.0
    question_text = item.get("question", "")
    question_tokens = set(tokenize_with_ngrams(question_text))

    for key in valid_choices:
        choice_text = item.get(key, "")
        choice_tokens = set(tokenize_with_ngrams(choice_text))
        choice_lower = choice_text.lower()
        best_doc_score = 0.0

        for doc in candidate_docs:
            doc_tokens = doc["token_set"]
            doc_text = doc["text"].lower()

            overlap = jaccard(choice_tokens, doc_tokens)
            question_overlap = jaccard(choice_tokens, question_tokens) if question_tokens else 0.0
            exact_bonus = 1.0 if choice_lower and choice_lower in doc_text else 0.0
            fuzzy_score = SequenceMatcher(None, choice_lower, doc_text).ratio() if choice_lower and doc_text else 0.0

            score = (
                0.50 * overlap
                + 0.15 * question_overlap
                + 0.20 * exact_bonus
                + 0.15 * fuzzy_score
            )
            if score > best_doc_score:
                best_doc_score = score

        if best_doc_score == 0.0 and choice_text:
            fallback_text = question_text.lower()
            if fallback_text:
                best_doc_score = 0.10 * SequenceMatcher(None, choice_lower, fallback_text).ratio()

        scores[key] = best_doc_score
        if best_doc_score > max_score:
            max_score = best_doc_score

    threshold = threshold_ratio * max_score if max_score > 0 else 0.0
    selected = [key for key in valid_choices if scores[key] >= threshold and scores[key] > 0]

    if not selected:
        selected = [max(scores, key=lambda k: scores[k])]

    return sorted(selected)


def make_submission(
    test_file: str = "de_thi.json",
    corpus_file: str = "dataset.json",
    output_file: str = "submission.json",
    zip_file: str = "submission.zip",
    threshold_ratio: float = 0.8,
    top_k_docs: int = 5,
):
    documents, doc_token_lists, _ = load_corpus(corpus_file)
    if not documents:
        print(f"Loi: khong tai duoc corpus tu {corpus_file}.")
        return

    print(f"Da tai {len(documents)} tai lieu tu tap corpus...")

    try:
        with open(test_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)
    except FileNotFoundError:
        print(f"Loai: khong tim thay file {test_file}.")
        return

    retriever = BM25Retriever(documents, doc_token_lists)
    submissions = []
    valid_choices = ["A", "B", "C", "D"]

    print("Bat dau truy xuat (BM25 + fuzzy) va du doan dap an...")

    for item in tqdm(test_data, desc="Processing..."):
        question_id = item.get("id")
        question_text = item.get("question", "")

        query_tokens = tokenize_with_ngrams(question_text)
        candidate_docs = retriever.search(query_tokens, k=top_k_docs)
        answers = select_answers(item, candidate_docs, valid_choices, threshold_ratio)

        submissions.append({
            "id": question_id,
            "answer": answers,
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(submissions, f, ensure_ascii=False, indent=2)

    current_script = Path(__file__).name if "__file__" in globals() else "sub.py"
    with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_file, arcname=os.path.basename(output_file))
        if os.path.exists(current_script):
            zipf.write(current_script, arcname=os.path.basename(current_script))

    print(f"Da xu ly xong {len(submissions)} cau hoi.")
    print(f"File nop bai ZIP da san sang tai: {zip_file}")


if __name__ == "__main__":
    make_submission(
        test_file="de_thi.json",
        corpus_file="dataset.json",
        output_file="submission.json",
        zip_file="submission.zip",
        threshold_ratio=0.8,
        top_k_docs=20,
    )
