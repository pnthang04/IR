import json
import os
import re
import sys
import time
import zipfile
from collections import defaultdict

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


TOP_K = 5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CORPUS_FILE = os.path.join(SCRIPT_DIR, "dataset.json")
DEFAULT_TEST_FILE = os.path.join(SCRIPT_DIR, "de_thi.json")
DEFAULT_OUTPUT_FILE = os.path.join(SCRIPT_DIR, "submission.json")
DEFAULT_ZIP_FILE = os.path.join(SCRIPT_DIR, "submission.zip")


def progress(iterable, desc=None, total=None):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total)


def tokenize(text):
    return re.findall(r"\w+", str(text).lower())


class FastBM25Okapi:
    def __init__(self, tokenized_docs, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.n_docs = len(tokenized_docs)
        self.doc_len = np.array([len(doc) for doc in tokenized_docs], dtype=float)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0
        self.vocab = {}

        postings = defaultdict(list)
        doc_freq = defaultdict(int)
        for doc_id, doc in enumerate(
            progress(tokenized_docs, desc="Building postings", total=len(tokenized_docs))
        ):
            term_freq = defaultdict(int)
            for term in doc:
                term_freq[term] += 1

            for term, tf in term_freq.items():
                postings[term].append((doc_id, tf))
                doc_freq[term] += 1

        rows = []
        cols = []
        data = []
        for term_id, (term, term_postings) in enumerate(
            progress(postings.items(), desc="Building sparse matrix", total=len(postings))
        ):
            self.vocab[term] = term_id
            df = doc_freq[term]
            idf = np.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

            for doc_id, tf in term_postings:
                denom = tf + self.k1 * (
                    1 - self.b + self.b * self.doc_len[doc_id] / self.avgdl
                )
                weight = idf * (tf * (self.k1 + 1) / denom)
                rows.append(doc_id)
                cols.append(term_id)
                data.append(weight)

        self.term_doc_matrix = coo_matrix(
            (data, (rows, cols)),
            shape=(self.n_docs, len(self.vocab)),
            dtype=float,
        ).tocsr()

    def get_scores(self, query_terms):
        if self.n_docs == 0 or not query_terms:
            return np.zeros(self.n_docs, dtype=float)

        query_freq = defaultdict(int)
        for term in query_terms:
            term_id = self.vocab.get(term)
            if term_id is not None:
                query_freq[term_id] += 1

        if not query_freq:
            return np.zeros(self.n_docs, dtype=float)

        query_vector = csr_matrix(
            (
                list(query_freq.values()),
                ([0] * len(query_freq), list(query_freq.keys())),
            ),
            shape=(1, len(self.vocab)),
            dtype=float,
        )
        return (query_vector @ self.term_doc_matrix.T).toarray().ravel()


def load_corpus(corpus_file):
    documents = []
    try:
        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in progress(f, desc="Loading corpus"):
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue

                title = doc.get("title", "")
                content = doc.get("content", "")
                text = f"{title} {content}".strip()
                if text:
                    documents.append(text)
    except FileNotFoundError:
        print(f"[ERROR] Cannot find corpus file: {corpus_file}")
        sys.exit(1)

    print(f"[INFO] Loaded {len(documents):,} documents from corpus.")
    return documents


def load_test_data(test_file):
    try:
        with open(test_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Cannot find test file: {test_file}")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Invalid JSON test file: {exc}")
        sys.exit(1)

    print(f"[INFO] Loaded {len(test_data)} questions from test set.")
    return test_data


def build_bm25_index(documents):
    print("[INFO] Tokenizing corpus...")
    tokenized_docs = [
        tokenize(doc) for doc in progress(documents, desc="Tokenizing corpus", total=len(documents))
    ]

    print("[INFO] Building global BM25 index...")
    bm25 = FastBM25Okapi(tokenized_docs)
    print(f"[INFO] BM25 index ready. Documents: {len(tokenized_docs):,}")
    return bm25


def retrieve_top_k(query, bm25, documents, top_k=TOP_K):
    scores = bm25.get_scores(tokenize(query))

    k = min(top_k, len(scores))
    if k <= 0:
        return []

    top_k_indices = np.argpartition(scores, -k)[-k:]
    top_k_indices = top_k_indices[np.argsort(scores[top_k_indices])[::-1]]

    retrieved_docs = []
    for idx in top_k_indices:
        if scores[idx] > 0:
            retrieved_docs.append(documents[idx])

    return retrieved_docs


def build_local_bm25(tokenized_context, k1=1.5, b=0.75, epsilon=0.25):
    doc_lengths = [len(tokens) for tokens in tokenized_context]
    avg_doc_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0
    postings = defaultdict(list)
    doc_freq = defaultdict(int)

    for doc_idx, tokens in enumerate(tokenized_context):
        term_freq = defaultdict(int)
        for term in tokens:
            term_freq[term] += 1

        for term, tf in term_freq.items():
            postings[term].append((doc_idx, tf))
            doc_freq[term] += 1

    n_docs = len(tokenized_context)
    idf = {}
    idf_sum = 0.0
    negative_terms = []

    for term, df in doc_freq.items():
        term_idf = np.log(n_docs - df + 0.5) - np.log(df + 0.5)
        idf[term] = term_idf
        idf_sum += term_idf
        if term_idf < 0:
            negative_terms.append(term)

    average_idf = idf_sum / len(idf) if idf else 0.0
    eps = epsilon * average_idf
    for term in negative_terms:
        idf[term] = eps

    return {
        "k1": k1,
        "b": b,
        "postings": postings,
        "idf": idf,
        "doc_lengths": doc_lengths,
        "avg_doc_len": avg_doc_len,
    }


def local_bm25_total_score(local_bm25, query_terms):
    if not query_terms or not local_bm25["avg_doc_len"]:
        return 0.0

    total_score = 0.0
    k1 = local_bm25["k1"]
    b = local_bm25["b"]
    avg_doc_len = local_bm25["avg_doc_len"]
    doc_lengths = local_bm25["doc_lengths"]

    for term in query_terms:
        term_idf = local_bm25["idf"].get(term, 0.0)
        if term_idf == 0.0:
            continue

        for doc_idx, tf in local_bm25["postings"].get(term, []):
            doc_len = doc_lengths[doc_idx]
            denom = tf + k1 * (1 - b + b * doc_len / avg_doc_len)
            total_score += term_idf * (tf * (k1 + 1) / denom)

    return total_score


def select_answer(question_text, choices, context_text):
    valid_keys = ["A", "B", "C", "D"]

    if not context_text.strip():
        return "A"

    context_sentences = re.split(r"[.;!?\n]+", context_text)
    context_sentences = [sentence.strip() for sentence in context_sentences if sentence.strip()]
    if not context_sentences:
        return "A"

    tokenized_context = [tokenize(sentence) for sentence in context_sentences]
    tokenized_context = [tokens for tokens in tokenized_context if tokens]
    if not tokenized_context:
        return "A"

    local_bm25 = build_local_bm25(tokenized_context)

    best_choice = "A"
    max_score = -1
    for key in valid_keys:
        choice_text = choices.get(key, "")
        candidate = f"{question_text} {choice_text}"
        tokenized_candidate = tokenize(candidate)
        if not tokenized_candidate:
            continue

        total_score = local_bm25_total_score(local_bm25, tokenized_candidate)
        if total_score > max_score:
            max_score = total_score
            best_choice = key

    return best_choice


def make_submission(
    test_file=DEFAULT_TEST_FILE,
    corpus_file=DEFAULT_CORPUS_FILE,
    output_file=DEFAULT_OUTPUT_FILE,
    zip_file=DEFAULT_ZIP_FILE,
):
    total_start = time.perf_counter()

    start_time = time.perf_counter()
    documents = load_corpus(corpus_file)
    if not documents:
        print("[ERROR] Corpus is empty.")
        return
    print(f"[TIME] Load corpus: {time.perf_counter() - start_time:.2f}s")

    start_time = time.perf_counter()
    bm25 = build_bm25_index(documents)
    print(f"[TIME] Build BM25: {time.perf_counter() - start_time:.2f}s")

    start_time = time.perf_counter()
    test_data = load_test_data(test_file)
    print(f"[TIME] Load test: {time.perf_counter() - start_time:.2f}s")

    submissions = []
    total = len(test_data)

    print(f"\n[INFO] Answering {total} questions...")
    print("=" * 60)

    start_time = time.perf_counter()
    for item in progress(test_data, desc="Predicting", total=total):
        question_id = item.get("id")
        question_text = item.get("question", "")
        choices = {
            "A": item.get("A", ""),
            "B": item.get("B", ""),
            "C": item.get("C", ""),
            "D": item.get("D", ""),
        }

        full_query = f"{question_text} {choices['A']} {choices['B']} {choices['C']} {choices['D']}"
        retrieved_docs = retrieve_top_k(full_query, bm25, documents, top_k=TOP_K)
        context_text = " ".join(retrieved_docs) if retrieved_docs else ""
        best_answer = select_answer(question_text, choices, context_text)

        submissions.append(
            {
                "id": question_id,
                "answer": best_answer,
            }
        )

    print(f"[TIME] Predict: {time.perf_counter() - start_time:.2f}s")
    print("=" * 60)

    start_time = time.perf_counter()
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(submissions, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved results to: {output_file}")

    with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_file, os.path.basename(output_file))
        zipf.write(os.path.abspath(__file__), os.path.basename(__file__))
    print(f"[INFO] Created ZIP at: {zip_file}")
    print(f"[TIME] Write output: {time.perf_counter() - start_time:.2f}s")

    answer_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for sub in submissions:
        ans = sub.get("answer", "")
        if ans in answer_counts:
            answer_counts[ans] += 1

    print(f"\n[STATS] Total questions: {len(submissions)}")
    print(
        f"  Answer distribution: A={answer_counts['A']}, B={answer_counts['B']}, "
        f"C={answer_counts['C']}, D={answer_counts['D']}"
    )
    print(f"[TIME] Total: {time.perf_counter() - total_start:.2f}s")


if __name__ == "__main__":
    make_submission()
