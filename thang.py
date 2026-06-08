import json
import math
import os
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


B_RELATIVE_SCORE_THRESHOLD = 0.95
CHUNK_WORDS = 175
CHUNK_OVERLAP = 20
QUESTION_OPTION_TOP_N = 10
LOCAL_BM25_OPTION_WEIGHT = 0.85
VALID_OPTIONS = ["A", "B", "C", "D"]

SCRIPT_DIR = Path(__file__).parent
DEFAULT_TEST_FILE = SCRIPT_DIR / "de_thi.json"
DEFAULT_CORPUS_FILE = SCRIPT_DIR / "dataset.json"
DEFAULT_OUTPUT_FILE = SCRIPT_DIR / "submission.json"
DEFAULT_ZIP_FILE = SCRIPT_DIR / "submission.zip"

RE_LEADING_ZERO = re.compile(r"\b0*(\d+)\b")
RE_REMOVE_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
RE_MULTI_SPACE = re.compile(r"\s+")
RE_NUM = re.compile(r"\d+")

def preprocess_text(text: str) -> str:
    text = (text or "").lower()
    text = RE_LEADING_ZERO.sub(r"\1", text)
    text = RE_REMOVE_PUNCT.sub(" ", text)
    text = RE_MULTI_SPACE.sub(" ", text).strip()
    return text


def tokenize(text: str):
    if not text:
        return []

    return text.split()


def get_choice(question, option):
    return question.get(option, "")


def extract_numbers(text: str):
    return set(RE_NUM.findall(text or ""))


def iter_json_records(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        first_char = ""
        while True:
            ch = f.read(1)
            if not ch:
                break
            if not ch.isspace():
                first_char = ch
                break

        f.seek(0)

        if first_char == "[":
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        yield item
            return

        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def split_into_paragraphs(title: str, content: str):
    full_text = f"{title}\n\n{content}".strip()
    if not full_text:
        return []

    parts = re.split(r"\n\s*\n+", full_text)
    return [part.strip() for part in parts if part.strip()]


def chunk_document(doc_id: int, title: str, content: str):
    paragraphs = split_into_paragraphs(title, content)
    if not paragraphs:
        return []

    tokens = []
    for paragraph in paragraphs:
        processed = preprocess_text(paragraph)
        if processed:
            tokens.extend(tokenize(processed))

    if not tokens:
        return []

    chunks = []
    if CHUNK_WORDS <= 0 or len(tokens) <= CHUNK_WORDS:
        return [
            {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_0",
                "text": " ".join(tokens),
            }
        ]

    step = max(1, CHUNK_WORDS - CHUNK_OVERLAP)
    for chunk_idx, start in enumerate(range(0, len(tokens), step)):
        chunk_tokens = tokens[start:start + CHUNK_WORDS]
        if not chunk_tokens:
            continue
        chunks.append(
            {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_{chunk_idx}",
                "text": " ".join(chunk_tokens),
            }
        )

    return chunks


def load_and_chunk_corpus(corpus_file="dataset.json"):
    chunks = []

    records = iter_json_records(corpus_file)
    for doc_id, doc in enumerate(records):
        title = doc.get("title", "")
        demuc_name = doc.get("demuc_name", "")
        chude_name = doc.get("chude_name", "")
        metadata = doc.get("metadata") or {}
        source_info = (
            metadata.get("source_info", "") if isinstance(metadata, dict) else ""
        )
        content = doc.get("content", "")

        enriched_title = f"{title}\n{demuc_name}\n{chude_name}\n{source_info}".strip()
        chunks.extend(chunk_document(doc_id, enriched_title, content))

    return chunks


def load_questions(question_file="de_thi.json"):
    with open(question_file, "r", encoding="utf-8") as f:
        return json.load(f)


class BM25Index:
    def __init__(self, chunks, k1=1.4, b=0.8):
        self.k1 = k1
        self.b = b

        self.chunk_ids = []
        self.chunk_texts = []
        self.chunk_token_sets = []
        self.chunk_numbers = []
        self.doc_lengths = []

        self.inverted_index = defaultdict(list)
        self.doc_freq = {}
        self.idf = {}
        self.avg_doc_len = 0.0
        self.num_docs = 0
        self.bm25_vocab = {}
        self.bm25_matrix = None

        self._build(chunks)

    def _build(self, chunks):
        total_len = 0
        temp_df = defaultdict(int)

        for doc_idx, chunk in enumerate(chunks):
            text = chunk["text"]
            tokens = tokenize(text)
            tf_counter = Counter(tokens)

            self.chunk_ids.append(chunk["chunk_id"])
            self.chunk_texts.append(text)
            self.chunk_token_sets.append(set(tf_counter.keys()))
            self.chunk_numbers.append(extract_numbers(text))
            self.doc_lengths.append(len(tokens))

            total_len += len(tokens)

            for term, tf in tf_counter.items():
                self.inverted_index[term].append((doc_idx, tf))
                temp_df[term] += 1

        self.num_docs = len(self.chunk_ids)
        self.avg_doc_len = total_len / self.num_docs if self.num_docs else 0.0
        self.doc_freq = dict(temp_df)

        for term, df in self.doc_freq.items():
            self.idf[term] = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)

        rows = []
        cols = []
        data = []
        for term_id, (term, postings) in enumerate(
            self.inverted_index.items()
        ):
            self.bm25_vocab[term] = term_id
            idf = self.idf.get(term, 0.0)
            if idf <= 0:
                continue

            for doc_idx, tf in postings:
                doc_len = self.doc_lengths[doc_idx]
                denom = tf + self.k1 * (
                    1 - self.b + self.b * (doc_len / self.avg_doc_len)
                )
                weight = idf * (tf * (self.k1 + 1) / denom)
                rows.append(doc_idx)
                cols.append(term_id)
                data.append(weight)

        self.bm25_matrix = coo_matrix(
            (data, (rows, cols)),
            shape=(self.num_docs, len(self.bm25_vocab)),
            dtype=np.float32,
        ).tocsr()

    def _bm25_scores(self, query_text):
        if self.bm25_matrix is None:
            return np.zeros(self.num_docs, dtype=np.float32)

        query_freq = defaultdict(int)
        for term in tokenize(query_text):
            term_id = self.bm25_vocab.get(term)
            if term_id is not None:
                query_freq[term_id] += 1

        if not query_freq:
            return np.zeros(self.num_docs, dtype=np.float32)

        query_vector = csr_matrix(
            (
                list(query_freq.values()),
                ([0] * len(query_freq), list(query_freq.keys())),
            ),
            shape=(1, len(self.bm25_vocab)),
            dtype=np.float32,
        )
        return (query_vector @ self.bm25_matrix.T).toarray().ravel()

    def _bm25_ranked(self, query_text, top_n):
        scores = self._bm25_scores(query_text)
        if scores.size == 0:
            return []

        k = min(top_n, scores.size)
        if k <= 0:
            return []

        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [(int(idx), float(scores[idx])) for idx in top_idx if scores[idx] > 0]

    def query(self, query_text, top_n=20):
        query_text = preprocess_text(query_text)
        return self._bm25_ranked(query_text, top_n)


def option_match_score(option_text, retrieved_docs, index):
    opt_text = preprocess_text(option_text)
    opt_tokens = tokenize(opt_text)
    opt_set = set(opt_tokens)
    opt_nums = extract_numbers(opt_text)

    if not opt_set and not opt_nums:
        return 0.0

    score = 0.0
    for rank, (doc_idx, bm25_score) in enumerate(retrieved_docs, start=1):
        chunk_text = index.chunk_texts[doc_idx]
        chunk_terms = index.chunk_token_sets[doc_idx]
        chunk_nums = index.chunk_numbers[doc_idx]

        overlap = len(opt_set & chunk_terms) / max(1, len(opt_set))
        exact_bonus = 1.5 if opt_text and opt_text in chunk_text else 0.0
        num_bonus = (
            len(opt_nums & chunk_nums) / max(1, len(opt_nums)) if opt_nums else 0.0
        )
        rank_weight = 1.0 / math.log2(rank + 1)

        score += rank_weight * (
            bm25_score * (0.65 + 0.35 * overlap)
            + exact_bonus
            + 0.8 * num_bonus
        )

    return score


def local_bm25_option_scores(question_text, question, retrieved_docs, index):
    context_docs = []
    for doc_idx, _ in retrieved_docs:
        tokens = tokenize(index.chunk_texts[doc_idx])
        if tokens:
            context_docs.append(tokens)

    if not context_docs:
        return {}

    local_doc_lengths = [len(tokens) for tokens in context_docs]
    avg_doc_len = sum(local_doc_lengths) / len(local_doc_lengths)
    doc_freq = defaultdict(int)
    term_freqs = []

    for tokens in context_docs:
        term_freq = Counter(tokens)
        term_freqs.append(term_freq)
        for term in term_freq:
            doc_freq[term] += 1

    n_docs = len(context_docs)
    scores_by_option = {}
    for option in VALID_OPTIONS:
        candidate = preprocess_text(f"{question_text} {get_choice(question, option)}")
        query_terms = list(dict.fromkeys(tokenize(candidate)))
        score = 0.0

        for term in query_terms:
            df = doc_freq.get(term)
            if not df:
                continue
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

            for doc_idx, term_freq in enumerate(term_freqs):
                tf = term_freq.get(term, 0)
                if not tf:
                    continue
                doc_len = local_doc_lengths[doc_idx]
                denom = tf + 1.5 * (1 - 0.75 + 0.75 * (doc_len / avg_doc_len))
                score += idf * (tf * (1.5 + 1) / denom)

        scores_by_option[option] = score

    return scores_by_option


def question_option_score(question_text, option_text, index):
    query = f"{question_text} {option_text}"
    hits = index.query(query, top_n=QUESTION_OPTION_TOP_N)
    if not hits:
        return 0.0
    return sum(score for _, score in hits) / len(hits)


def answer_question(question, index, retrieval_cache, top_k_chunks=15):
    question_text = question.get("question", "")

    normalized_question = preprocess_text(question_text)
    if normalized_question in retrieval_cache:
        retrieved_docs = retrieval_cache[normalized_question]
    else:
        retrieved_docs = index.query(normalized_question, top_n=top_k_chunks)
        retrieval_cache[normalized_question] = retrieved_docs

    if not retrieved_docs:
        expanded_query = " ".join(
            [question_text] + [get_choice(question, option) for option in VALID_OPTIONS]
        )
        retrieved_docs = index.query(expanded_query, top_n=top_k_chunks)

    if not retrieved_docs:
        return "B"

    option_scores = {
        option: option_match_score(get_choice(question, option), retrieved_docs, index)
        for option in VALID_OPTIONS
    }

    local_scores = local_bm25_option_scores(question_text, question, retrieved_docs, index)
    for option, score in local_scores.items():
        option_scores[option] += LOCAL_BM25_OPTION_WEIGHT * score

    if all(score == 0.0 for score in option_scores.values()):
        for option in VALID_OPTIONS:
            option_scores[option] = question_option_score(
                question_text,
                get_choice(question, option),
                index,
            )

    best_answer = max(option_scores, key=option_scores.get)
    best_score = option_scores[best_answer]
    b_score = option_scores.get("B", 0.0)

    if b_score >= best_score * B_RELATIVE_SCORE_THRESHOLD:
        return "B"

    return best_answer


def write_submission_files(predictions, output_file, zip_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    with zipfile.ZipFile(zip_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_file, arcname=os.path.basename(output_file))
        zf.write(__file__, arcname=os.path.basename(__file__))


def make_submission(
    test_file=DEFAULT_TEST_FILE,
    corpus_file=DEFAULT_CORPUS_FILE,
    output_file=DEFAULT_OUTPUT_FILE,
    zip_file=DEFAULT_ZIP_FILE,
    top_k_chunks=20,
):
    print("Loading and chunking corpus...")
    start_time = time.perf_counter()
    chunks = load_and_chunk_corpus(corpus_file)
    if not chunks:
        print("Corpus is empty or could not be read.")
        return

    chunk_time = time.perf_counter() - start_time
    print(f"Created {len(chunks)} chunks in {chunk_time:.2f}s.")

    print("Building BM25 index...")
    start_time = time.perf_counter()
    index = BM25Index(chunks)
    index_time = time.perf_counter() - start_time
    print(f"Built index for {index.num_docs} chunks in {index_time:.2f}s.")

    print("Loading questions...")
    start_time = time.perf_counter()
    questions = load_questions(test_file)
    question_load_time = time.perf_counter() - start_time
    print(f"Loaded {len(questions)} questions in {question_load_time:.2f}s.")

    predictions = []
    retrieval_cache = {}

    print("Predicting answers...")
    start_time = time.perf_counter()
    for question in questions:
        answer = answer_question(
            question,
            index,
            retrieval_cache,
            top_k_chunks=top_k_chunks,
        )
        predictions.append(
            {
                "id": question.get("id"),
                "answer": answer,
            }
        )
    predict_time = time.perf_counter() - start_time
    print(f"Predicted {len(questions)} questions in {predict_time:.2f}s.")

    print("Writing submission.json...")
    start_time = time.perf_counter()
    print("Creating submission.zip...")
    write_submission_files(predictions, output_file, zip_file)
    write_time = time.perf_counter() - start_time

    total_time = chunk_time + index_time + question_load_time + predict_time + write_time
    print(f"Wrote files in {write_time:.2f}s.")
    print(f"Done in {total_time:.2f}s. Created: {output_file} and {zip_file}")


if __name__ == "__main__":
    make_submission(
        test_file=DEFAULT_TEST_FILE,
        corpus_file=DEFAULT_CORPUS_FILE,
        output_file=DEFAULT_OUTPUT_FILE,
        zip_file=DEFAULT_ZIP_FILE,
    )
