import json
import re
import zipfile
import os
from collections import defaultdict
from tqdm import tqdm



def tokenize(text: str) -> list:
    if not text:
        return []
    return re.findall(r'\w+', text.lower())

def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def bm25_score(query_tokens: list, doc_tokens: list, doc_len: int, avg_doc_len: float, total_docs: int, doc_freq: dict, k1: float = 1.5, b: float = 0.75) -> float:
    score = 0.0
    for token in query_tokens:
        df = doc_freq.get(token, 0)
        if df > 0:
            idf = (total_docs + 0.5) / (df + 0.5)
            tf = doc_tokens.count(token)
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))
    return score

class BM25Retriever:
    def __init__(self, doc_token_sets: list, doc_token_lists: list):
        self.doc_token_sets = doc_token_sets
        self.doc_token_lists = doc_token_lists
        self.doc_len = [len(tokens) for tokens in doc_token_lists]
        self.total_docs = len(doc_token_sets)
        self.avg_doc_len = sum(self.doc_len) / self.total_docs if self.total_docs else 1
        self.doc_freq = defaultdict(int)
        for tokens in doc_token_sets:
            for token in tokens:
                self.doc_freq[token] += 1

    def search(self, query_tokens: list, k: int = 5) -> list:
        scores = []
        for i in range(self.total_docs):
            score = bm25_score(query_tokens, self.doc_token_lists[i], self.doc_len[i], self.avg_doc_len, self.total_docs, self.doc_freq)
            if score > 0:
                scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [self.doc_token_sets[idx] for idx, _ in scores[:k]]

def load_corpus(corpus_file: str = "dataset.json"):
    documents = []
    doc_token_lists = []
    doc_token_sets = []
    try:
        with open(corpus_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for doc in data:
            text = f"{doc.get('title', '')} {doc.get('content', '')}"
            documents.append(text)
            tokens = tokenize(text)
            doc_token_lists.append(tokens)
            doc_token_sets.append(set(tokens))
    except FileNotFoundError:
        return [], [], []
    except json.JSONDecodeError:
        return [], [], []
    return documents, doc_token_lists, doc_token_sets

def edit_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]

def select_answers(item: dict, candidate_docs: list, valid_choices: list, threshold_ratio: float = 0.8) -> list:
    scores = {}
    max_score = 0.0
    
    for key in valid_choices:
        choice_text = item.get(key, "")
        best_doc_score = 0.0
        for doc_tokens in candidate_docs:
            choice_tokens = tokenize(choice_text)
            score = jaccard(set(choice_tokens), doc_tokens)
            if score > best_doc_score:
                best_doc_score = score
        
                # Thêm điểm fuzzy matching dựa trên khoảng cách edit distance
        if best_doc_score == 0 and choice_text:
            # Tìm từ chung trong doc để tính edit distance nếu không có từ chung
            # Tối ưu: so sánh câu trả lời với toàn bộ doc text (đơn giản hóa)
            best_doc_text = " ".join(candidate_docs[0]) if candidate_docs else ""
            # Tính khoảng cách chuẩn hóa (normalized edit distance)
            # Lưu ý: Edit distance tính trên chuỗi, không phải tập hợp từ
            # Ở đây ta giả sử so sánh chuỗi câu trả lời với chuỗi tài liệu
            # Để tránh nặng, ta chỉ lấy một đoạn văn bản đại diện
            dist = edit_distance(choice_text.lower(), best_doc_text.lower())
            if best_doc_text:
                norm_dist = dist / max(len(choice_text), len(best_doc_text))
                # Chuyển khoảng cách thành điểm (1 - khoảng cách)
                fuzzy_score = 1.0 - norm_dist
                best_doc_score = fuzzy_score
        
        scores[key] = best_doc_score
        if best_doc_score > max_score:
            max_score = best_doc_score

    threshold = threshold_ratio * max_score if max_score > 0 else 0.0
    
    selected = []
    for key in valid_choices:
        if scores[key] >= threshold and scores[key] > 0:
            selected.append(key)
    
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
    documents, doc_token_lists, doc_token_sets = load_corpus(corpus_file)
    if not documents:
        return
    print(f"Đã tải {len(documents)} tài liệu từ tập corpus...")

    try:
        with open(test_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {test_file}.")
        return

    retriever = BM25Retriever(doc_token_sets, doc_token_lists)
    submissions = []
    valid_choices = ["A", "B", "C", "D"]

    print("Bắt đầu truy xuất (BM25 + Fuzzy) và dự đoán đáp án...")

    for item in tqdm(test_data, desc='Processing...'):
        question_id = item.get("id")
        question_text = item.get("question", "")
        
        query_tokens = tokenize(question_text)
        candidate_docs = retriever.search(query_tokens, k=top_k_docs)

        answers = select_answers(item, candidate_docs, valid_choices, threshold_ratio)

        submissions.append({
            "id": question_id,
            "answer": answers
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(submissions, f, ensure_ascii=False, indent=2)

    current_script = __file__
    with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_file, arcname=os.path.basename(output_file))
        zipf.write(current_script, arcname=os.path.basename(current_script))

    print(f"Đã xử lý xong {len(submissions)} câu hỏi.")
    print(f"File nộp bài ZIP đã sẵn sàng tại: {zip_file}")

if __name__ == "__main__":
    make_submission(
        test_file="de_thi.json",
        corpus_file="dataset.json",
        output_file="submission.json",
        zip_file="submission.zip",
        threshold_ratio=0.1,
        top_k_docs=20,
    )