from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────

_SENTENCE_MODEL = None

def _get_sentence_model():
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SENTENCE_MODEL


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    metadata = metadata.copy() if metadata else {}
    if not text or not text.strip():
        return []

    from numpy import dot
    from numpy.linalg import norm

    # Split text thành sentences
    raw_sentences = re.split(r'(?<=[.!?])\s+|\n\n+', text.strip())
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(text=sentences[0], metadata={**metadata, "strategy": "semantic", "chunk_index": 0})]

    model = _get_sentence_model()
    embeddings = model.encode(sentences)

    def cosine_sim(a, b):
        norm_a = norm(a)
        norm_b = norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot(a, b) / (norm_a * norm_b + 1e-9))

    chunk_groups: list[list[str]] = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = cosine_sim(embeddings[i - 1], embeddings[i])
        if sim < threshold:
            chunk_groups.append([sentences[i]])
        else:
            chunk_groups[-1].append(sentences[i])

    chunks = []
    for idx, group in enumerate(chunk_groups):
        chunk_text = " ".join(group).strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={**metadata, "strategy": "semantic", "chunk_index": idx},
                )
            )
    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata.copy() if metadata else {}
    if not text or not text.strip():
        return ([], [])

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return ([], [])

    # 1. Gộp paragraphs thành parent chunks (mỗi parent ≤ parent_size chars)
    parent_texts = []
    current_parent = ""
    for para in paragraphs:
        if current_parent and (len(current_parent) + len(para) + 2 > parent_size):
            parent_texts.append(current_parent.strip())
            current_parent = ""
        
        if len(para) > parent_size and not current_parent:
            # Paragraph lớn hơn parent_size -> cắt thành các phần nhỏ
            for i in range(0, len(para), parent_size):
                sub = para[i:i + parent_size].strip()
                if sub:
                    parent_texts.append(sub)
        else:
            current_parent = f"{current_parent}\n\n{para}".strip() if current_parent else para

    if current_parent.strip():
        parent_texts.append(current_parent.strip())

    # 2. Tạo parents và cắt thành children
    parents = []
    children = []

    for p_idx, p_text in enumerate(parent_texts):
        pid = f"parent_{p_idx}"
        parent_chunk = Chunk(
            text=p_text,
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid, "parent_index": p_idx},
            parent_id=pid,
        )
        parents.append(parent_chunk)

        # Cắt parent thành children (mỗi child ≤ child_size chars)
        raw_units = re.split(r'(?<=[.!?])\s+|\n+', p_text)
        units = [u.strip() for u in raw_units if u.strip()]
        if not units:
            units = [p_text]

        child_texts = []
        current_child = ""
        for unit in units:
            if current_child and (len(current_child) + len(unit) + 1 > child_size):
                child_texts.append(current_child.strip())
                current_child = ""
            
            if len(unit) > child_size and not current_child:
                for j in range(0, len(unit), child_size):
                    sub = unit[j:j + child_size].strip()
                    if sub:
                        child_texts.append(sub)
            else:
                current_child = f"{current_child} {unit}".strip() if current_child else unit

        if current_child.strip():
            child_texts.append(current_child.strip())

        for c_text in child_texts:
            child_chunk = Chunk(
                text=c_text,
                metadata={
                    **metadata,
                    "chunk_type": "child",
                    "parent_id": pid,
                    "child_index": len(children),
                },
                parent_id=pid,
            )
            children.append(child_chunk)

    return (parents, children)


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata.copy() if metadata else {}
    if not text or not text.strip():
        return []

    lines = text.splitlines()
    header_pattern = re.compile(r'^(#{1,3})\s+(.+)$')

    sections: list[dict] = []
    current_header = ""
    current_section_title = "General"
    current_lines = []

    for line in lines:
        match = header_pattern.match(line.strip())
        if match:
            # Header mới: lưu section trước nếu có nội dung
            if current_lines:
                content = "\n".join(current_lines).strip()
                if content:
                    full_text = f"{current_header}\n\n{content}".strip() if current_header else content
                    sections.append({
                        "header": current_section_title,
                        "text": full_text,
                    })
                current_lines = []
            
            current_header = line.strip()
            current_section_title = match.group(2).strip()
        else:
            current_lines.append(line)

    if current_lines:
        content = "\n".join(current_lines).strip()
        if content:
            full_text = f"{current_header}\n\n{content}".strip() if current_header else content
            sections.append({
                "header": current_section_title,
                "text": full_text,
            })
    elif current_header and not sections:
        sections.append({
            "header": current_section_title,
            "text": current_header,
        })

    chunks = []
    for idx, sec in enumerate(sections):
        chunks.append(
            Chunk(
                text=sec["text"],
                metadata={
                    **metadata,
                    "section": sec["header"],
                    "strategy": "structure",
                    "chunk_index": idx,
                },
            )
        )

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
