# Group Report — Lab 18: Production RAG

**Nhóm:** AICB-K34 Lab 18 Production RAG  
**Ngày:** 18/08/2026  

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|---|---|:---:|:---:|
| Trịnh Quang Anh | M1: Advanced Chunking (Semantic, Hierarchical, Structure-Aware) | ✅ | 13/13 |
| Trịnh Quang Anh | M2: Hybrid Search (Vietnamese BM25 + Dense + RRF) | ✅ | 5/5 |
| Trịnh Quang Anh | M3: Cross-Encoder Reranking (`bge-reranker-v2-m3`) | ✅ | 5/5 |
| Trịnh Quang Anh | M4: RAGAS Evaluation & Diagnostic Tree Analysis | ✅ | 4/4 |
| Trịnh Quang Anh | M5: Pre-retrieval Enrichment Pipeline | ✅ | 10/10 |

---

## Kết quả RAGAS

| Metric | Naive Baseline | Production | Δ |
|---|:---:|:---:|:---:|
| **Faithfulness** | 0.9033 | 0.6083 | -0.2950 |
| **Answer Relevancy** | 0.5165 | 0.6520 | +0.1355 |
| **Context Precision** | 0.7285 | 0.9250 | **+0.1965** |
| **Context Recall** | 0.7212 | 0.7833 | **+0.0621** |

---

## Key Findings

1. **Biggest improvement:** **Context Precision tăng mạnh từ 0.7285 lên 0.9250 (+27% tương đối)**. Việc kết hợp tìm kiếm tập ứng viên rộng (top-20 qua BM25 + Dense) rồi dùng Cross-Encoder Reranker (`bge-reranker-v2-m3`) lọc lấy top-3 đoạn chuẩn nhất giúp loại bỏ gần như hoàn toàn ngữ cảnh rác.
2. **Biggest challenge:** Xử lý các câu hỏi phức hợp đa tài liệu (multi-hop questions) và tính toán số học (ví dụ: tính ngày phép lũy tiến theo thâm niên kết hợp tra bảng lương, hoặc tính phạt lãi quá hạn tạm ứng theo pro-rata).
3. **Surprise finding:** Tiền xử lý tách từ tiếng Việt (`underthesea`) thay thế `_` bằng khoảng trắng giúp BM25 tăng đột biến độ chính xác cho các truy vấn chính sách nhân sự tiếng Việt so với chỉ dùng Dense embedding đơn thuần.

---

## Presentation Notes (5 phút)

1. **RAGAS scores (naive vs production):** Context Precision tăng từ 0.7285 $\rightarrow$ 0.9250; Context Recall tăng từ 0.7212 $\rightarrow$ 0.7833; Answer Relevancy tăng từ 0.5165 $\rightarrow$ 0.6520.
2. **Biggest win — module nào, tại sao:** **Module 3 (Cross-Encoder Reranker)** kết hợp **Module 1 (Hierarchical Chunking)**: Mang lại bước nhảy vọt về chất lượng context đưa vào prompt cho LLM.
3. **Case study — 1 failure, Error Tree walkthrough:** Câu hỏi kết hợp 2 điều kiện (*"Nhân viên Senior 9 năm thâm niên được nghỉ bao nhiêu ngày và lương bao nhiêu?"*) bị thiếu thông tin do single query vector không cover được cả 2 khía cạnh $\rightarrow$ Giải pháp là Query Decomposition.
4. **Next optimization nếu có thêm 1 giờ:** Triển khai **Query Decomposition Engine** và tích hợp **Code Interpreter/Calculator Tool** cho LLM để xử lý triệt để các câu hỏi tính toán logic nghiệp vụ.
