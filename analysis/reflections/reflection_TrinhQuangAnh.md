# Individual Reflection — Lab 18: Production RAG Pipeline

**Tên:** Trịnh Quang Anh  
**Module phụ trách:** Toàn bộ Pipeline (M1: Chunking, M2: Hybrid Search, M3: Reranking, M4: Evaluation, M5: Enrichment)

---

## 1. Đóng góp kỹ thuật

- **Các module đã implement:**
  - **M1 (Advanced Chunking):** `chunk_semantic`, `chunk_hierarchical`, `chunk_structure_aware`.
  - **M2 (Hybrid Search):** `segment_vietnamese`, `BM25Search`, `DenseSearch` (Qdrant + in-memory fallback), `reciprocal_rank_fusion` (RRF).
  - **M3 (Reranking):** `CrossEncoderReranker` (`BAAI/bge-reranker-v2-m3`), `FlashrankReranker`, `benchmark_reranker`.
  - **M4 (Evaluation):** `evaluate_ragas` (tính 4 chỉ số Faithfulness, Answer Relevancy, Context Precision, Context Recall), `failure_analysis` (Diagnostic Tree).
  - **M5 (Enrichment):** `summarize_chunk`, `generate_hypothesis_questions` (HyQA), `contextual_prepend`, `extract_metadata`, và chế độ Combined single-call `_enrich_single_call`.
- **Số tests pass:** **37 / 37 tests (100% pass)** qua `pytest tests/ -v`.

---

## 2. Mapping bài giảng & Kiến thức học được

| Lecture Concept | Module | Hàm/Class cụ thể | Quan sát & Đánh giá kỹ thuật |
|---|---|---|---|
| **Semantic Chunking** | M1 | `chunk_semantic()` | Tách câu và tính cosine similarity giữa các embeddings liên tiếp (`all-MiniLM-L6-v2`), gom nhóm câu theo ngưỡng 0.85 giúp giữ trọn vẹn ngữ nghĩa từng ý. |
| **Hierarchical Chunking** | M1 | `chunk_hierarchical()` | Tạo cấu trúc Parent-Child (Parent 2048 chars, Child 256 chars), truy xuất Child để đạt precision cao và trả về Parent để cung cấp đủ ngữ cảnh cho LLM. |
| **Structure-Aware Chunking** | M1 | `chunk_structure_aware()` | Phân tích cú pháp Markdown headers (`#`, `##`, `###`) để chia đoạn theo section logic, bảo toàn nguyên vẹn danh sách, bảng biểu và metadata section. |
| **Hybrid Search & RRF** | M2 | `reciprocal_rank_fusion()` | Kết hợp BM25 (xử lý từ vựng chính xác sau khi tách từ tiếng Việt qua `underthesea`) và Dense Search (`bge-m3`), RRF hòa trộn điểm rank giúp Context Recall tăng lên 0.7833. |
| **Cross-Encoder Reranking** | M3 | `CrossEncoderReranker.rerank()` | Nhận diện tương tác sâu cặp `(query, document)` qua mô hình `bge-reranker-v2-m3`, đưa **Context Precision từ 0.7285 lên 0.9250 (+27%)**. |
| **RAGAS 4 Metrics** | M4 | `evaluate_ragas()` | Tự động hóa đánh giá chất lượng RAG trên 4 chiều: Faithfulness (độ trung thực), Answer Relevancy (độ phù hợp câu trả lời), Context Precision (độ chuẩn của tài liệu trích xuất), Context Recall (độ bao phủ tài liệu). |
| **Contextual Prepend** | M5 | `contextual_prepend()` | Bổ sung 1 câu ngữ cảnh xuất xứ tài liệu vào đầu mỗi chunk, giúp các chunk nhỏ không bị mất thông tin nguồn gốc. |
| **HyQA (Hypothesis QA)** | M5 | `generate_hypothesis_questions()` | Sinh câu hỏi tiềm năng cho từng chunk giúp thu hẹp khoảng cách từ vựng (vocabulary gap) giữa câu hỏi của người dùng và văn bản gốc. |

---

## 3. Khó khăn kỹ thuật & Cách giải quyết

1. **Lỗi biên dịch package trên môi trường Windows & Python 3.13:**
   - *Lỗi:* `numpy` phiên bản cũ (1.26.4) cố gắng build từ mã nguồn qua Meson và thất bại do thiếu C/C++ compiler (`cl.exe`, `gcc`).
   - *Cách giải quyết:* Nâng cấp cấu hình `requirements.txt` và môi trường sang các phiên bản hỗ trợ Python 3.13 pre-built binary wheels (`langchain>=0.3`, `ragas>=0.2`, `numpy 2.x`), đồng thời cấu hình fallback tính toán an toàn khi gọi RAGAS.
2. **Khóa file tải trọng số HuggingFace (`WeakFileLock`):**
   - *Lỗi:* Các tiến trình tải đồng thời mô hình embedding lớn (>2.2GB) gây khóa file lock.
   - *Cách giải quyết:* Đảm bảo cơ chế tải tuần tự và pre-cache trọng số vào ổ đĩa cục bộ, cho phép kiểm thử và chạy pipeline không phụ thuộc mạng.
3. **Mã hóa ký tự Unicode trên Windows PowerShell:**
   - *Lỗi:* `UnicodeEncodeError: 'charmap' codec can't encode characters` khi in emoji hoặc tiếng Việt ra màn hình console mặc định `cp1252`.
   - *Cách giải quyết:* Sử dụng biến môi trường `$env:PYTHONUTF8=1` để đồng bộ hóa mã hóa UTF-8 toàn hệ thống.

---

## 4. Kế hoạch ứng dụng cho Project (Action Plan)

### Định hướng áp dụng vào hệ thống RAG thực tế:
1. **Chunking Strategy:** Ưu tiên kết hợp **Structure-Aware** cho các tài liệu có cấu trúc phân cấp (tài liệu kỹ thuật, quy chế, hợp đồng) và **Hierarchical Chunking** cho các văn bản pháp lý dài để cân bằng giữa độ chính xác khi tìm kiếm và độ bao phủ ngữ cảnh khi sinh câu trả lời.
2. **Search Pipeline:** Bắt buộc sử dụng **Hybrid Search (BM25 + Dense Vector)** kết hợp tách từ tiếng Việt cho toàn bộ tài liệu tiếng Việt, khắc phục triệt để hiện tượng bỏ sót từ khóa kỹ thuật hoặc mã số văn bản.
3. **Reranking:** Tích hợp **Cross-Encoder (`bge-reranker-v2-m3`)** cho top-20 ứng viên đầu tiên, rút gọn về top-3 đưa vào prompt nhằm tối ưu Context Precision và giảm thiểu chi phí token của LLM.
4. **Evaluation & Continuous Monitoring:** Thiết lập pipeline đánh giá tự động định kỳ với **RAGAS 4 metrics** trên bộ test set tiêu chuẩn để phát hiện kịp thời các điểm nghẽn về truy xuất (Retrieval) hoặc tạo sinh (Generation).

---

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) | Ghi chú |
|---|:---:|---|
| **Hiểu bài giảng** | 5/5 | Nắm vững toàn bộ chu trình 5 modules và cơ chế đánh giá RAGAS |
| **Code quality** | 5/5 | Code clean, xử lý ngoại lệ đầy đủ, pass 100% unit tests |
| **Problem solving** | 5/5 | Tự xử lý độc lập các lỗi phụ thuộc môi trường Python 3.13 và tối ưu pipeline |
| **Completeness** | 5/5 | Hoàn thành toàn bộ deliverables và báo cáo phân tích theo yêu cầu |
