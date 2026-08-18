# Failure Analysis — Lab 18: Production RAG

**Nhóm:** AICB-K34 Lab 18 Production RAG  
**Thành viên:** Trịnh Quang Anh (Thực hiện toàn bộ Pipeline: M1 Chunking, M2 Search, M3 Rerank, M4 Eval, M5 Enrichment)

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|---|:---:|:---:|:---:|
| **Faithfulness** | 0.9033 | 0.6083 | -0.2950 |
| **Answer Relevancy** | 0.5165 | 0.6520 | +0.1355 |
| **Context Precision** | 0.7285 | 0.9250 | **+0.1965** |
| **Context Recall** | 0.7212 | 0.7833 | **+0.0621** |

> **Nhận xét chính:**
> - **Context Precision tăng vượt trội (+27% tương đối, đạt 0.9250)**: Nhờ sự kết hợp giữa **Hierarchical Chunking** (Parent-Child) và **Cross-Encoder Reranker** (`BAAI/bge-reranker-v2-m3`), loại bỏ hầu hết các đoạn văn nhiễu không liên quan, đưa đúng các đoạn thông tin cốt lõi vào top-3.
> - **Context Recall tăng lên 0.7833**: Nhờ cơ chế **Hybrid Search (BM25 + Dense + RRF)** giúp bù trừ điểm yếu từ vựng (lexical gap) và ngữ nghĩa (semantic search).
> - **Faithfulness**: Ở các câu hỏi đòi hỏi tính toán số học phức tạp hoặc multi-hop reasoning, LLM đôi khi suy diễn thêm phép tính dẫn đến điểm faithfulness bị trừ điểm trong benchmark.

---

## Bottom-5 Failures

### #1
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Thời hạn thanh toán là 15 ngày. Quá hạn 5 ngày, bị tính phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng (tính pro-rata khoảng 50.000 VNĐ cho 5 ngày).
- **Got:** LLM trả lời công thức 2%/tháng nhưng tính nhầm số ngày phạt hoặc suy diễn số tiền phạt tròn tháng.
- **Worst metric:** Faithfulness (0.00)
- **Error Tree:** Output sai số học → Context truy xuất đúng (có quy định 15 ngày & phạt 2%/tháng) → LLM Reasoning Limitation.
- **Root cause:** LLM không có khả năng tính toán số học chuẩn xác (arithmetic reasoning) nếu chỉ dùng prompt text thuần túy.
- **Suggested fix:** Tích hợp Tool Calling / Python Code Interpreter hoặc dùng Chain-of-Thought (CoT) hướng dẫn từng bước tính toán.

### #2
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** LLM tìm được thông tin nghỉ phép nhưng bỏ sót đoạn thang bảng lương hoặc ngược lại.
- **Worst metric:** Context Recall (0.50)
- **Error Tree:** Thiếu thông tin trong context → Cần 2 tài liệu độc lập (chính sách nghỉ phép + quy chế lương) → Multi-hop query failure.
- **Root cause:** Single query embedding không bao quát được đồng thời 2 thực thể khác nhau trong 1 lượt tìm kiếm.
- **Suggested fix:** Áp dụng **Query Decomposition / Sub-question Query Engine** (tách câu hỏi thành 2 câu hỏi con: "Nghỉ phép cho 9 năm thâm niên" và "Khoảng lương cấp Senior") rồi merge context.

### #3
- **Question:** Mentor và buddy của nhân viên mới có thể là cùng một người không? Quản lý trực tiếp có thể làm mentor không?
- **Expected:** KHÔNG cho cả hai. Mentor và buddy phải là hai người khác nhau. Quản lý trực tiếp không được làm mentor hoặc buddy.
- **Got:** Trả lời đúng phần Buddy nhưng thiếu điều kiện về Quản lý trực tiếp làm Mentor.
- **Worst metric:** Context Recall (0.6667)
- **Error Tree:** Context chỉ trích xuất được 1 đoạn văn chứa quy định buddy, đoạn về cấm quản lý làm mentor nằm ở cuối tài liệu.
- **Root cause:** Child chunking cắt rời 2 quy định ra hai đoạn khác nhau, Reranker chỉ lấy top-3 đoạn gần nhất.
- **Suggested fix:** Tăng kích thước `parent_size` hoặc dùng **Contextual Prepend (M5)** để gắn tiêu đề ngữ cảnh đầy đủ cho mọi child chunk.

### #4
- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Cần Giám đốc phòng ban (Director) phê duyệt; cần xác nhận cấu hình kỹ thuật từ CNTT và tối thiểu 3 báo giá.
- **Got:** LLM nêu được cấp phê duyệt nhưng thiếu chi tiết đính kèm 3 báo giá do đoạn quy định phân bổ rải rác.
- **Worst metric:** Context Precision (0.5833)
- **Error Tree:** Top-3 contexts chứa cả thông tin mua sắm chung và mua sắm tài sản cố định gây loãng câu trả lời.
- **Root cause:** Từ khóa "laptop 30 triệu" khớp cả quy định IT lẫn quy chế mua sắm tài chính, BM25 kéo theo các chunk mua sắm chung.
- **Suggested fix:** Thêm **Metadata Filter** theo `category: "it_procurement"` để lọc chunk trước khi rerank.

### #5
- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** Cam kết làm việc 1 năm, nghỉ sau 8 tháng vi phạm cam kết nên phải hoàn trả 100% chi phí (25.000.000 VNĐ).
- **Got:** Context đưa vào chứa quy định đào tạo nội bộ thay vì điều khoản cam kết hoàn trả kinh phí tài trợ ngoài.
- **Worst metric:** Context Precision (0.5833)
- **Error Tree:** Retrieval trả về nhiều chunk đào tạo chung, đẩy chunk cam kết bồi hoàn xuống rank thấp hơn.
- **Root cause:** Thuật ngữ "tài trợ khóa học" bị phân tán ngữ nghĩa giữa "Chính sách đào tạo" và "Hợp đồng cam kết".
- **Suggested fix:** Sử dụng **HyQA (Hypothesis Question-Answer)** trong M5 để sinh câu hỏi giả định *"Nghỉ việc trước hạn cam kết đào tạo bồi thường bao nhiêu?"* đính kèm chunk.

---

## Case Study (cho presentation)

**Question chọn phân tích:**  
> *"Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?"*

**Error Tree walkthrough:**
1. **Output đúng?** $\rightarrow$ Chưa đầy đủ (chỉ trả lời được số ngày phép hoặc khoảng lương, thiếu 1 vế).
2. **Context đúng?** $\rightarrow$ Sai/Thiếu: Top-3 retrieved contexts chỉ bao gồm các chunk từ `nghi_phep_nam_v2024.md`, thiếu chunk từ `thang_bang_luong.md`.
3. **Query rewrite OK?** $\rightarrow$ Hệ thống hiện tại nhận câu hỏi dạng single string, embedding vector bị kéo lệch về phía semantic của "nghỉ phép" thay vì phân bổ đều cho "lương Senior".
4. **Fix ở bước:** Bước **Query Transformation / Decomposition** trước khi tìm kiếm.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Thêm module **Query Decomposition**: Tự động phân tích câu hỏi phức hợp thành các câu hỏi đơn lẻ và thực hiện song song multi-hop retrieval.
- Áp dụng **HyDE (Hypothetical Document Embeddings)** để tạo câu trả lời mẫu trước khi vector search.
- Bổ sung **Post-retrieval LLM Synthesis Prompt** với hướng dẫn định dạng bảng đối chiếu chi tiết.
