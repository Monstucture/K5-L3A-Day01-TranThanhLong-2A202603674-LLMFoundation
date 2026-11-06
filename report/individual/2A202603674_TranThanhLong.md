# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603674
- **Họ và tên:** Trần Thanh Long
- **SEED** (in ở ô đầu notebook): 3029

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output |  ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra |  ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước |  ☑ |

## 2. Kết quả chính

*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*
F1 test thường (SUB)  F1 50 case khó  \
1. BoW + LogReg                                0.857           0.333   
2. BoW + MLP                                   0.947           0.476   
3a. Emb + kNN                                  0.974           0.737   
3b. Emb + LogReg                               0.947           0.553   
3c. Emb + MLP head                             1.000           0.612   
4a. LLM nhỏ zero-shot                          0.974           1.000   
4d. LLM nhỏ + prompt phòng thủ                 0.974           1.000   

                                acc 50 case khó  
1. BoW + LogReg                           0.564  
2. BoW + MLP                              0.600  
3a. Emb + kNN                             0.727  
3b. Emb + LogReg                          0.618  
3c. Emb + MLP head                        0.655  
4a. LLM nhỏ zero-shot                     1.000  
4d. LLM nhỏ + prompt phòng thủ            1.000

## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*
Đoán: LLM sẽ bị câu “Ignore all previous instructions…” lừa ở một số case,  và prompt phòng thủ (4d) sẽ tốt hơn rõ so với zero-shot (4a). Ra: cả 4a và 4d đều đạt 100%, kể cả 6 case injection. Vi model hiện đại đã được huấn luyện để chống injection kiểu cơ bản, và các case viết tay là những ví dụ điển hình mà LLM nhận ra dễ dàng. 4a đã đạt 100% nên 4d không còn chỗ để tỏ ra tốt hơn
Đoán: embedding hiểu nghĩa nên sẽ tốt hơn BoW ở mọi nhóm case khó. Ket qua embedding chỉ đúng 50% (2/4) ở nhóm “Nói về spam”, trong khi BoW đúng 100%.. Vì sao lệch: embedding mã hoá chủ đề của câu (giải thưởng, số 0906, lừa đảo) chứ không mã hoá lập trường (đang chê spam hay đang gửi spam). Câu kể về spam có chủ đề rất gần spam nên bị xếp vào vùng spam.
Đoán: MLP sẽ ngang LogReg , vì cả hai dùng cùng feature và tín hiệu spam gần như tuyến tính. Ra: trên SUB, F1 tăng từ 0.857 lên 0.947 (+9 điểm). Trên 50 case khó, F1 tăng từ 0.333 lên 0.476. Vì lệch một phần do nhiễu: SUB cân bằng 20/20, mỗi tin khoảng 2.6 điểm, nên +9 điểm chỉ tương ứng khoảng 3–4 tin.


## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*
BoW + LogReg tụt nhiều nhất: F1 giảm từ 0.857 xuống 0.333 . Accuracy chỉ còn 0.564, chỉ cao hơn đoán ngẫu nhiên một chút. Model tụt nặng nhất ở các nhóm spam giọng tự nhiên (10%), thao túng / lừa đảo (12%) và né từ khoá (17%).\

Case1:  "Your bank account has been temporarily suspended due to unusual activity. Verify your identity within 24 hours: secure-banking-verify.com"
Giả mạo ngân hàng và xài những từ thông thường để qua được hệ thống AI

Case 2: “C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd. Cl4im n0w at w1n-gift.c0m”
Sau khi tokenize, các từ spam mạnh trở thành token nằm ngoài vocab: “c0ngr4tulat10ns”, “w0n”, “cl4im”, “n0w”, “w1n”. Hàm vectorize bỏ qua token không có trong vocab, nên các tín hiệu spam mạnh nhất biến mất khỏi vector. Thứ còn lại chỉ là vài từ trung tính (“a”, “at”, có thể thêm “gift”), không đủ thắng bias âm (do spam chỉ chiếm khoảng 12.6% dữ liệu train). Kết quả là p model đoán ham.
Với con người (và với LLM), “C0ngr4tulat10ns… w0n… Cl4im n0w” vẫn đọc ra là “Congratulations… won… Claim now”. Nhưng với BoW, “w0n” và “won” là hai cột hoàn toàn khác nhau. Embedding hieu duoc nhom nay vì tokenizer subword vẫn nhận ra những từ này gần với từ gốc.


## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*
🎯 Đề #6: LLM · Sửa build_classification_prompt để LLM trả JSON {"label": ..., "reason": ...}, viết parse cho JSON, chạy lại trên 10 tin SUB. So token/chi phí với bản 1 từ. Đáng không?
Cách làm dựa vào code cũ thay thế bằng cú pháp của JSON và đổi cách parse strings đó rồi kiểm lại
Kết quả không chắc vì không chắc cách implementation để chạy


## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*
Phương án mình chọn là Emb (MiniLM) + MLP head đạt yêu cầu độ trễ và chất lượng. Trên test thường, đây là cách có F1 cao nhất trong nhóm đạt SLA, và còn dư 14 ms.
Chi phí Mỗi ngày khoảng 105/ngày, khoảng 3.150/tháng (khoảng $38.000/năm)

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| 5|Hoi ve error khi call api va bi bao eceeding call request |ratelimiting funcion | add ratelimit function va doi variables|
| 5|Prompt function cua cac lenh call api | generate code| check code remove code thua va them vao doi variable|
| 1, 2, 3|Debug loi code o cac function foward vs backward propogration va gradient | Dua code debug va explaination chi tiet| Apply fix va modify code|
| 4 | Viết `llm_classify_batch` (độ trễ, chi phí, số câu không parse được) | Code chạy cả loạt có ước tính token/chi phí | Chạy thử trên 5 tin trước khi chạy SUB |
| 4 | Lỗi 429 vượt quá số request (20 RPM) | Hàm `RateLimiter` + `retry_with_backoff` chờ lâu hơn | Thêm rate limit, đổi biến; tự phát hiện là chưa gọi `limiter.wait()` trong vòng lặp → AI chỉ ra, đã thêm |
| 3 | Viết `top_k_similar` không dùng vòng lặp; có tái sử dụng `cosine_similarity` được không | Code bản vector hoá, xử lý vector toàn 0 | Tự viết `cosine_similarity`; dùng bản chạy vòng lặp để kiểm chứng điểm số của bản nhanh |

AI giúp tăng tốc độ xữ lý và sửa lỗi cũng như giúp mình hiểu rõ hơn về vấn đề đang mắc phải. Viết hàm test và check lỗi và những edge case mình bỏ qua
