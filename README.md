# DGA Detection Engine (Enterprise V-Master Edition)

An advanced, production-scale AI ecosystem designed to detect and classify Domain Generation Algorithms (DGA), Combosquatting, Homoglyphs (Typosquatting), and Dictionary-based phishing domains using Deep Learning and Explainable AI (XAI).

---

## 1. Công nghệ và Thư viện sử dụng (Technology Stack)

Dự án được xây dựng dựa trên các công nghệ tiên tiến nhất phù hợp cho hệ thống ML Ops 2025/2026:
- **Ngôn ngữ lập trình:** Python (Core AI/Backend), JavaScript/HTML/CSS (Frontend). 
- **Thư viện Deep Learning:** `TensorFlow` & `Keras` (Xây dựng, huấn luyện mô hình thần kinh).
- **Tối ưu hóa (Hyperparameter Tuning):** `keras-tuner` (Quét tự động learning rate và dropout).
- **Xử lý Dữ liệu & Oversampling:** `pandas`, `numpy`, `scikit-learn` (chia tách train/test, đánh giá metrics), `imblearn` (thuật toán SMOTE cân bằng dữ liệu).
- **Explainable AI (Trí tuệ nhân tạo có thể giải thích):** `lime` (LIME Text Explainer cấp độ ký tự).
- **Web Service & Phân phối (API):** `FastAPI` và `Uvicorn` (High-performance Async REST API).

## 2. Kỹ năng yêu cầu (Skills Required)
Để phát triển và duy trì dự án, các kỹ năng sau là thiết yếu:
- **Kỹ năng AI/Deep Learning:** Hiểu biết sâu về kiến trúc mạng Nơ-ron (Convolutional Neural Networks, Recurrent Neural Networks), kỹ thuật trích xuất đặc trưng văn bản, và cách chống hiện tượng Class Collapse/Vanish Gradient (Focal Loss).
- **Kỹ năng An toàn thông tin (Cybersecurity):** Nắm vững bản chất mã độc, cấu trúc TLDs (Top-Level Domains), và các mánh khóe phishing như Homoglyphs (`g00gle.com`), Combosquatting (`paypal-secure-login.com`).
- **Kỹ năng ML Ops & API:** Triển khai mô hình Machine Learning thành các Web Service có độ trễ siêu thấp sử dụng FastAPI và tích hợp giao diện (Frontend integration).

## 3. Nguyên lý của các thuật toán chủ đạo
Hệ thống KHÔNG dùng Machine Learning truyền thống (như Random Forest hay SVM) tạo feature thủ công (entropy, độ dài). Nó trực tiếp "đọc" domain thông qua kiến trúc Hybrid Deep Learning cấp độ ký tự (Character-Level):

*   **1D-CNN (1-Dimensional Convolutional Neural Network):** Đóng vai trò làm bộ "quét" (scanner). CNN di chuyển dọc theo chuỗi ký tự của domain để trích xuất các đặc trưng cục bộ (Local Features), ví dụ như các n-grams dính liền bất thường hoặc các đoạn nối từ vựng (ví dụ: `pay`, `pal`).
*   **Bi-LSTM (Bidirectional Long Short-Term Memory):** Lớp mạng hồi quy mạnh mẽ. "Bidirectional" cho phép mạng đọc domain theo cả 2 chiều: từ trái sang phải, và từ phải sang trái. Nhờ đó, nó hiểu được *ngữ cảnh không gian thời gian*, biết được vị trí của dấu chấm (`.`) hay TLD (`.com`) đang đứng trước hay đứng sau các subdomains.
*   **Custom Attention Mechanism:** Lớp thuật toán học tự động gán trọng số (weights). Nó bắt chước ánh nhìn của mắt người, giúp mô hình "tập trung" cao độ vào các khu vực lạ của chuỗi ký tự (đặc biệt là TLD hoặc các điểm nối Combosquatting) và bỏ qua độ dài vô hại của các Legit CDNs.
*   **Binary Focal Crossentropy (Hàm mất mát):** Cốt lõi ép mô hình "trưởng thành". Bằng việc đặt `gamma=2.0` và `alpha=0.25`, thuật toán sẽ trừng phạt mô hình cực kỳ nặng nếu nó tự mãn đoán sai những ca mạo danh tinh vi (Homoglyphs/Typosquatting), thay vì chỉ học thuộc các mẫu DGA phèn (dễ đoán).

## 4. Các Nguyên Tắc Tuân Thủ Dự Án
Project này được áp dụng vòng đời Machine Learning (ML Ops) tiêu chuẩn vàng:
*   **Thiết kế kiến trúc rành mạch:** Mô hình tuần tự định nghĩa rõ số block (Embedding -> CNN -> LSTM -> Attention -> Dense), hàm kích hoạt `ReLU` ở lớp ẩn và `Sigmoid` ở node quyết định dự đoán cuối, cùng hàm mất mát Focal Loss.
*   **Chia tách dữ liệu (Training Split):** Dữ liệu được hệ thống tự động băm theo nguyên tắc **7:2:1** tương ứng cho Tập Huấn luyện (Train), Tập Đánh giá (Validation để Tuning), và Tập Kiểm thử mù (Test Set mù hoàn toàn để đo lựờng).
*   **Tối ưu hóa siêu tham số (Optimization):** Hệ thống tích hợp `KerasTuner RandomSearch` quét không gian `learning_rate` và `dropout_rate` thay vì phán đoán cảm tính bằng tay.
*   **Chỉ số đánh giá sơ bộ chuyên nghiệp (Evaluation):** Không phụ thuộc mù quáng vào Độ chính xác (Accuracy). Cung cấp Full Classification Report (F1-score, Precision, Recall), Confusion Matrix, và dựng sơ đồ phổ ROC/AUC để đánh giá khả năng phân loại.

---

## 🚀 Hướng Dẫn Sử Dụng (Quick Start)

### 1. Huấn luyện trí tuệ nhân tạo (Full Retrain)
Chỉ một dòng lệnh để kích hoạt toàn bộ V-Master Pipeline (Data Augmentation, KerasTuner, 7:2:1 Validation, Evaluate):
```bash
python main.py
```
*(Sau khi chạy xong, ảnh ROC/Confusion Matrix và file `best_dga_model.h5` sẽ xuất hiện ở folder `/models/`)*

### 2. Khởi động Giao diện Cảnh báo 3 Cấp (V2 Interactive UI)
Mở Backend API lên trước:
```bash
uvicorn api.app:app --reload
```
Sau đó, chỉ cần mở trực tiếp file `frontend/index.html` bằng trình duyệt web. 
* Giao diện tích hợp Cảnh báo 3 lớp: Xanh (An Toàn) -> Vàng (Đáng Ngờ / Suspicious) -> Đỏ (Nguy Hiểm / DGA).
* Giao diện tích hợp công nghệ LIME (Giải thích cấp độ hạt ký tự) để xuất ra đoạn văn bản lý luận tư duy giải thích tại sao AI lại chốt nhãn đó!
