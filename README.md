# ⚽ Premier League 2024-2025: Player Performance & Value Prediction

Đây là dự án phân tích dữ liệu chuyên sâu về các cầu thủ bóng đá tại giải Ngoại hạng Anh mùa giải 2024-2025. Dự án này thể hiện quy trình làm việc toàn diện, từ thu thập dữ liệu tự động cho đến xây dựng mô hình dự đoán.

## ✨ Điểm nhấn của dự án

- **Thu thập dữ liệu**: Sử dụng **Selenium** và **BeautifulSoup** để tự động cào dữ liệu thống kê chi tiết của cầu thủ từ trang web FBRef. Toàn bộ dữ liệu được làm sạch, xử lý và hợp nhất một cách có hệ thống.
- **Phân tích chuyên sâu**: Thực hiện phân tích thống kê để tìm ra những cầu thủ và đội bóng xuất sắc nhất. Các biểu đồ histogram được tạo ra để trực quan hóa sự phân bố hiệu suất của cầu thủ và cả đội bóng.
- **Phân loại cầu thủ**: Áp dụng thuật toán **K-means** để phân nhóm cầu thủ dựa trên các chỉ số hiệu suất. Sử dụng **PCA** để giảm chiều dữ liệu, giúp việc hình dung các cụm trở nên dễ dàng hơn.
- **Dự đoán giá trị chuyển nhượng**: Xây dựng mô hình học máy để dự đoán giá trị cầu thủ dựa trên hiệu suất thi đấu của họ. Mô hình **Random Forest** được tinh chỉnh và chọn làm giải pháp tối ưu, đạt độ chính xác cao nhất.

## 🛠️ Công nghệ sử dụng

| Công cụ & Thư viện | Mục đích |
| :--- | :--- |
| **Python** | Ngôn ngữ chính của dự án |
| **Pandas**, **NumPy** | Xử lý và phân tích dữ liệu hiệu quả |
| **Selenium**, **BeautifulSoup** | Thu thập dữ liệu từ web |
| **Scikit-learn** | Áp dụng các thuật toán ML |
| **Matplotlib** | Trực quan hóa dữ liệu |
| **RapidFuzz** | Xử lý so khớp tên cầu thủ |
