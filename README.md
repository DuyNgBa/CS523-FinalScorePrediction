# CS523 - Dự đoán điểm cuối kì môn học

## Giới thiệu:

Project này xây dựng chương trình dự đoán điểm cuối kì của sinh viên dựa trên điểm giữa kì của môn học

Dataset được sử dụng là file Excel 'TRAIN2.xlsx', trong đó bao gồm dữ liệu điểm giữa kì và điểm cuối kì
Mô hình được sử dụng là **Linear Regression (Hồi quy tuyến tính)**, có dạng công thức:

```math
\hat{y} = \beta_0 + \beta_1x

Trong đó:
* x: điểm giữa kỳ
* ŷ: điểm cuối kỳ dự đoán
* β0: hệ số chặn
* β1: hệ số của điểm giữa kỳ

Mô hình dự đoán: 
Project sử dụng mô hình hồi quy tuyến tính:

final_pred=β0 + β1 × midterm

Các bước xử lý chính:

* Đọc dữ liệu từ file Excel.
* Làm sạch dữ liệu.
* Chọn cột midterm làm biến đầu vào.
* Chọn cột final làm biến cần dự đoán.
* Chia dữ liệu thành tập train và test.
* Huấn luyện mô hình Linear Regression.
* Đánh giá mô hình bằng các chỉ số:
    MAE
    RMSE
    R²
* Vẽ đồ thị hồi quy và đồ thị so sánh kết quả thực tế với dự đoán.
* Xây dựng demo bằng Streamlit.