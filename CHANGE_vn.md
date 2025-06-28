# Cách chạy code

- cd thư mục src
- "python train.py <market_num> <gMLP_num_layers>" (vd: python train.py 16 3)
- Hoặc "chmod +x param_search.sh"  => "./param_search.sh" để duyệt qua các tham số

## Chi tiết code và tham số

- Mặc định đang chạy thị trường SP500, thay đổi file train.py để chạy thị trường NASDAQ => market_name = "NASDAQ",
  stock_num = 1026, valid_index = 756, test_index = 1008
- Kết quả tốt nhất cho thị trường SP500 => python train.py 16 3
- Kết quả tốt nhất cho thị trường NASDAQ => python train.py 25 1

## Files

- file preprocess.py => tiền xử lý tính đặc trưng market context cho từng window (16 ngày) và chuẩn hóa
- file gMLP => Cơ chế MLP gating học chọn lọc đặc trưng của thị trường

## Đặc trưng "market context"

### Chuyển thành return ratio, tính 5 đặc trưng:

- Window Mean Return => Xu hướng tăng giá vs giảm giá chung trong cửa sổ
- Momentum slope => Hướng và độ mạnh của xu hướng thị trường
- Realized Volatility => Rủi ro trung bình trong cửa sổ
- Dispersion => Các cổ phiếu đang di chuyển cùng nhau hay tách biệt
- chạy PCA trên ma trận hiệp phương sai  (16, N) để lấy phần của tổng phương sai được giải thích bởi principal component
  đầu tiên. => Bao nhiêu phần trăm tổng phương sai được giải thích bởi một yếu tố thị trường
  ≈1.0 → dữ liệu gần như đồng bộ, < 1.0 → dữ liệu high-dimensional (nhiều yếu tố độc lập so với nhau)

### Chuẩn hóa kiểu min-max

## MLP gating (file gMLP)

- Nhánh trunk activate bằng hàm Hardwish
- Đặc trưng market context qua một lớp MLP và cộng vào nhánh gate.
- Nhánh gate cũng activate bằng hàm Hardwish.
