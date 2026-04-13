# Phân loại ảnh ImageNet-100

Dự án này huấn luyện mô hình phân loại ảnh cho 100 lớp của bộ dữ liệu MiniImageNet bằng `PyTorch` và `ResNet18`. Mã nguồn gồm các bước chuẩn bị dữ liệu, chia dữ liệu thành ba phần `train`, `validation`, `test`, huấn luyện với nhiều cấu hình khác nhau, lưu mô hình tốt nhất theo từng cấu hình và chạy giao diện dự đoán bằng `Gradio`.

## Bộ dữ liệu

- Bộ dữ liệu sử dụng: [MiniImageNet trên Kaggle](https://www.kaggle.com/datasets/arjunashok33/miniimagenet?select=n01930112)
- Sau khi tải về và giải nén, đặt dữ liệu gốc vào thư mục `dataset/`.
- Mỗi lớp là một thư mục con theo mã lớp của ImageNet, ví dụ:

```text
dataset/
|-- n01532829/
|-- n01558993/
|-- n01704323/
`-- ...
```

Dự án hiện làm việc với 100 lớp và dùng tệp `split_dataset.py` để chia dữ liệu theo tỉ lệ:

- `70%` cho `train`
- `15%` cho `validation`
- `15%` cho `test`

## Mô hình và quy trình xử lý

- Mô hình nền: `ResNet18` có trọng số khởi tạo sẵn từ `torchvision`
- Tinh chỉnh có chọn lọc: cố định trọng số của `layer1` và `layer2`
- Đầu phân loại: `Dropout(0.6)` kết hợp với lớp `Linear`
- Kích thước ảnh đầu vào: `224 x 224`
- Các phép biến đổi cho tập `train`: `RandomResizedCrop`, `RandomHorizontalFlip`, `RandomErasing`
- Hàm mất mát: `CrossEntropyLoss(label_smoothing=0.2)`
- Bộ tối ưu: `AdamW`
- Dừng sớm: cấu hình bằng `PATIENCE` trong `config.py`

## Cấu trúc dự án

```text
.
|-- config.py
|-- count_mini_imagenet.py
|-- dataset.py
|-- demo.py
|-- main.py
|-- model.py
|-- split_dataset.py
|-- train_eval.py
|-- class_mapping.json
`-- class_names.json
```

Ý nghĩa các tệp chính:

- `config.py`: chứa các cấu hình huấn luyện và đường dẫn dữ liệu
- `dataset.py`: nạp dữ liệu và định nghĩa các phép biến đổi ảnh
- `split_dataset.py`: chia dữ liệu gốc thành `train`, `validation`, `test`
- `model.py`: định nghĩa mô hình `ResNetClassifier`
- `train_eval.py`: chứa các hàm huấn luyện, đánh giá và kiểm thử
- `main.py`: chạy toàn bộ quá trình huấn luyện
- `demo.py`: mở giao diện dự đoán ảnh bằng `Gradio`
- `count_mini_imagenet.py`: đếm số ảnh trong từng phần dữ liệu đã chia

## Cài đặt

Nếu cần, hãy tạo môi trường ảo trước. Sau đó cài các thư viện chính:

```bash
pip install torch torchvision gradio pillow tqdm
```

## Chuẩn bị dữ liệu

1. Tải bộ dữ liệu từ Kaggle.
2. Giải nén vào thư mục `dataset/`.
3. Chạy lệnh:

```bash
python split_dataset.py
```

Sau bước này, dự án sẽ tạo thư mục `dataset_split/` với cấu trúc:

```text
dataset_split/
|-- train/
|-- validation/
`-- test/
```

## Huấn luyện mô hình

Chạy lệnh:

```bash
python main.py
```

Chương trình sẽ:

- nạp dữ liệu từ `dataset_split/`
- thử các cấu hình đã khai báo trong `config.py`
- chạy mỗi cấu hình `3` lần
- theo dõi độ chính xác trên tập `validation`
- lưu mô hình tốt nhất của từng cấu hình thành các tệp như `best_model_cfg1.pth`, `best_model_cfg2.pth`, ...
- in ra cấu hình tốt nhất sau khi huấn luyện xong

## Chạy giao diện dự đoán

Sau khi đã có tệp mô hình, chạy:

```bash
python demo.py
```

Giao diện sẽ cho phép tải ảnh lên và trả về:

- 5 nhãn có xác suất cao nhất
- nhãn dự đoán chính cùng độ tin cậy
- tên lớp bằng tiếng Anh và tiếng Việt lấy từ `class_mapping.json`

Hiện tại `demo.py` dùng mặc định tệp `best_model_cfg1.pth`. Nếu muốn dùng tệp khác, hãy sửa biến `model_path` trong `demo.py`.