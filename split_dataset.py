import os
import shutil
import random
from tqdm import tqdm

# Cấu hình chia tập
src_dataset = "dataset"
dst_root = "dataset_split"
train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
random.seed(42)

# Tạo thư mục đầu ra
for split in ['train', 'validation', 'test']:
    os.makedirs(os.path.join(dst_root, split), exist_ok=True)

# Duyệt qua từng lớp
class_names = [d for d in os.listdir(src_dataset) if os.path.isdir(os.path.join(src_dataset, d))]

for cls in tqdm(class_names, desc="Đang chia dữ liệu"):
    src_cls_path = os.path.join(src_dataset, cls)
    images = os.listdir(src_cls_path)
    random.shuffle(images)

    n = len(images)
    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)

    splits = {
        "train": images[:train_end],
        "validation": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, imgs in splits.items():
        dst_cls_path = os.path.join(dst_root, split, cls)
        os.makedirs(dst_cls_path, exist_ok=True)
        for img in imgs:
            shutil.copy(os.path.join(src_cls_path, img), os.path.join(dst_cls_path, img))