import os

def count_images_in_dataset(root_dir, extensions={".jpg", ".jpeg", ".png"}):
    total_images = 0
    class_image_counts = {}

    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        count = 0
        for file in os.listdir(class_path):
            if os.path.splitext(file)[1].lower() in extensions:
                count += 1
        class_image_counts[class_name] = count
        total_images += count

    return total_images, class_image_counts

# Thư mục chứa dataset đã chia
base_path = r"D:/1_Documents/Vscode/Python/DeepLearning/dataset_split"

# Đếm ảnh cho từng phần
for split in ['train', 'validation', 'test']:
    print(f"\nPhần: {split}")
    split_path = os.path.join(base_path, split)
    total, per_class = count_images_in_dataset(split_path)
    print(f"Tổng số ảnh: {total}")
    print("Số ảnh theo lớp:")
    for cls, count in sorted(per_class.items()):
        print(f"{cls}: {count}")