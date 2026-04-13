import torch
import gradio as gr
from torchvision import transforms
from PIL import Image
import json

from model import ResNetClassifier
from dataset import load_datasets

# Load ánh xạ từ class_id → tên thật
with open("class_mapping.json", "r", encoding="utf-8") as f:
    id2name = json.load(f)

# Load class names từ tập train
DATASET_ROOT = "./dataset_split"
datasets_dict = load_datasets(DATASET_ROOT)
class_names = datasets_dict["train"].classes
num_classes = len(class_names)

# Load model tốt nhất
model_path = "best_model_cfg1.pth"
model = ResNetClassifier(num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# Transform ảnh giống lúc validation/test
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Hàm dự đoán
def predict(image):
    image = image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).squeeze()

    top_idx = torch.argmax(probs).item()
    class_id = class_names[top_idx]

    # Ghép tên tiếng Anh và tiếng Việt
    name_entry = id2name.get(class_id, {})
    class_name = f"{name_entry.get('en', class_id)} ({name_entry.get('vi', '')})"
    confidence = probs[top_idx].item()

    # Tạo Top-5 label với tên ghép
    top_probs = torch.topk(probs, 5)
    top_labels = {
        f"{id2name.get(class_names[i], {}).get('en', class_names[i])} ({id2name.get(class_names[i], {}).get('vi', '')})": float(probs[i])
        for i in top_probs.indices
    }

    return top_labels, f"{class_name} ({confidence*100:.2f}%)"

# Giao diện Gradio
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Tải ảnh lên"),
    outputs=[
        gr.Label(num_top_classes=5, label="Top-5 Classes"),
        gr.Textbox(label="Dự đoán chính")
    ],
    title="MiniImageNet ResNet18 Demo",
    description="Tải lên một ảnh bất kỳ, model sẽ dự đoán lớp phù hợp trong bộ MiniImageNet (100 lớp)."
)

demo.launch()