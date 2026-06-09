import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    10
)

model.load_state_dict(
    torch.load(
        "resnet18_cifar10_finetuned.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image_path = "test_image.jpg"

image = Image.open(image_path).convert("RGB")
image = transform(image)
image = image.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image)
    probabilities = torch.softmax(output, dim=1)

    top_probs, top_classes = torch.topk(probabilities, 3)

print("The 3 Predictions:")

for i in range(3):
    class_index = top_classes[0][i].item()
    prob = top_probs[0][i].item()

    print(
        f"{i+1}, {class_names[class_index]} ({prob:.4f})"
    )