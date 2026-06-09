import torch
from torchvision import datasets, transforms
from PIL import Image

from model import CNN

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

model = CNN().to(device)

model.load_state_dict(
    torch.load("cifar10_cnn.pth", map_location=device)
)

model.eval()

image_Path = "test_image.jpg"

image = Image.open(image_Path).convert("RGB")
image = transform(image)
image = image.unsqueeze(0)
image = image.to(device)

import torch.nn.functional as F

with torch.no_grad():
    output = model(image)
    probabilities = F.softmax(output, dim=1)
    confidence, predicted = torch.max(probabilities, 1)

    print("Prediction:", class_names[predicted.item()])
    print("Confidence:", confidence.item())

    top_probs, top_classes = torch.topk(probabilities, 3)

    print("\nTop 3 Predictions:")

    for i in range(3):

        class_index = top_classes[0][i].item()

        prob = top_probs[0][i].item()

        print(
            f"{i+1}. "
            f"{class_names[class_index]} "
            f"({prob:.4f})"
        )
