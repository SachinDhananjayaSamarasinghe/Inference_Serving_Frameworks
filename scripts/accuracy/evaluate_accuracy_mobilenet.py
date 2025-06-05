import requests
import json
import numpy as np
from PIL import Image
import os
import torchvision.transforms as transforms

# Triton inference URL
URL = "http://localhost:8000/v2/models/mobilenet/infer"

# Load ImageNet class labels
labels_txt = "imagenet_classes.txt"
with open(labels_txt) as f:
    class_labels = [line.strip() for line in f.readlines()]

# Ground-truth labels for test images
ground_truth = {
    "dog.jpg": "samoyed",
    "kitten.jpg": "british shorthair",
    "panda.jpg": "giant panda",
    "car.jpg": "sedan car",
    "airplane.jpg": "airliner"
}

# Image preprocessing for MobileNet
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

print("\n Evaluating MobileNet Predictions:\n")

correct = 0
total = 0
image_folder = "/home/ubuntu/triton_models/mobilenet/test_images"

for filename in os.listdir(image_folder):
    if not filename.endswith(".jpg"):
        continue

    if os.path.getsize(os.path.join(image_folder, filename)) == 0:
        print(f" Skipping empty image: {filename}")
        continue

    image_path = os.path.join(image_folder, filename)
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).numpy()

    payload = {
        "inputs": [
            {
                "name": "input",  # This is specific to MobileNet model config
                "shape": list(input_tensor.shape),
                "datatype": "FP32",
                "data": input_tensor.tolist()
            }
        ]
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(URL, headers=headers, json=payload)

    if response.status_code != 200:
        print(f" Inference failed for {filename}: {response.text}")
        continue

    output = response.json()["outputs"][0]["data"]
    top1_index = np.argmax(output)
    predicted_label = class_labels[top1_index]

    expected = ground_truth.get(filename, None)
    if expected is None:
        print(f"o ground truth label found for {filename}")
        continue

    match = expected.lower() in predicted_label.lower()
    status = ""if match else""
    print(f"{status} {filename} → Top-1 Prediction: {predicted_label} | Expected: {expected}")

    total += 1
    correct += int(match)

# Final accuracy
if total > 0:
    accuracy = correct / total * 100
    print(f"\n Top-1 Accuracy: {accuracy:.2f}% ({correct}/{total})")
else:
    print("\n No images evaluated — check dataset and filenames.")

