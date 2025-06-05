import requests
import json
import numpy as np
from PIL import Image
import os
import torchvision.transforms as transforms

# Triton inference URL
URL = "http://localhost:8000/v2/models/resnet50/infer"

# ImageNet class labels
imagenet_labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels_txt = "imagenet_classes.txt"

if not os.path.exists(labels_txt):
    import urllib.request
    urllib.request.urlretrieve(imagenet_labels_url, labels_txt)

with open(labels_txt) as f:
    class_labels = [line.strip() for line in f.readlines()]

# Expected ground-truth labels for the test images
ground_truth = {
    "dog.jpg": "samoyed",
    "kitten.jpg": "british shorthair",
    "panda.jpg": "giant panda",
    "car.jpg": "sedan car",
    "airplane.jpg": "airliner"
}

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

correct = 0
total = 0
image_folder = os.path.join(os.path.dirname(__file__), "test_images")
#image_folder = "test_images"  # Relative to this script's location

print("\nEvaluating ResNet50 Predictions:\n")

for filename in os.listdir(image_folder):
    if not filename.endswith(".jpg"):
        continue

    image_path = os.path.join(image_folder, filename)

    # Skip empty or corrupted files
    if os.path.getsize(image_path) == 0:
        print(f"Skipping empty image: {filename}")
        continue

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Failed to open {filename}: {e}")
        continue

    input_tensor = transform(image).unsqueeze(0).numpy()  # Shape: [1, 3, 224, 224]

    payload = {
        "inputs": [
            {
                "name": "data",
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

    expected_label = ground_truth.get(filename, "unknown")
    match = expected_label.lower() in predicted_label.lower()

    status = "" if match else ""
    print(f"{status} {filename} → Top-1 Prediction: {predicted_label} | Expected: {expected_label}")

    total += 1
    correct += int(match)

# Final Accuracy
if total > 0:
    accuracy = correct / total * 100
    print(f"\n Top-1 Accuracy: {accuracy:.2f}% ({correct}/{total})")
else:
    print("No images evaluated — check dataset and filenames.")

