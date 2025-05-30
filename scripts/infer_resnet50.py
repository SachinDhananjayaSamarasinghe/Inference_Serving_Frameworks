import numpy as np
from PIL import Image
import requests
import json

# Load and preprocess image
img = Image.open("input.jpg").resize((224, 224)).convert("RGB")
img_np = np.array(img).astype(np.float32) / 255.0
img_np = img_np.transpose(2, 0, 1)  # HWC → CHW
img_np = np.expand_dims(img_np, axis=0)  # Add batch dim

# Prepare request payload
payload = {
    "inputs": [
        {
            "name": "data",
            "shape": [1, 3, 224, 224],
            "datatype": "FP32",
            "data": img_np.flatten().tolist()
        }
    ]
}

# Send inference request
response = requests.post("http://localhost:8000/v2/models/resnet50/infer",
                         headers={"Content-Type": "application/json"},
                         data=json.dumps(payload))

# Print top-5 class probabilities
output = response.json()["outputs"][0]["data"]
top5 = np.argsort(output)[::-1][:5]
print("Top 5 predicted class indices:", top5)

