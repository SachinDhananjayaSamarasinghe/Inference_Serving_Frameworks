import requests
import base64
import json
import time
import threading
import numpy as np
import statistics

# URL of Triton HTTP Server
URL = "http://localhost:8000/v2/models/mobilenet/infer"

# Load dummy image (used just to simulate input presence)
with open("kitten.jpg", "rb") as f:
    image_bytes = f.read()
image_base64 = base64.b64encode(image_bytes).decode('utf-8')

# Configuration
NUM_THREADS = 10
REQUESTS_PER_THREAD = 50

# Shared latency storage
all_latencies = [[] for _ in range(NUM_THREADS)]

def send_inference(thread_id, latencies):
    for i in range(REQUESTS_PER_THREAD):
        infer_input = {
            "inputs": [
                {
                    "name": "input",
                    "shape": [1, 3, 224, 224],
                    "datatype": "FP32",
                    "data": np.random.rand(1, 3, 224, 224).tolist()
                }
            ]
        }

        headers = {"Content-Type": "application/json"}

        start = time.time()
        response = requests.post(URL, headers=headers, json=infer_input)
        end = time.time()

        latencies.append((end - start) * 1000)  # latency in milliseconds

        if response.status_code != 200:
            print(f"[Thread {thread_id}] Request {i} failed: {response.text}")

        # Optional: simulate low request arrival (used in low-load scenarios)
        # time.sleep(0.2)

threads = []
start_time = time.time()

print(f"Launching {NUM_THREADS} threads × {REQUESTS_PER_THREAD} requests each...")

for thread_id in range(NUM_THREADS):
    thread = threading.Thread(target=send_inference, args=(thread_id, all_latencies[thread_id]))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

end_time = time.time()

# Flatten latency list
flat_latencies = [lat for thread_lat in all_latencies for lat in thread_lat]

# Stats
total_requests = NUM_THREADS * REQUESTS_PER_THREAD
total_time = end_time - start_time
throughput = total_requests / total_time

print("Heavy Benchmark Completed!")
print(f"Total Requests: {total_requests}")
print(f"Total Time Taken: {total_time:.2f} seconds")
print(f"Throughput: {throughput:.2f} requests/sec")
print(f"Average Latency: {np.mean(flat_latencies):.2f} ms")
print(f"95th Percentile Latency: {np.percentile(flat_latencies, 95):.2f} ms")
print(f"Max Latency: {np.max(flat_latencies):.2f} ms")

