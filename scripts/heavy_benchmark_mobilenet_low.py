import requests
import base64
import json
import time
import threading
import numpy as np
import statistics
import os

# Fix: Determine script directory and construct path to kitten.jpg
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(SCRIPT_DIR, "kitten.jpg")

URL = "http://localhost:8000/v2/models/mobilenet/infer"

with open(image_path, "rb") as f:
    image_bytes = f.read()
image_base64 = base64.b64encode(image_bytes).decode('utf-8')

NUM_THREADS = 10
REQUESTS_PER_THREAD = 50

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
        latencies.append((end - start) * 1000)

        if response.status_code != 200:
            print(f"[Thread {thread_id}] Request {i} failed: {response.text}")

        time.sleep(0.2)  # simulate low frequency load

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

flat_latencies = [lat for thread_lat in all_latencies for lat in thread_lat]
total_requests = NUM_THREADS * REQUESTS_PER_THREAD
total_time = end_time - start_time
throughput = total_requests / total_time

print(" Low Load Benchmark Completed!")
print(f"Total Requests: {total_requests}")
print(f"Duration: {total_time:.2f} sec")
print(f"Throughput: {throughput:.2f} req/sec")
print(f"Avg Latency: {np.mean(flat_latencies):.2f} ms | P95: {np.percentile(flat_latencies, 95):.2f} ms | Max: {np.max(flat_latencies):.2f} ms")

