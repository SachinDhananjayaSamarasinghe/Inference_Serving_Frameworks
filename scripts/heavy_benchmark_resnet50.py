import requests
import base64
import json
import time
import threading
import numpy as np
import csv

# URL of Triton HTTP Server
URL = "http://localhost:8000/v2/models/resnet50/infer"

# Number of parallel threads and requests per thread
NUM_THREADS = 10
REQUESTS_PER_THREAD = 50

latencies = []

def send_inference(thread_id):
    for i in range(REQUESTS_PER_THREAD):
        infer_input = {
            "inputs": [
                {
                    "name": "data",
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

        if response.status_code != 200:
            print(f"[Thread {thread_id}] Request {i} failed: {response.text}")
        else:
            latency_ms = (end - start) * 1000  # Convert to milliseconds
            latencies.append(latency_ms)

threads = []
start_time = time.time()

print(f"Launching {NUM_THREADS} threads × {REQUESTS_PER_THREAD} requests each...")

for thread_id in range(NUM_THREADS):
    thread = threading.Thread(target=send_inference, args=(thread_id,))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

end_time = time.time()

# Save latency data
with open("resnet50_latency.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["latency_ms"])
    for latency in latencies:
        writer.writerow([latency])

# Summary stats
latencies_sorted = sorted(latencies)
avg_latency = sum(latencies) / len(latencies)
p95_latency = latencies_sorted[int(len(latencies) * 0.95)]
max_latency = max(latencies)

print(f"Heavy Benchmark Completed!")
print(f"Total Requests: {NUM_THREADS * REQUESTS_PER_THREAD}")
print(f"Total Time Taken: {end_time - start_time:.2f} seconds")
print(f"Throughput: {(NUM_THREADS * REQUESTS_PER_THREAD) / (end_time - start_time):.2f} requests/sec")
print(f"Average Latency: {avg_latency:.2f} ms")
print(f"95th Percentile Latency: {p95_latency:.2f} ms")
print(f"Max Latency: {max_latency:.2f} ms")

