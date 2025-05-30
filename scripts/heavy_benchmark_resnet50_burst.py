# Burst Load Benchmark Script (ResNet50)
# Simulates bursts with short gaps and staggered thread start

import requests
import json
import time
import threading
import numpy as np
import csv

URL = "http://localhost:8000/v2/models/resnet50/infer"
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

        if response.status_code == 200:
            latencies.append((end - start) * 1000)
        else:
            print(f"[Thread {thread_id}] Request {i} failed: {response.text}")

        time.sleep(0.02)  # small delay to mimic burstiness

threads = []
start_time = time.time()

print(f"Launching {NUM_THREADS} threads × {REQUESTS_PER_THREAD} requests each...")

for t_id in range(NUM_THREADS):
    t = threading.Thread(target=send_inference, args=(t_id,))
    threads.append(t)
    t.start()
    time.sleep(0.01)  # stagger thread start

for t in threads:
    t.join()

end_time = time.time()

# Save latency log
with open("resnet_latency_burst.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["latency_ms"])
    for l in latencies:
        writer.writerow([l])

latencies.sort()
avg = sum(latencies) / len(latencies)
p95 = latencies[int(len(latencies) * 0.95)]
max_latency = max(latencies)

print(" Burst Load Benchmark Completed!")
print(f"Total Requests: {NUM_THREADS * REQUESTS_PER_THREAD}")
print(f"Duration: {end_time - start_time:.2f} sec")
print(f"Throughput: {(NUM_THREADS * REQUESTS_PER_THREAD)/(end_time - start_time):.2f} req/sec")
print(f"Avg Latency: {avg:.2f} ms | P95: {p95:.2f} ms | Max: {max_latency:.2f} ms")

