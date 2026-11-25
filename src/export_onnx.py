"""
Step 1: Export YOLOv8 to ONNX format for Real-Time Inference Optimization.
Requirement Match: "Real-Time AI Analytics (Inference Optimization)"
"""
import os
import time
import shutil
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort

def export_yolo_to_onnx(model_name="yolov8n.pt", target_dir="models", imgsz=640):
    os.makedirs(target_dir, exist_ok=True)
    
    print("\n=======================================================")
    print(f"[1/3] Loading PyTorch YOLOv8 Nano model ({model_name})...")
    print("=======================================================")
    model = YOLO(model_name)
    
    print("\n[2/3] Exporting to ONNX format with graph simplification...")
    # opset=12: industry standard ONNX operator set
    # simplify=True: prunes redundant graph nodes using onnxslim
    exported_path = model.export(format="onnx", imgsz=imgsz, simplify=True, opset=12)
    
    target_path = os.path.join(target_dir, "yolov8n.onnx")
    if os.path.abspath(exported_path) != os.path.abspath(target_path):
        shutil.move(exported_path, target_path)
        
    file_size_mb = os.path.getsize(target_path) / (1024 * 1024)
    print(f"\n[3/3] Success! Optimized ONNX model saved to: {target_path}")
    print(f"       File Size: {file_size_mb:.2f} MB")
    
    # Benchmark CPU Inference with ONNX Runtime
    print("\n[BENCHMARK] Testing ONNX Runtime CPU Inference Speed...")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(target_path, opts, providers=["CPUExecutionProvider"])
    
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)
    
    # Warmup
    for _ in range(3):
        _ = session.run(None, {input_name: dummy_input})
        
    # Measure latency
    runs = 10
    start = time.perf_counter()
    for _ in range(runs):
        _ = session.run(None, {input_name: dummy_input})
    latency_ms = ((time.perf_counter() - start) / runs) * 1000
    fps = 1000.0 / latency_ms
    
    print(f"       Average CPU Latency: {latency_ms:.1f} ms per frame")
    print(f"       Estimated Throughput: ~{fps:.1f} FPS")
    print("=======================================================\n")
    return target_path

if __name__ == "__main__":
    export_yolo_to_onnx()
