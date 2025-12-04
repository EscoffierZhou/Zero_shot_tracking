#!/usr/bin/env python3
"""Simple test to check TRT engine loading"""
import sys
import os
sys.path.append('/mnt/f/desktop/CV_Project/CV_TRACKING_ADVANCED')
os.chdir('/mnt/f/desktop/CV_Project/CV_TRACKING_ADVANCED')

import tensorrt as trt
print(f"TensorRT version: {trt.__version__}")

# Load engine
engine_path = "ostrack.engine"
logger = trt.Logger(trt.Logger.WARNING)

print(f"\nLoading engine from: {engine_path}")
with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

if engine is None:
    print("❌ Failed to load engine!")
    sys.exit(1)

print("✓ Engine loaded successfully!")
print(f"Number of bindings: {engine.num_bindings}")

for i in range(engine.num_bindings):
    name = engine.get_binding_name(i)
    shape = engine.get_binding_shape(i)
    dtype = engine.get_binding_dtype(i)
    is_input = engine.binding_is_input(i)
    print(f"  [{i}] {name}: shape={shape}, dtype={dtype}, is_input={is_input}")

print("\n✅ Engine verification successful!")
