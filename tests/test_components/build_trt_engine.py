#!/usr/bin/env python3
"""Build TensorRT engine using Python API to match the runtime version"""
import tensorrt as trt
import sys

print(f"TensorRT version: {trt.__version__}")

# Paths
ONNX_FILE = "/mnt/f/desktop/CV_Project/CV_TRACKING_ADVANCED/ostrack.onnx"
ENGINE_FILE = "/mnt/f/desktop/CV_Project/CV_TRACKING_ADVANCED/ostrack.engine"

# Create builder
logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

print(f"Loading ONNX model: {ONNX_FILE}")
with open(ONNX_FILE, 'rb') as model:
    if not parser.parse(model.read()):
        print('ERROR: Failed to parse ONNX file')
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        sys.exit(1)

print("✓ ONNX model parsed successfully")

# Build engine config
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

# Enable FP16
if builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)
    print("✓ FP16 mode enabled")
else:
    print("⚠ FP16 not supported on this platform")

print("Building TensorRT engine (this may take a few minutes)...")
serialized_engine = builder.build_serialized_network(network, config)

if serialized_engine is None:
    print("ERROR: Failed to build engine")
    sys.exit(1)

print(f"✓ Engine built successfully")
print(f"Saving engine to: {ENGINE_FILE}")

with open(ENGINE_FILE, 'wb') as f:
    f.write(serialized_engine)

print("✅ Done! Engine saved successfully")
