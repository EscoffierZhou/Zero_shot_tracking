import sys
print(f"Python: {sys.executable}")

try:
    import onnx
    print(f"✅ onnx: {onnx.__version__}")
except ImportError:
    print("❌ onnx not installed")

try:
    import onnxsim
    print(f"✅ onnxsim: {onnxsim.__version__}")
except ImportError:
    print("❌ onnxsim not installed")

try:
    import tensorrt
    print(f"✅ tensorrt: {tensorrt.__version__}")
except ImportError:
    print("❌ tensorrt not installed")
