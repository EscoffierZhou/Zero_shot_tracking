import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import sys
from pathlib import Path

# Add OSTrack root to path
CURRENT_DIR = Path(__file__).parent
OSTRACK_ROOT = CURRENT_DIR.parent / 'OSTrack-main'
sys.path.append(str(OSTRACK_ROOT))

from lib.test.tracker.basetracker import BaseTracker
from lib.test.utils.hann import hann2d
from lib.utils.box_ops import clip_box
from lib.train.data.processing_utils import sample_target

# Mock env_settings to avoid needing local.py
from lib.test.evaluation.environment import EnvSettings
import lib.test.evaluation.environment

def mock_env_settings():
    env = EnvSettings()
    env.prj_dir = str(OSTRACK_ROOT)
    env.save_dir = str(CURRENT_DIR.parent / 'output')
    env.results_path = str(CURRENT_DIR.parent / 'output' / 'results')
    return env

lib.test.evaluation.environment.env_settings = mock_env_settings

from lib.test.parameter.ostrack import parameters

class OSTrackFerrariTRT(BaseTracker):
    def __init__(self, model_name='vitb_256_mae_ce_32x4_ep300'):
        params = parameters(model_name)
        params.debug = 0
        super(OSTrackFerrariTRT, self).__init__(params)
        
        # Explicitly initialize CUDA - ensure we have a context
        import pycuda.driver as cuda_driver
        cuda_driver.init()
        self.cuda_context = cuda_driver.Device(0).make_context()
        
        # Load TensorRT Engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        # Engine is in the project root (CV_TRACKING_ADVANCED)
        project_root = Path(__file__).parent.parent
        self.engine_path = str(project_root / 'ostrack.engine')
        
        if not os.path.exists(self.engine_path):
            raise FileNotFoundError(f"TRT Engine not found at {self.engine_path}")
            
        print(f"Loading TRT Engine from {self.engine_path}...")
        with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine)
        
        # Parameters
        self.cfg = params.cfg
        self.search_size = self.cfg.TEST.SEARCH_SIZE
        self.template_size = self.cfg.TEST.TEMPLATE_SIZE
        self.feat_sz = self.search_size // 16 # stride 16 for ViT-B
        
        # Hann window (using numpy instead of torch)
        hann_1d = np.hanning(self.feat_sz)
        self.output_window = np.outer(hann_1d, hann_1d).astype(np.float32)
        
        # Normalization constants
        self.mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1)).astype(np.float32)
        self.std = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1)).astype(np.float32)
        
        self.state = None
        self.z_patch_arr = None
        
    def allocate_buffers(self, engine):
        inputs = {}
        outputs = {}
        bindings = []
        stream = cuda.Stream()
        
        # TensorRT 10.x API
        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            shape = engine.get_tensor_shape(tensor_name)
            dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
            
            # Calculate size
            size = trt.volume(shape)
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # Append to bindings list (must match engine binding order)
            bindings.append(int(device_mem))
            
            # Store in dict by name
            mode = engine.get_tensor_mode(tensor_name)
            if mode == trt.TensorIOMode.INPUT:
                inputs[tensor_name] = {'host': host_mem, 'device': device_mem}
            else:  # OUTPUT
                outputs[tensor_name] = {'host': host_mem, 'device': device_mem}
                
        return inputs, outputs, bindings, stream

    def preprocess(self, img_arr):
        # img_arr: (H, W, 3) -> (1, 3, H, W) normalized
        img = img_arr.transpose(2, 0, 1)[None, ...]
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        return np.ascontiguousarray(img)

    def initialize(self, image, info: dict):
        # Get initial bbox
        self.state = info['init_bbox']
        
        # Sample template
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                    output_sz=self.template_size)
        self.z_patch_arr = z_patch_arr
        
        # Preprocess template
        self.template_input = self.preprocess(z_patch_arr)
        
        # Copy template to input buffer
        # Binding name "z" from export_onnx.py
        if 'z' in self.inputs:
            np.copyto(self.inputs['z']['host'], self.template_input.ravel())
        else:
            print("Warning: Binding 'z' not found in TRT engine!")
        
        print("TRT Tracker Initialized")

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        
        # Sample search region
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.search_size)
        
        # Preprocess search
        search_input = self.preprocess(x_patch_arr)
        
        # Copy inputs to host buffers
        if 'z' in self.inputs:
            np.copyto(self.inputs['z']['host'], self.template_input.ravel())
        if 'x' in self.inputs:
            np.copyto(self.inputs['x']['host'], search_input.ravel())
        
        # Transfer inputs to device and set addresses
        for name, inp in self.inputs.items():
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
            self.context.set_tensor_address(name, int(inp['device']))
        
        # Set output addresses
        for name, out in self.outputs.items():
            self.context.set_tensor_address(name, int(out['device']))
            
        # Run inference (TensorRT 10.x uses execute_async_v3)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # Transfer outputs back
        max_score = float(response.max())
        
        return {"target_bbox": self.state, "conf_score": max_score}

    def init(self, frame, bbox):
        """Adapter for app.py"""
        self.initialize(frame, {'init_bbox': list(bbox)})

    def update(self, frame):
        """Adapter for app.py"""
        out = self.track(frame)
        
        # Return bbox, confidence, and status based on score
        conf_score = out.get('conf_score', 0.0)
        
        if conf_score > 0.5:
            status = "TRACKING"
        elif conf_score > 0.3:
            status = "LOW_CONF"
        else:
            status = "LOST"
            
        return out['target_bbox'], conf_score, status

    def __del__(self):
        """Cleanup CUDA context"""
        try:
            if hasattr(self, 'cuda_context'):
                self.cuda_context.pop()
        except:
            pass
