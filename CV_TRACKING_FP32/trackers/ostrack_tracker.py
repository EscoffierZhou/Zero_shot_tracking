import sys
import os
import cv2
import torch
import numpy as np
from pathlib import Path

# Add OSTrack root to path
CURRENT_DIR = Path(__file__).parent
OSTRACK_ROOT = CURRENT_DIR.parent / 'OSTrack-main'
sys.path.append(str(OSTRACK_ROOT))

# Mock env_settings to avoid needing local.py
from lib.test.evaluation.environment import EnvSettings
from lib.test.tracker.ostrack import OSTrack
from lib.test.parameter.ostrack import parameters
import lib.test.evaluation.environment

def mock_env_settings():
    env = EnvSettings()
    env.prj_dir = str(OSTRACK_ROOT)
    env.save_dir = str(CURRENT_DIR.parent / 'output')
    env.results_path = str(CURRENT_DIR.parent / 'output' / 'results')
    return env

lib.test.evaluation.environment.env_settings = mock_env_settings

class OSTrackFerrari(OSTrack):
    """
    OSTrack Ferrari Edition:
    - High Performance (OSTrack ViT)
    - Global Search Recovery
    - Adaptive Logic
    """
    def __init__(self, model_name='vitb_256_mae_ce_32x4_ep300'):
        # Load parameters
        # Check if we should use the non-CE version based on available weights
        weight_path = CURRENT_DIR.parent / 'model' / 'vitb_256_mae_32x4_ep300' / 'OSTrack_ep0300.pth.tar'
        if weight_path.exists():
            print(f"🔍 Found weights at: {weight_path}")
            # If the folder is named without 'ce', maybe we should use the non-ce config?
            # But let's stick to the requested model_name if possible, or auto-switch.
            # For now, we just set the checkpoint.
            pass
        else:
             # Try the CE folder if it exists
             weight_path_ce = CURRENT_DIR.parent / 'model' / 'vitb_256_mae_ce_32x4_ep300' / 'OSTrack_ep0300.pth.tar'
             if weight_path_ce.exists():
                 weight_path = weight_path_ce
        
        params = parameters(model_name)
        
        # FIX: Manually set checkpoint path
        if weight_path.exists():
            params.checkpoint = str(weight_path)
        else:
            print(f"⚠️ Weights not found at {weight_path}")
            print(f"   Using default path from config: {params.checkpoint}")
            
        # FIX: Missing debug attribute
        params.debug = 0
            
        super(OSTrackFerrari, self).__init__(params, "otb")
        
        self.init_state = None # Store initial state for global search reference
        
    def init(self, frame, bbox):
        """Adapter for app.py interface"""
        # Ensure bbox is a list [x, y, w, h]
        bbox_list = list(bbox)
        self.initialize(frame, {'init_bbox': bbox_list})

    def initialize(self, image, info: dict):
        # Call parent initialize
        out = super().initialize(image, info)
        self.init_state = info['init_bbox'] # [x, y, w, h]
        return out

    def update(self, frame):
        """
        Custom update logic:
        - TRACKING (>0.5): Normal
        - LOW_CONF (0.3-0.5): Normal (OSTrack handles it)
        - LOST (<0.3): Global Search
        """
        # Run tracking
        # We need the score. Parent track() doesn't return it easily.
        # We need to hook into track() or modify it. 
        # Since we inherited, we can copy-paste track() logic or access internal vars if available.
        # OSTrack.track() calculates `pred_score_map` and `max_score` but doesn't return max_score.
        # Let's override track() slightly or just use the internal network call logic here?
        # Overriding is safer to keep consistent state.
        
        # But wait, rewriting track() is complex. 
        # Let's inspect OSTrack.track again. It doesn't save score to self.
        # It returns {'target_bbox': self.state}.
        
        # Hack: We can modify the parent class method at runtime or just copy it here.
        # Copying is safer for stability.
        
        H, W, _ = frame.shape
        self.frame_id += 1
        
        # 1. Sample target
        x_patch_arr, resize_factor, x_amask_arr = self.sample_target(frame, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_dict1.tensors, search=x_dict.tensors, ce_template_mask=self.box_mask_z)

        # 2. Get result
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        
        # Calculate max score for confidence
        # response is (B, 1, H, W)
        max_score = response.max().item()
        
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        
        # get the final box result
        from lib.utils.box_ops import clip_box
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        
        # 3. Logic based on score
        status = "TRACKING"
        if max_score > 0.5:
            status = "TRACKING"
        elif max_score > 0.3:
            status = "LOW_CONF"
        else:
            status = "LOST"
            # Trigger Global Search
            print(f"📉 Score {max_score:.3f} < 0.3. Triggering Global Search...")
            best_bbox, best_score = self.global_search(frame)
            if best_score > 0.4: # Threshold to accept global search result
                self.state = best_bbox
                max_score = best_score
                status = "RECOVERED"
                print(f"✅ Global Search successful! Score: {best_score:.3f}")
            else:
                print(f"❌ Global Search failed. Best score: {best_score:.3f}")
        
        return self.state, max_score, status

    def global_search(self, frame):
        """
        Sliding window global search
        """
        H, W, _ = frame.shape
        w_target, h_target = self.init_state[2], self.init_state[3]
        
        # Grid settings
        stride = int(min(W, H) / 4) # 4x4 grid roughly
        if stride < 50: stride = 50
        
        best_score = -1
        best_bbox = None
        
        # Save current state
        original_state = self.state
        
        for y in range(0, H - h_target, stride):
            for x in range(0, W - w_target, stride):
                # Set state to this window
                self.state = [x, y, w_target, h_target]
                
                # Run inference (simplified, just forward pass)
                # We duplicate logic to avoid side effects of full update
                try:
                    x_patch_arr, resize_factor, x_amask_arr = self.sample_target(frame, self.state, self.params.search_factor,
                                                                            output_sz=self.params.search_size)
                    search = self.preprocessor.process(x_patch_arr, x_amask_arr)
                    with torch.no_grad():
                        out_dict = self.network.forward(
                            template=self.z_dict1.tensors, search=search.tensors, ce_template_mask=self.box_mask_z)
                    
                    pred_score_map = out_dict['score_map']
                    response = self.output_window * pred_score_map
                    score = response.max().item()
                    
                    if score > best_score:
                        best_score = score
                        # Calculate bbox
                        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
                        pred_boxes = pred_boxes.view(-1, 4)
                        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()
                        best_bbox = self.map_box_back(pred_box, resize_factor)
                        
                except Exception as e:
                    continue
        
        # Restore state if failed
        if best_bbox is None:
            self.state = original_state
            return original_state, 0.0
            
        return best_bbox, best_score

    # Helper to access sample_target which is imported in ostrack.py but not a method
    def sample_target(self, image, bbox, search_factor, output_sz):
        from lib.train.data.processing_utils import sample_target
        return sample_target(image, bbox, search_factor, output_sz)

