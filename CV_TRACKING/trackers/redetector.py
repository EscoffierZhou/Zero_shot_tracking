"""
YOLO11-based re-detection for occlusion recovery
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


class ReDetector:
    """
    Trigger-based re-detection using YOLO11n with CUDA acceleration
    """
    
    def __init__(self, device='cuda', model_path=None):
        """
        Initialize YOLO11 detector
        Args:
            device: 'cuda' or 'cpu'
            model_path: Path to YOLO11 model file (default: models/yolo11n.pt)
        """
        if model_path is None:
            # Use YOLO11n (fastest) from local models directory
            model_path = Path(__file__).parent.parent / 'models' / 'yolo11n.pt'
        
        print(f"🔍 Loading YOLO11 model from: {model_path}")
        self.model = YOLO(str(model_path))
        self.device = device
        
        # Move model to device
        if device == 'cuda':
            self.model.to(device)
        print(f"✓ YOLO11 loaded on {device.upper()}")
        
    def detect(self, frame, search_region=None, conf_threshold=0.25):
        """
        Detect objects in frame
        Args:
            frame: Input frame (BGR)
            search_region: Optional (x, y, w, h) to limit search area
            conf_threshold: Confidence threshold for detections
        Returns:
            List of detections [(x, y, w, h, conf, class_id), ...]
        """
        # If search region specified, crop frame
        if search_region is not None:
            x, y, w, h = [int(v) for v in search_region]
            h_frame, w_frame = frame.shape[:2]
            
            # Validate search region
            if w <= 0 or h <= 0:
                print(f"⚠️ Invalid search region: w={w}, h={h}, using full frame")
                search_frame = frame
                offset = (0, 0)
            else:
                # Expand search region by 50%
                expand_factor = 1.5
                x_center, y_center = x + w//2, y + h//2
                w_expanded = int(max(w * expand_factor, 100))  # Minimum 100px
                h_expanded = int(max(h * expand_factor, 100))  # Minimum 100px
                
                x1 = max(0, x_center - w_expanded//2)
                y1 = max(0, y_center - h_expanded//2)
                x2 = min(w_frame, x_center + w_expanded//2)
                y2 = min(h_frame, y_center + h_expanded//2)
                
                # Ensure valid crop region
                if x2 <= x1 or y2 <= y1:
                    print(f"⚠️ Invalid crop region: ({x1},{y1}) to ({x2},{y2}), using full frame")
                    search_frame = frame
                    offset = (0, 0)
                else:
                    search_frame = frame[y1:y2, x1:x2]
                    offset = (x1, y1)
                    
                    # Final validation - ensure non-zero dimensions
                    if search_frame.shape[0] == 0 or search_frame.shape[1] == 0:
                        print(f"⚠️ Zero-dimension search frame, using full frame")
                        search_frame = frame
                        offset = (0, 0)
        else:
            search_frame = frame
            offset = (0, 0)
        
        # Run YOLO11 detection
        try:
            results = self.model.predict(search_frame, conf=conf_threshold, verbose=False)[0]
        except Exception as e:
            print(f"⚠️ YOLO11 detection error: {e}")
            return []
        
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # Convert to (x, y, w, h) format with offset
            x = int(x1) + offset[0]
            y = int(y1) + offset[1]
            w = int(x2 - x1)
            h = int(y2 - y1)
            
            detections.append((x, y, w, h, conf, cls))
        
        return detections
    
    def find_best_match(self, frame, detections, template, template_hist, prev_bbox):
        """
        Find best matching detection using re-identification
        Args:
            frame: Current frame
            detections: List of detections from YOLO
            template: Initial template image
            template_hist: Initial template histogram
            prev_bbox: Previous bounding box for spatial proximity
        Returns:
            Best matching bbox (x, y, w, h) or None
        """
        if not detections:
            return None
        
        best_score = -1
        best_bbox = None
        
        prev_center = (prev_bbox[0] + prev_bbox[2]//2, prev_bbox[1] + prev_bbox[3]//2)
        
        for det in detections:
            x, y, w, h, conf, cls = det
            
            # Extract current patch
            try:
                patch = frame[y:y+h, x:x+w]
                if patch.size == 0:
                    continue
                
                # Resize for comparison
                patch_resized = cv2.resize(patch, (64, 64))
                template_resized = cv2.resize(template, (64, 64))
                
                # Histogram similarity
                hsv = cv2.cvtColor(patch_resized, cv2.COLOR_BGR2HSV)
                patch_hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
                cv2.normalize(patch_hist, patch_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                hist_sim = cv2.compareHist(template_hist, patch_hist, cv2.HISTCMP_CORREL)
                
                # Template matching
                tmpl_result = cv2.matchTemplate(patch_resized, template_resized, cv2.TM_CCOEFF_NORMED)
                tmpl_score = tmpl_result[0, 0]
                
                # Spatial proximity (normalized distance)
                curr_center = (x + w//2, y + h//2)
                dist = np.sqrt((curr_center[0] - prev_center[0])**2 + (curr_center[1] - prev_center[1])**2)
                max_dist = np.sqrt(frame.shape[0]**2 + frame.shape[1]**2)
                proximity_score = 1.0 - (dist / max_dist)
                
                # Combined score
                score = 0.4 * hist_sim + 0.3 * tmpl_score + 0.2 * proximity_score + 0.1 * conf
                
                if score > best_score:
                    best_score = score
                    best_bbox = (x, y, w, h)
                    
            except Exception as e:
                print(f"Re-ID error for detection: {e}")
                continue
        
        # Return best match if score is above threshold
        if best_score > 0.5:
            return best_bbox
        return None
