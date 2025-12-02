"""
Hybrid Tracker combining CSRT, Kalman Filter, and YOLOv8 re-detection
"""
import cv2
import numpy as np
from .csrt_tracker import CSRTTracker
from .kalman_filter import KalmanTracker
from .redetector import ReDetector


class HybridTracker:
    """
    Multi-layer tracking system:
    Layer 1: CSRT (primary tracking)
    Layer 2: Kalman Filter (motion prediction & smoothing)
    Layer 3: Template Matching (quick recovery)
    Layer 4: YOLO11 Re-detection (occlusion recovery)
    """
    
    def __init__(self, device='cuda', 
                 conf_low=0.3, conf_high=0.7,
                 redetect_threshold=10):
        """
        Args:
            device: 'cuda' or 'cpu'
            conf_low: Low confidence threshold (trigger template matching)
            conf_high: High confidence threshold (normal tracking)
            redetect_threshold: Frames of low confidence before re-detection
        """
        self.csrt = CSRTTracker()
        self.kalman = KalmanTracker()
        self.redetector = ReDetector(device=device)
        
        self.conf_low = conf_low
        self.conf_high = conf_high
        self.redetect_threshold = redetect_threshold
        
        self.low_conf_count = 0
        self.frame_count = 0
        self.initialized = False
        
        # Template for re-ID
        self.template = None
        self.template_hist = None
        
    def init(self, frame, bbox):
        """
        Initialize all tracking components
        Args:
            frame: First frame (BGR)
            bbox: Initial bounding box (x, y, w, h)
        """
        self.csrt.init(frame, bbox)
        self.kalman.init(bbox)
        self.template, self.template_hist = self.csrt.get_template()
        self.initialized = True
        self.frame_count = 0
        self.low_conf_count = 0
        
        print(f"✓ Tracker initialized at bbox: {bbox}")
    
    def update(self, frame):
        """
        Update tracker with new frame (adaptive multi-layer strategy)
        Args:
            frame: Current frame (BGR)
        Returns:
            bbox: Tracked bounding box (x, y, w, h)
            confidence: Tracking confidence [0, 1]
            status: Tracking status string
        """
        if not self.initialized:
            return None, 0.0, "NOT_INITIALIZED"
        
        self.frame_count += 1
        
        # Step 1: CSRT tracking
        success, csrt_bbox, confidence = self.csrt.update(frame)
        
        # Step 2: Kalman prediction
        kalman_pred = self.kalman.predict()
        
        # Adaptive strategy based on confidence
        if confidence > self.conf_high:
            # High confidence: use CSRT directly
            final_bbox = csrt_bbox
            self.kalman.update(csrt_bbox)
            self.low_conf_count = 0
            status = "TRACKING"
            
        elif confidence > self.conf_low:
            # Medium confidence: blend CSRT with Kalman
            if kalman_pred is not None:
                final_bbox = self._blend_bbox(csrt_bbox, kalman_pred, alpha=0.7)
            else:
                final_bbox = csrt_bbox
            self.kalman.update(final_bbox)
            self.low_conf_count += 1
            status = "TRACKING_KALMAN"
            
        else:
            # Low confidence: increment counter
            self.low_conf_count += 1
            
            if self.low_conf_count < self.redetect_threshold:
                # Use Kalman prediction
                if kalman_pred is not None:
                    final_bbox = kalman_pred
                    status = "PREDICTING"
                else:
                    final_bbox = csrt_bbox
                    status = "LOW_CONFIDENCE"
            else:
                # Trigger re-detection
                print(f"⚠ Low confidence for {self.low_conf_count} frames, triggering re-detection...")
                final_bbox = self._redetect(frame, kalman_pred or csrt_bbox)
                
                if final_bbox is not None:
                    # Successfully re-detected, reinitialize tracker
                    self.csrt.reinit(frame, final_bbox)
                    self.kalman.init(final_bbox)
                    self.low_conf_count = 0
                    confidence = 0.8
                    status = "REDETECTED"
                    print(f"✓ Re-detection successful at {final_bbox}")
                else:
                    # Re-detection failed, keep predicting
                    final_bbox = kalman_pred or csrt_bbox
                    confidence = 0.1
                    status = "LOST"
                    print("✗ Re-detection failed, using prediction")
        
        return final_bbox, confidence, status
    
    def _blend_bbox(self, bbox1, bbox2, alpha=0.7):
        """
        Blend two bounding boxes
        Args:
            bbox1: First bbox (x, y, w, h)
            bbox2: Second bbox (x, y, w, h)
            alpha: Weight for bbox1 (1-alpha for bbox2)
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        x = int(alpha * x1 + (1-alpha) * x2)
        y = int(alpha * y1 + (1-alpha) * y2)
        w = int(alpha * w1 + (1-alpha) * w2)
        h = int(alpha * h1 + (1-alpha) * h2)
        
        return (x, y, w, h)
    
    def _redetect(self, frame, search_bbox):
        """
        Trigger YOLOv8 re-detection
        Args:
            frame: Current frame
            search_bbox: Region to search around
        Returns:
            Best matching bbox or None
        """
        detections = self.redetector.detect(frame, search_region=search_bbox, conf_threshold=0.20)
        
        if not detections:
            return None
        
        # Find best match using re-ID
        best_bbox = self.redetector.find_best_match(
            frame, detections, self.template, self.template_hist, search_bbox
        )
        
        return best_bbox
    
    def get_status_info(self):
        """Get current tracking status information"""
        return {
            'frame_count': self.frame_count,
            'low_conf_count': self.low_conf_count,
            'initialized': self.initialized
        }
