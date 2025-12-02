"""
CSRT Tracker wrapper with confidence monitoring
"""
import cv2
import numpy as np


class CSRTTracker:
    """
    OpenCV CSRT tracker with confidence estimation
    """
    
    def __init__(self):
        self.tracker = None
        self.last_bbox = None
        self.confidence = 1.0
        self.template = None
        self.template_hist = None
        
    def init(self, frame, bbox):
        """
        Initialize tracker with first frame and bounding box
        Args:
            frame: First frame (BGR)
            bbox: Initial bounding box (x, y, w, h)
        """
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, bbox)
        self.last_bbox = bbox
        
        # Save initial template for re-identification
        x, y, w, h = [int(v) for v in bbox]
        self.template = frame[y:y+h, x:x+w].copy()
        
        # Calculate color histogram for re-ID
        hsv = cv2.cvtColor(self.template, cv2.COLOR_BGR2HSV)
        self.template_hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(self.template_hist, self.template_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    def update(self, frame):
        """
        Update tracker with new frame
        Args:
            frame: Current frame (BGR)
        Returns:
            success: Bool indicating if tracking succeeded
            bbox: Updated bounding box (x, y, w, h)
            confidence: Tracking confidence [0, 1]
        """
        if self.tracker is None:
            return False, None, 0.0
        
        success, bbox = self.tracker.update(frame)
        
        if success:
            self.last_bbox = bbox
            # Estimate confidence based on template matching
            self.confidence = self._estimate_confidence(frame, bbox)
        else:
            self.confidence = 0.0
            bbox = self.last_bbox  # Return last known position
        
        return success, bbox, self.confidence
    
    def _estimate_confidence(self, frame, bbox):
        """
        Estimate tracking confidence using template matching
        Args:
            frame: Current frame
            bbox: Current bounding box
        Returns:
            confidence: Estimated confidence [0, 1]
        """
        try:
            x, y, w, h = [int(v) for v in bbox]
            
            # Ensure bbox is within frame bounds
            h_frame, w_frame = frame.shape[:2]
            if x < 0 or y < 0 or x+w > w_frame or y+h > h_frame or w <= 0 or h <= 0:
                return 0.0
            
            current_patch = frame[y:y+h, x:x+w]
            
            # Check if patch is valid
            if current_patch.size == 0 or self.template.size == 0:
                return 0.0
            
            # Resize patches to same size for comparison
            size = (64, 64)
            current_resized = cv2.resize(current_patch, size)
            template_resized = cv2.resize(self.template, size)
            
            # Calculate color histogram similarity
            hsv = cv2.cvtColor(current_resized, cv2.COLOR_BGR2HSV)
            current_hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(current_hist, current_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            hist_similarity = cv2.compareHist(self.template_hist, current_hist, cv2.HISTCMP_CORREL)
            
            # Calculate template matching score
            result = cv2.matchTemplate(current_resized, template_resized, cv2.TM_CCOEFF_NORMED)
            template_score = result[0, 0]
            
            # Combine scores
            confidence = 0.6 * hist_similarity + 0.4 * template_score
            confidence = max(0.0, min(1.0, confidence))
            
            return confidence
            
        except Exception as e:
            print(f"Confidence estimation error: {e}")
            return 0.5
    
    def get_template(self):
        """Get initial template for re-identification"""
        return self.template, self.template_hist
    
    def reinit(self, frame, bbox):
        """Re-initialize tracker with new detection"""
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, bbox)
        self.last_bbox = bbox
        self.confidence = 1.0
