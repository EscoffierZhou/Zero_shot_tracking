"""
Kalman Filter for motion prediction and trajectory smoothing
"""
import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanTracker:
    """
    Kalman Filter for 2D bounding box tracking
    State: [x, y, w, h, vx, vy, vw, vh]
    """
    
    def __init__(self):
        # Initialize 8-state Kalman filter (x, y, w, h, vx, vy, vw, vh)
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State transition matrix (constant velocity model)
        dt = 1.0
        self.kf.F = np.array([
            [1, 0, 0, 0, dt, 0, 0, 0],
            [0, 1, 0, 0, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, 0, dt, 0],
            [0, 0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix (we only observe x, y, w, h)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        
        # Measurement noise covariance
        self.kf.R *= 10.0
        
        # Process noise covariance
        self.kf.Q[4:, 4:] *= 0.01
        
        # Initial state covariance
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        
        self.initialized = False
    
    def init(self, bbox):
        """
        Initialize Kalman filter with first bounding box
        Args:
            bbox: (x, y, w, h)
        """
        x, y, w, h = bbox
        self.kf.x = np.array([x, y, w, h, 0, 0, 0, 0]).reshape(8, 1)
        self.initialized = True
    
    def predict(self):
        """
        Predict next state
        Returns:
            Predicted bounding box (x, y, w, h)
        """
        if not self.initialized:
            return None
        
        self.kf.predict()
        predicted = self.kf.x[:4].flatten()
        return tuple(predicted.astype(int))
    
    def update(self, bbox):
        """
        Update filter with measurement
        Args:
            bbox: (x, y, w, h)
        """
        if not self.initialized:
            self.init(bbox)
            return
        
        measurement = np.array(bbox).reshape(4, 1)
        self.kf.update(measurement)
    
    def get_state(self):
        """
        Get current state estimate
        Returns:
            Current bounding box (x, y, w, h)
        """
        if not self.initialized:
            return None
        
        state = self.kf.x[:4].flatten()
        return tuple(state.astype(int))
