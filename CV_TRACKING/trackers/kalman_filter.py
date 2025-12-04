"""
卡尔曼滤波器：用于运动预测和轨迹平滑
"""
import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanTracker:
    """
    用于 2D 边界框跟踪的卡尔曼滤波器
    状态向量: [x, y, w, h, vx, vy, vw, vh]
    """
    
    def __init__(self):
        # 初始化 8 状态卡尔曼滤波器 (x, y, w, h, vx, vy, vw, vh)
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # 状态转移矩阵 (恒速模型)
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
        
        # 测量矩阵 (我们只观测 x, y, w, h)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        
        # 测量噪声协方差
        self.kf.R *= 10.0
        
        # 过程噪声协方差
        self.kf.Q[4:, 4:] *= 0.01
        
        # 初始状态协方差
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        
        self.initialized = False
    
    def init(self, bbox):
        """
        使用第一个边界框初始化卡尔曼滤波器
        参数:
            bbox: (x, y, w, h)
        """
        x, y, w, h = bbox
        self.kf.x = np.array([x, y, w, h, 0, 0, 0, 0]).reshape(8, 1)
        self.initialized = True
    
    def predict(self):
        """
        预测下一个状态
        返回:
            预测的边界框 (x, y, w, h)
        """
        if not self.initialized:
            return None
        
        self.kf.predict()
        predicted = self.kf.x[:4].flatten()
        return tuple(predicted.astype(int))
    
    def update(self, bbox):
        """
        使用测量值更新滤波器
        参数:
            bbox: (x, y, w, h)
        """
        if not self.initialized:
            self.init(bbox)
            return
        
        measurement = np.array(bbox).reshape(4, 1)
        self.kf.update(measurement)
    
    def get_state(self):
        """
        获取当前状态估计
        返回:
            当前边界框 (x, y, w, h)
        """
        if not self.initialized:
            return None
        
        state = self.kf.x[:4].flatten()
        return tuple(state.astype(int))
