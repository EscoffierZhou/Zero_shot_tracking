"""
混合跟踪器：结合 CSRT、卡尔曼滤波和 YOLOv8 重检测
"""
import cv2
import numpy as np
from .csrt_tracker import CSRTTracker
from .kalman_filter import KalmanTracker
from .redetector import ReDetector


class HybridTracker:
    """
    多层跟踪系统：
    第 1 层：CSRT（主跟踪）
    第 2 层：卡尔曼滤波（运动预测和平滑）
    第 3 层：模板匹配（快速恢复）
    第 4 层：YOLO11 重检测（遮挡恢复）
    """
    
    def __init__(self, device='cuda', 
                 conf_low = 0.2, # 0.3
                 conf_high = 0.7,# 0.7
                 redetect_threshold=10
                 ):
        """
        参数:
            device: 'cuda' 或 'cpu'
            conf_low: 低置信度阈值（触发模板匹配）
            conf_high: 高置信度阈值（正常跟踪）
            redetect_threshold: 触发重检测前的低置信度帧数
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
        
        # Re-ID 模板
        self.template = None
        self.template_hist = None
        
    def init(self, frame, bbox):
        """
        初始化所有跟踪组件
        参数:
            frame: 第一帧 (BGR)
            bbox: 初始边界框 (x, y, w, h)
        """
        self.csrt.init(frame, bbox)
        self.kalman.init(bbox)
        self.template, self.template_hist = self.csrt.get_template()
        self.initialized = True
        self.frame_count = 0
        self.low_conf_count = 0
        
        print(f"✓ 跟踪器初始化于边界框: {bbox}")
    
    def update(self, frame):
        """
        使用新帧更新跟踪器（自适应多层策略）
        参数:
            frame: 当前帧 (BGR)
        返回:
            bbox: 跟踪到的边界框 (x, y, w, h)
            confidence: 跟踪置信度 [0, 1]
            status: 跟踪状态字符串
        """
        if not self.initialized:
            return None, 0.0, "NOT_INITIALIZED"
        
        self.frame_count += 1
        
        # 步骤 1: CSRT 跟踪
        success, csrt_bbox, confidence = self.csrt.update(frame)
        
        # 步骤 2: 卡尔曼预测
        kalman_pred = self.kalman.predict()
        
        # 基于置信度的自适应策略
        if confidence > self.conf_high:
            # 高置信度：直接使用 CSRT
            final_bbox = csrt_bbox
            self.kalman.update(csrt_bbox)
            self.low_conf_count = 0
            status = "TRACKING"
            
        elif confidence > self.conf_low:
            # 中等置信度：混合 CSRT 和卡尔曼
            if kalman_pred is not None:
                final_bbox = self._blend_bbox(csrt_bbox, kalman_pred, alpha=0.7)
            else:
                final_bbox = csrt_bbox
            self.kalman.update(final_bbox)
            self.low_conf_count += 1
            status = "TRACKING_KALMAN"
            
        else:
            # 低置信度：增加计数器
            self.low_conf_count += 1
            
            if self.low_conf_count < self.redetect_threshold:
                # 使用卡尔曼预测
                if kalman_pred is not None:
                    final_bbox = kalman_pred
                    status = "PREDICTING"
                else:
                    final_bbox = csrt_bbox
                    status = "LOW_CONFIDENCE"
            else:
                # 触发重检测
                print(f"⚠ 低置信度持续 {self.low_conf_count} 帧，触发重检测...")
                final_bbox = self._redetect(frame, kalman_pred or csrt_bbox)
                
                if final_bbox is not None:
                    # 重检测成功，重新初始化跟踪器
                    self.csrt.reinit(frame, final_bbox)
                    self.kalman.init(final_bbox)
                    self.low_conf_count = 0
                    confidence = 0.8
                    status = "REDETECTED"
                    print(f"✓ 重检测成功于 {final_bbox}")
                else:
                    # 重检测失败，继续预测
                    final_bbox = kalman_pred or csrt_bbox
                    confidence = 0.1
                    status = "LOST"
                    print("✗ 重检测失败，使用预测值")
        
        return final_bbox, confidence, status
    
    def _blend_bbox(self, bbox1, bbox2, alpha=0.7):
        """
        混合两个边界框
        参数:
            bbox1: 第一个边界框 (x, y, w, h)
            bbox2: 第二个边界框 (x, y, w, h)
            alpha: bbox1 的权重 (1-alpha 为 bbox2 的权重)
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
        触发 YOLOv8 重检测
        参数:
            frame: 当前帧
            search_bbox: 搜索区域
        返回:
            最佳匹配的边界框或 None
        """
        detections = self.redetector.detect(frame, search_region=search_bbox, conf_threshold=0.20)
        
        if not detections:
            return None
        
        # 使用 Re-ID 寻找最佳匹配
        best_bbox = self.redetector.find_best_match(
            frame, detections, self.template, self.template_hist, search_bbox
        )
        
        return best_bbox
    
    def get_status_info(self):
        """获取当前跟踪状态信息"""
        return {
            'frame_count': self.frame_count,
            'low_conf_count': self.low_conf_count,
            'initialized': self.initialized
        }
