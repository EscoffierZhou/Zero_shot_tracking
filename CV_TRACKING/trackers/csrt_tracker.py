"""
CSRT 跟踪器封装：包含置信度监控
"""
import cv2
import numpy as np


class CSRTTracker:
    """
    OpenCV CSRT 跟踪器，带有置信度评估功能
    """
    
    def __init__(self):
        self.tracker = None
        self.last_bbox = None
        self.confidence = 1.0
        self.template = None
        self.template_hist = None
        
    def init(self, frame, bbox):
        """
        使用第一帧和边界框初始化跟踪器
        参数:
            frame: 第一帧 (BGR)
            bbox: 初始边界框 (x, y, w, h)
        """
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, bbox)
        self.last_bbox = bbox
        
        # 保存初始模板用于重识别
        x, y, w, h = [int(v) for v in bbox]
        self.template = frame[y:y+h, x:x+w].copy()
        
        # 计算颜色直方图用于 Re-ID
        hsv = cv2.cvtColor(self.template, cv2.COLOR_BGR2HSV)
        self.template_hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(self.template_hist, self.template_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    def update(self, frame):
        """
        使用新帧更新跟踪器
        参数:
            frame: 当前帧 (BGR)
        返回:
            success: 指示跟踪是否成功的布尔值
            bbox: 更新后的边界框 (x, y, w, h)
            confidence: 跟踪置信度 [0, 1]
        """
        if self.tracker is None:
            return False, None, 0.0
        
        success, bbox = self.tracker.update(frame)
        
        if success:
            self.last_bbox = bbox
            # 基于模板匹配评估置信度
            self.confidence = self._estimate_confidence(frame, bbox)
        else:
            self.confidence = 0.0
            bbox = self.last_bbox  # 返回最后已知位置
        
        return success, bbox, self.confidence
    
    def _estimate_confidence(self, frame, bbox):
        """
        使用模板匹配评估跟踪置信度
        参数:
            frame: 当前帧
            bbox: 当前边界框
        返回:
            confidence: 估计的置信度 [0, 1]
        """
        try:
            x, y, w, h = [int(v) for v in bbox]
            
            # 确保边界框在帧范围内
            h_frame, w_frame = frame.shape[:2]
            if x < 0 or y < 0 or x+w > w_frame or y+h > h_frame or w <= 0 or h <= 0:
                return 0.0
            
            current_patch = frame[y:y+h, x:x+w]
            
            # 检查 patch 是否有效
            if current_patch.size == 0 or self.template.size == 0:
                return 0.0
            
            # 调整 patch 大小以便比较
            size = (64, 64)
            current_resized = cv2.resize(current_patch, size)
            template_resized = cv2.resize(self.template, size)
            
            # 计算颜色直方图相似度
            hsv = cv2.cvtColor(current_resized, cv2.COLOR_BGR2HSV)
            current_hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(current_hist, current_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            hist_similarity = cv2.compareHist(self.template_hist, current_hist, cv2.HISTCMP_CORREL)
            
            # 计算模板匹配分数
            result = cv2.matchTemplate(current_resized, template_resized, cv2.TM_CCOEFF_NORMED)
            template_score = result[0, 0]
            
            # 组合分数
            confidence = 0.6 * hist_similarity + 0.4 * template_score
            confidence = max(0.0, min(1.0, confidence))
            
            return confidence
            
        except Exception as e:
            print(f"置信度评估错误: {e}")
            return 0.5
    
    def get_template(self):
        """获取初始模板用于重识别"""
        return self.template, self.template_hist
    
    def reinit(self, frame, bbox):
        """使用新的检测结果重新初始化跟踪器"""
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, bbox)
        self.last_bbox = bbox
        self.confidence = 1.0
