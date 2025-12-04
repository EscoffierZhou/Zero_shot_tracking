"""
基于 YOLO11 的重检测器，用于遮挡恢复
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


class ReDetector:
    """
    基于触发机制的重检测器，使用 YOLO11n 和 CUDA 加速
    """
    
    def __init__(self, device='cuda', model_path=None):
        """
        初始化 YOLO11 检测器
        参数:
            device: 'cuda' 或 'cpu'
            model_path: YOLO11 模型文件路径 (默认: models/yolo11n.pt)
        """
        if model_path is None:
            # 使用本地 models 目录中的 YOLO11n (最快)
            model_path = Path(__file__).parent.parent / 'models' / 'yolo11n.pt'
        
        print(f"🔍 正在从以下位置加载 YOLO11 模型: {model_path}")
        self.model = YOLO(str(model_path))
        self.device = device
        
        # 将模型移动到设备
        if device == 'cuda':
            self.model.to(device)
        print(f"✓ YOLO11 已加载到 {device.upper()}")
        
    def detect(self, frame, search_region=None, conf_threshold=0.25):
        """
        在帧中检测对象
        参数:
            frame: 输入帧 (BGR)
            search_region: 可选 (x, y, w, h) 以限制搜索区域
            conf_threshold: 检测的置信度阈值
        返回:
            检测列表 [(x, y, w, h, conf, class_id), ...]
        """
        # 如果指定了搜索区域，则裁剪帧
        if search_region is not None:
            x, y, w, h = [int(v) for v in search_region]
            h_frame, w_frame = frame.shape[:2]
            
            # 验证搜索区域
            if w <= 0 or h <= 0:
                print(f"⚠️ 无效的搜索区域: w={w}, h={h}, 使用全帧")
                search_frame = frame
                offset = (0, 0)
            else:
                # 将搜索区域扩大 50%
                expand_factor = 1.5
                x_center, y_center = x + w//2, y + h//2
                w_expanded = int(max(w * expand_factor, 100))  # 最小 100px
                h_expanded = int(max(h * expand_factor, 100))  # 最小 100px
                
                x1 = max(0, x_center - w_expanded//2)
                y1 = max(0, y_center - h_expanded//2)
                x2 = min(w_frame, x_center + w_expanded//2)
                y2 = min(h_frame, y_center + h_expanded//2)
                
                # 确保裁剪区域有效
                if x2 <= x1 or y2 <= y1:
                    print(f"⚠️ 无效的裁剪区域: ({x1},{y1}) 到 ({x2},{y2}), 使用全帧")
                    search_frame = frame
                    offset = (0, 0)
                else:
                    search_frame = frame[y1:y2, x1:x2]
                    offset = (x1, y1)
                    
                    # 最终验证 - 确保非零尺寸
                    if search_frame.shape[0] == 0 or search_frame.shape[1] == 0:
                        print(f"⚠️ 搜索帧尺寸为零, 使用全帧")
                        search_frame = frame
                        offset = (0, 0)
        else:
            search_frame = frame
            offset = (0, 0)
        
        # 运行 YOLO11 检测
        try:
            results = self.model.predict(search_frame, conf=conf_threshold, verbose=False)[0]
        except Exception as e:
            print(f"⚠️ YOLO11 检测错误: {e}")
            return []
        
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # 转换为 (x, y, w, h) 格式并加上偏移量
            x = int(x1) + offset[0]
            y = int(y1) + offset[1]
            w = int(x2 - x1)
            h = int(y2 - y1)
            
            detections.append((x, y, w, h, conf, cls))
        
        return detections
    
    def find_best_match(self, frame, detections, template, template_hist, prev_bbox):
        """
        使用重识别寻找最佳匹配的检测结果
        参数:
            frame: 当前帧
            detections: YOLO 检测列表
            template: 初始模板图像
            template_hist: 初始模板直方图
            prev_bbox: 上一个边界框，用于空间邻近度计算
        返回:
            最佳匹配的边界框 (x, y, w, h) 或 None
        """
        if not detections:
            return None
        
        best_score = -1
        best_bbox = None
        
        prev_center = (prev_bbox[0] + prev_bbox[2]//2, prev_bbox[1] + prev_bbox[3]//2)
        
        for det in detections:
            x, y, w, h, conf, cls = det
            
            # 提取当前 patch
            try:
                patch = frame[y:y+h, x:x+w]
                if patch.size == 0:
                    continue
                
                # 调整大小以便比较
                patch_resized = cv2.resize(patch, (64, 64))
                template_resized = cv2.resize(template, (64, 64))
                
                # 直方图相似度
                hsv = cv2.cvtColor(patch_resized, cv2.COLOR_BGR2HSV)
                patch_hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
                cv2.normalize(patch_hist, patch_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                hist_sim = cv2.compareHist(template_hist, patch_hist, cv2.HISTCMP_CORREL)
                
                # 模板匹配
                tmpl_result = cv2.matchTemplate(patch_resized, template_resized, cv2.TM_CCOEFF_NORMED)
                tmpl_score = tmpl_result[0, 0]
                
                # 空间邻近度 (归一化距离)
                curr_center = (x + w//2, y + h//2)
                dist = np.sqrt((curr_center[0] - prev_center[0])**2 + (curr_center[1] - prev_center[1])**2)
                max_dist = np.sqrt(frame.shape[0]**2 + frame.shape[1]**2)
                proximity_score = 1.0 - (dist / max_dist)
                
                # 组合分数
                score = 0.4 * hist_sim + 0.3 * tmpl_score + 0.2 * proximity_score + 0.1 * conf
                
                if score > best_score:
                    best_score = score
                    best_bbox = (x, y, w, h)
                    
            except Exception as e:
                print(f"检测结果 Re-ID 错误: {e}")
                continue
        
        # 如果分数高于阈值，则返回最佳匹配
        if best_score > 0.5:
            return best_bbox
        return None
