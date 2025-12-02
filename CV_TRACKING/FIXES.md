# 修复记录

## 问题1: ZeroDivisionError in YOLO11 Detection ✅

**错误原因**：
- 当bbox坐标异常时，裁剪的`search_frame`尺寸为0
- YOLO11的`letterbox`预处理函数对零尺寸图像进行除法运算导致错误

**修复方案** (`trackers/redetector.py`):
1. 添加search_region有效性检查（w > 0, h > 0）
2. 添加最小尺寸限制（100px）
3. 验证裁剪后的区域坐标有效（x2 > x1, y2 > y1）
4. 验证裁剪后的图像非零尺寸
5. 添加try-except捕获YOLO11异常
6. 所有验证失败时fallback到全帧检测

## 问题2: 低置信度时显示边框 ✅

**需求**：
- 置信度 ≤ 0.15 时不显示边界框

**修复方案** (`app.py`):
- 修改可视化条件：`if bbox is not None and confidence > 0.15:`
- 帧计数器始终显示（移到if外）

## 测试建议

重启应用后测试：
```bash
&"F:\Anaconda\envs\pytorch\python.exe" app.py
```

预期行为：
- ✅ 不再出现ZeroDivisionError
- ✅ 低置信度帧（conf ≤ 0.15）不显示边框
- ✅ 跟踪失败时使用全帧重检测
- ✅ 控制台显示警告信息但不崩溃
