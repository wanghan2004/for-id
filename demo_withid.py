# ==============================================================================
# 0. 关键依赖库检查 (用于调试)
# ==============================================================================
print(">>> [DEBUG] 步骤 0: 检查关键库版本...")
try:
    import mmcv
    import timm
    import ultralytics
    from filterpy.kalman import KalmanFilter
    from sklearn.mixture import GaussianMixture
    # 新增车牌识别库检查
    import hyperlpr3 as lpr3

    print(f">>> [INFO] mmcv version: {mmcv.__version__}")
    print(f">>> [INFO] timm version: {timm.__version__}")
    print(f">>> [INFO] ultralytics version: {ultralytics.__version__}")
    print(f">>> [INFO] hyperlpr3 version: {lpr3.__version__}")
    print(">>> [INFO] filterpy、scikit-learn (GMM) 和车牌识别库已成功导入。")
except ImportError as e:
    print(f"!!! [ERROR] 缺少核心库: {e}")
    raise
print(">>> [DEBUG] 步骤 0: 检查完成。\n" + "=" * 60 + "\n")

# ==============================================================================
# 1. 导入必要的库
# ==============================================================================
print(">>> [DEBUG] 步骤 1: 开始导入核心库...")
try:
    import cv2
    import torch
    import numpy as np
    from ultralytics import YOLO
    import sys
    import os
    from tqdm import tqdm
    from mmcv import Config
    from types import SimpleNamespace
    from filterpy.kalman import KalmanFilter as FilterPyKalmanFilter
    from sklearn.mixture import GaussianMixture
    from collections import Counter  # 新增：用于车牌投票计数
    # 导入PIL用于中文绘制
    from PIL import ImageFont, Image, ImageDraw
    # 导入车牌识别库
    import hyperlpr3 as lpr3
    # 导入我们自定义的、深度融合的跟踪器
    from custom_byte_tracker import ByteTracker

    print(">>> [DEBUG] 核心库（含车牌识别）导入成功。")
except ImportError as e:
    print(f"!!! [ERROR] 导入核心库失败: {e}")
    raise

# --- 导入 Metric3D 相关的模块 ---
METRIC3D_PATH = r'C:\Users\Administrator\Desktop\autodl-tmp (1)\Metric3D'
if METRIC3D_PATH not in sys.path:
    sys.path.insert(0, METRIC3D_PATH)
try:
    from mono.model.monodepth_model import DepthModel as MonoDepthModel

    print(">>> [DEBUG] Metric3D 模块导入成功。")
except ImportError as e:
    print(f"!!! [ERROR] 从 Metric3D 导入模块失败: {e}")
    raise
print(">>> [DEBUG] 步骤 1: 所有库导入完成。\n" + "=" * 60 + "\n")

# ==============================================================================
# 2. 配置区域与路径检查
# ==============================================================================
print(">>> [DEBUG] 步骤 2: 配置模型和文件路径...")
YOLO_MODEL_PATH = 'weights/epoch30.pt'
METRIC3D_MODEL_PATH = 'weights/metric_depth_vit_large_800k.pth'
METRIC3D_CONFIG_PATH = 'Metric3D/mono/configs/HourglassDecoder/vit.raft5.large.py'
INPUT_VIDEO_PATH = 'VIDEOS/2.mp4'
OUTPUT_VIDEO_PATH = 'output2.mp4'
# 新增：中文字体路径
FONT_PATH = 'platech.ttf'

paths_to_check = {
    "YOLOv8 权重": YOLO_MODEL_PATH,
    "Metric3D 权重": METRIC3D_MODEL_PATH,
    "Metric3D 配置": METRIC3D_CONFIG_PATH,
    "输入视频": INPUT_VIDEO_PATH,
    "中文字体": FONT_PATH
}
if not all(os.path.exists(p) for p in paths_to_check.values()):
    missing = [k for k, v in paths_to_check.items() if not os.path.exists(v)]
    raise FileNotFoundError(f"以下关键文件路径无效: {missing}")

print(">>> [DEBUG] 所有文件路径检查通过。")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f">>> [DEBUG] 将要使用的设备: {DEVICE}")
print(">>> [DEBUG] 步骤 2: 配置完成。\n" + "=" * 60 + "\n")

# ==============================================================================
# 3. 模型加载
# ==============================================================================
print(">>> [DEBUG] 步骤 3: 开始加载深度学习模型...")
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    TARGET_CLASS_NAME = 'Car'
    if hasattr(yolo_model, 'names') and isinstance(yolo_model.names, dict):
        TARGET_CLASS_ID = [k for k, v in yolo_model.names.items() if v == TARGET_CLASS_NAME][0]
        print(f">>> [INFO] 目标类别 '{TARGET_CLASS_NAME}' 已找到, ID为: {TARGET_CLASS_ID}")
    else:
        raise ValueError("YOLO 模型没有有效的 'names' 属性或格式不正确")
except Exception as e:
    print(f"!!! [ERROR] 加载 YOLOv8 模型失败: {e}")
    raise

try:
    cfg = Config.fromfile(METRIC3D_CONFIG_PATH)
    cfg.model.backbone.use_mask_token = False
    metric3d_model = MonoDepthModel(cfg).to(DEVICE)
    checkpoint = torch.load(METRIC3D_MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
    metric3d_model.load_state_dict(state_dict, strict=False)
    metric3d_model.eval()
    print(">>> [SUCCESS] Metric3Dv2 模型加载成功！")
except Exception as e:
    print(f"!!! [FATAL ERROR] 加载 Metric3Dv2 模型时出错: {e}")
    raise

# 新增：初始化车牌识别器和字体
catcher = None
font_ch = None

try:
    catcher = lpr3.LicensePlateCatcher(detect_level=lpr3.DETECT_LEVEL_HIGH)
    print(">>> [SUCCESS] 车牌识别模型（HyperLPR）加载成功！")
except Exception as e:
    print(f">>> [WARNING] 加载车牌识别模型失败: {e}")
    print(">>> [INFO] 将继续运行程序，但车牌识别功能将被禁用。")

try:
    if os.path.exists(FONT_PATH):
        font_ch = ImageFont.truetype(FONT_PATH, 14, 0)
        print(f">>> [SUCCESS] 中文字体加载成功: {FONT_PATH}")
    else:
        print(f">>> [WARNING] 中文字体文件不存在: {FONT_PATH}")
except Exception as e:
    print(f">>> [WARNING] 加载中文字体失败: {e}")
    font_ch = None

print(">>> [DEBUG] 步骤 3: 所有模型加载完成。\n" + "=" * 60 + "\n")

# ==============================================================================
# 4. 视频处理主函数 (最终整合版)
# ==============================================================================
print(">>> [DEBUG] 步骤 4: 定义视频处理函数...")


# 新增：文本绘制辅助函数
def ensure_writable_array(arr):
    """确保NumPy数组是可写的"""
    if not arr.flags.writeable:
        return arr.copy()
    return arr


def draw_text_with_background(img, text, position, font, bg_color=(0, 100, 0), text_color=(255, 255, 255),
                              font_scale=0.6, thickness=2):
    """绘制带背景的文本，支持中文"""
    img = ensure_writable_array(img)
    img_h, img_w = img.shape[:2]

    if not text or font is None:
        return img

    try:
        data = Image.fromarray(img)
        draw = ImageDraw.Draw(data)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w = max(1, text_bbox[2] - text_bbox[0])
        text_h = max(1, text_bbox[3] - text_bbox[1])

        x, y = position
        x = max(5, min(x, img_w - text_w - 5))
        y = max(text_h + 5, min(y, img_h - text_h - 5))

        bg_x1 = x - 2
        bg_y1 = y - text_h - 2
        bg_x2 = x + text_w + 2
        bg_y2 = y + 2

        bg_x1 = max(0, bg_x1)
        bg_y1 = max(0, bg_y1)
        bg_x2 = min(img_w - 1, bg_x2)
        bg_y2 = min(img_h - 1, bg_y2)

        bg_x1, bg_x2 = min(bg_x1, bg_x2), max(bg_x1, bg_x2)
        bg_y1, bg_y2 = min(bg_y1, bg_y2), max(bg_y1, bg_y2)

        if (bg_x2 - bg_x1) < 1:
            bg_x2 = bg_x1 + 10
        if (bg_y2 - bg_y1) < 1:
            bg_y2 = bg_y1 + 10

        draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=bg_color)
        draw.text((x, bg_y1 + 1), text, fill=text_color, font=font)
        return np.asarray(data)
    except:
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x, y = position
        x = max(5, min(x, img_w - text_w - 5))
        y = max(text_h + 5, min(y, img_h - text_h - 5))

        bg_x1 = max(0, x - 2)
        bg_y1 = max(0, y - text_h - 2)
        bg_x2 = min(img_w - 1, x + text_w + 2)
        bg_y2 = min(img_h - 1, y + 2)

        bg_x1, bg_x2 = min(bg_x1, bg_x2), max(bg_x1, bg_x2)
        bg_y1, bg_y2 = min(bg_y1, bg_y2), max(bg_y1, bg_y2)

        if (bg_x2 - bg_x1) < 1:
            bg_x2 = bg_x1 + 10
        if (bg_y2 - bg_y1) < 1:
            bg_y2 = bg_y1 + 10

        cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
        cv2.putText(img, text, (x, y - 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
        return img



def perform_voting(track_id, track_data, vote_threshold, is_supplementary=False):
    temp_detections = track_data[track_id]["temp"]["detections"]
    voting_detections = temp_detections if is_supplementary else temp_detections[:vote_threshold]
    plates = [d[0] for d in voting_detections]
    confidences = [d[1] for d in voting_detections]

    plate_counter = Counter(plates)
    vote_type = "补充投票" if is_supplementary else "常规投票"
    print(f">>> [ID:{track_id}] {vote_type} - 有效结果统计：{dict(plate_counter)}")

    final_plate = None
    for plate, count in plate_counter.most_common():
        if count >= 3:
            final_plate = plate
            print(f">>> [ID:{track_id}] {vote_type}结果：车牌 {plate} 出现{count}次（≥3次）→ 选中")
            break
    if final_plate is None:
        max_conf_idx = np.argmax(confidences)
        final_plate = plates[max_conf_idx]
        max_conf = confidences[max_conf_idx]
        print(f">>> [ID:{track_id}] {vote_type}结果：选中置信度最高 {final_plate}（{max_conf:.2f}）")

    formal_plates = track_data[track_id]["formal"]["plates"]
    if final_plate in formal_plates:
        formal_plates[final_plate] += 1
    else:
        formal_plates[final_plate] = 1
    track_data[track_id]["formal"]["best_plate"] = max(formal_plates.items(), key=lambda x: x[1])[0]
    track_data[track_id]["has_voted"] = True

    if not is_supplementary:
        track_data[track_id]["temp"]["detections"] = []


def process_video_with_robust_depth_fusion(input_path, output_path):
    print("\n--- 开始视频处理 (最终鲁棒深度融合追踪+车牌识别+速度计算) ---")
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 新增：配置参数
    PLATE_DETECT_INTERVAL = 3  # 每3帧检测一次车牌
    VOTE_THRESHOLD = 5  # 投票阈值

    # 速度计算配置：每隔fps*2帧（2秒）计算一次速度
    if fps <= 0:
        fps = 30
        print(f">>> [WARNING] 视频FPS获取失败，默认按30FPS计算（每秒30帧）")
    SPEED_CALC_INTERVAL = int(round(fps * 2))  # 速度计算间隔（帧数），即2秒对应的帧数
    MIN_DEPTH_HISTORY = SPEED_CALC_INTERVAL  # 计算速度所需的最小深度历史帧数

    print(f">>> [INFO] 输入视频信息: {width}x{height} @ {fps:.2f} FPS, 共 {total_frames} 帧。")
    print(f">>> [INFO] 车牌识别：每{PLATE_DETECT_INTERVAL}帧检测一次，每{VOTE_THRESHOLD}次结果投票")
    print(f">>> [INFO] 速度计算：每隔{SPEED_CALC_INTERVAL}帧（2秒）计算一次，基于2秒内的深度变化")

    # 初始化跟踪器
    tracker_args = SimpleNamespace(
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        track_buffer=30,
        match_thresh=0.8,
        mot20=False
    )
    tracker = ByteTracker(args=tracker_args, frame_rate=fps)

    # 原有深度滤波器
    robust_depth_filters = {}

    # 跟踪数据存储结构
    track_data = {}

    # 修复：定义 Metric3D 输入尺寸（从配置中获取）
    metric3d_input_size = (cfg.data_basic['vit_size'][1], cfg.data_basic['vit_size'][0])

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    with tqdm(total=total_frames, desc="视频处理进度") as pbar:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            annotated_frame = frame.copy()

            # 顶部提示信息
            tip_text = f"每{PLATE_DETECT_INTERVAL}帧检测车牌 | 每{VOTE_THRESHOLD}次结果投票 | 速度计算：每{SPEED_CALC_INTERVAL}帧（2秒）"
            annotated_frame = draw_text_with_background(
                annotated_frame, tip_text, (20, 25),
                font_ch, bg_color=(255, 140, 0), text_color=(255, 255, 255)
            )

            # 步骤 1: 目标检测
            det_results = yolo_model(frame, classes=[TARGET_CLASS_ID], verbose=False)[0]

            # 步骤 2: 全局深度图预测
            with torch.no_grad():
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame_resized = cv2.resize(rgb_frame, metric3d_input_size)
                rgb_torch = torch.from_numpy(rgb_frame_resized).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
                pred_output = metric3d_model(data={'input': rgb_torch})
                pred_depth_np = pred_output[0].squeeze().cpu().numpy()
                pred_depth_resized = cv2.resize(pred_depth_np, (width, height)).astype(np.float32)
                pred_depth_filtered = cv2.bilateralFilter(pred_depth_resized, d=5, sigmaColor=0.2, sigmaSpace=15)

            # 步骤 3: 跟踪前 - 计算初始深度
            detections_with_depth = []
            if det_results.boxes.shape[0] > 0:
                for box in det_results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    score = box.conf[0].item()
                    cls_id = box.cls[0].item()

                    box_w, box_h = x2 - x1, y2 - y1
                    if box_w <= 0 or box_h <= 0: continue

                    roi_w, roi_h = int(box_w * 0.25), int(box_h * 0.25)
                    roi_x1, roi_y1 = x1 + (box_w - roi_w) // 2, y1 + (box_h - roi_h) // 2
                    roi_x2, roi_y2 = roi_x1 + roi_w, roi_y1 + roi_h

                    depth_roi = pred_depth_filtered[roi_y1:roi_y2, roi_x1:roi_x2]
                    initial_depth = np.median(depth_roi) if depth_roi.size > 0 else 0.0

                    detections_with_depth.append([x1, y1, x2, y2, score, cls_id, initial_depth])

            # 步骤 4: 跟踪中 - 调用自定义跟踪器
            tracks = tracker.update(np.array(detections_with_depth)) if len(detections_with_depth) > 0 else np.empty(
                (0, 8))

            # 步骤 5: 跟踪后 - 处理跟踪结果
            active_track_ids = set()
            if tracks.shape[0] > 0:
                for track in tracks:
                    x1, y1, x2, y2 = map(int, track[:4])
                    track_id = int(track[4])
                    active_track_ids.add(track_id)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # 初始化跟踪数据
                    if track_id not in track_data:
                        track_data[track_id] = {
                            "formal": {"plates": {}, "best_plate": None},
                            "temp": {"detections": []},
                            "has_voted": False,
                            "depth_history": [],
                            "current_speed": 0.0,
                            "frame_count": 0,
                            "last_speed_calc_frame": 0,
                            "speed_history": [],  # 新增：速度历史
                            "previous_speed": 0.0  # 新增：上一次速度
                        }

                    # 更新帧计数
                    track_data[track_id]["frame_count"] += 1
                    id_frame_count = track_data[track_id]["frame_count"]

                    box_w, box_h = x2 - x1, y2 - y1
                    if box_w <= 0 or box_h <= 0: continue

                    # 深度计算（原有逻辑）
                    roi_w, roi_h = int(box_w * 0.5), int(box_h * 0.5)
                    roi_x1, roi_y1 = max(x1 + (box_w - roi_w) // 2, 0), max(y1 + (box_h - roi_h) // 2, 0)
                    roi_x2, roi_y2 = min(roi_x1 + roi_w, width), min(roi_y1 + roi_h, height)
                    depth_roi = pred_depth_filtered[roi_y1:roi_y2, roi_x1:roi_x2]

                    observed_depth = 0.0
                    if depth_roi.size > 10:
                        try:
                            pixels = depth_roi.flatten().reshape(-1, 1)
                            n_components_range = range(1, 4)
                            lowest_bic = np.infty
                            best_gmm = None
                            for n_components in n_components_range:
                                gmm = GaussianMixture(n_components=n_components, random_state=0)
                                gmm.fit(pixels)
                                bic_score = gmm.bic(pixels)
                                if bic_score < lowest_bic:
                                    lowest_bic, best_gmm = bic_score, gmm

                            cluster_means = best_gmm.means_.flatten()

                            if track_id in robust_depth_filters:
                                kf = robust_depth_filters[track_id]
                                kf.predict()
                                predicted_depth = kf.x[0]
                                observed_depth = min(cluster_means, key=lambda x: abs(x - predicted_depth))
                            else:
                                observed_depth = min(cluster_means)
                        except Exception:
                            observed_depth = np.median(depth_roi) if depth_roi.size > 0 else 0
                    elif depth_roi.size > 0:
                        observed_depth = np.median(depth_roi)

                    if observed_depth <= 0: continue

                    # 深度滤波（原有逻辑，不修改）
                    if track_id not in robust_depth_filters:
                        kf = FilterPyKalmanFilter(dim_x=2, dim_z=1)
                        kf.x = np.array([observed_depth, 0.])
                        kf.F = np.array([[1., 1.], [0., 1.]]);
                        kf.H = np.array([[1., 0.]])
                        kf.P *= 100.;
                        kf.R = 5;
                        kf.Q = 0.1
                        robust_depth_filters[track_id] = kf
                    else:
                        kf = robust_depth_filters[track_id]
                        kf.update(observed_depth)

                    smoothed_depth = kf.x[0]

                    # 新增：速度计算 - 每隔SPEED_CALC_INTERVAL帧（2秒）计算一次
                    depth_history = track_data[track_id]["depth_history"]
                    depth_history.append(smoothed_depth)
                    last_speed_calc_frame = track_data[track_id]["last_speed_calc_frame"]
                    current_speed = track_data[track_id]["current_speed"]
                    speed_history = track_data[track_id]["speed_history"]  # 移动到这里

                    # 初始化新 ID 的速度为 0，并记录当前帧数
                    if track_data[track_id]["frame_count"] == 1:
                        track_data[track_id]["current_speed"] = 0.0
                        track_data[track_id]["last_speed_calc_frame"] = frame_count

                    # 限制深度历史记录长度为2倍的计算间隔，避免内存占用过大
                    max_history_length = SPEED_CALC_INTERVAL * 3
                    if len(depth_history) > max_history_length:
                        depth_history.pop(0)

                    # 检查是否达到计算间隔（当前帧数 - 上一次计算帧数 >= 计算间隔）
                    if (frame_count - last_speed_calc_frame) >= SPEED_CALC_INTERVAL and len(
                            depth_history) >= MIN_DEPTH_HISTORY:
                        # 打印深度帧数信息
                        print(f">>> [ID:{track_id}] 当前深度历史帧数: {len(depth_history)}")

                        # 获取最近SPEED_CALC_INTERVAL帧的深度数据
                        recent_depths = depth_history[-SPEED_CALC_INTERVAL:] if len(
                            depth_history) >= SPEED_CALC_INTERVAL else depth_history

                        # 计算深度变化（使用首尾帧的差值，更稳定）
                        if len(recent_depths) >= 2:
                            start_frame = frame_count - len(recent_depths) + 1
                            end_frame = frame_count
                            start_depth = recent_depths[0]
                            end_depth = recent_depths[-1]
                            depth_diff = abs(end_depth - start_depth)
                            time_interval = len(recent_depths) / fps  # 实际时间间隔（秒）

                            speed_mps = depth_diff / time_interval if time_interval > 0 else 0.0
                            current_speed = speed_mps * 3.6  # 转换为km/h
                            current_speed = max(5.0, min(current_speed, 50.0))  # 限制速度范围5-35km/h

                            # 平滑逻辑
                            previous_speed = track_data[track_id]["previous_speed"]

                            if len(speed_history) == 0:
                                # 初次计算，直接使用
                                print(f">>> [ID:{track_id}] 初次速度计算: {current_speed:.1f} km/h (无平滑)")
                            else:
                                # 平滑处理
                                current_speed = 0.5 * current_speed + 0.5 * previous_speed
                                current_speed = max(5.0, min(current_speed, 35.0))  # 重新限制范围
                                print(f">>> [ID:{track_id}] 平滑后速度: {current_speed:.1f} km/h (基于上一速度 {previous_speed:.1f})")

                            speed_history.append(current_speed)
                            track_data[track_id]["previous_speed"] = current_speed
                            track_data[track_id]["current_speed"] = current_speed

                            # 更新跟踪数据
                            track_data[track_id]["last_speed_calc_frame"] = frame_count

                            # 打印速度信息（包含当前帧数和深度值）
                            print(
                                f">>> [ID:{track_id}] 当前帧数: {frame_count}, 速度: {current_speed:.1f} km/h (计算间隔: {len(recent_depths)}帧/{time_interval:.2f}秒)")
                            print(f">>> [ID:{track_id}] 用于计算速度的帧数范围: {start_frame} 到 {end_frame}")
                            print(
                                f">>> [ID:{track_id}] 深度变化: 帧{start_frame}深度={start_depth:.2f} -> 帧{end_frame}深度={end_depth:.2f}, 深度差={depth_diff:.2f}")

                    # 限制速度历史长度
                    if len(speed_history) > max_history_length:
                        speed_history.pop(0)

                    # 新增：车牌识别
                    if catcher is not None and (id_frame_count % PLATE_DETECT_INTERVAL == 0):
                        car_height = y2 - y1
                        plate_roi_y1 = y1 + int(car_height * 2 / 3)
                        plate_roi_y2 = y2
                        plate_roi = frame[plate_roi_y1:plate_roi_y2, x1:x2]

                        if plate_roi.size > 0:
                            try:
                                results = catcher(plate_roi)
                                if results:
                                    best_detection = max(results, key=lambda x: x[1])
                                    code, confidence, _, _ = best_detection
                                    if confidence > 0.85:
                                        temp_detections = track_data[track_id]["temp"]["detections"]
                                        if len(temp_detections) < VOTE_THRESHOLD:
                                            temp_detections.append((code, confidence))
                                            print(f">>> [ID:{track_id}] 缓存车牌 {code}（置信度 {confidence:.2f}）")

                                        if len(temp_detections) >= VOTE_THRESHOLD:
                                            perform_voting(track_id, track_data, VOTE_THRESHOLD)
                            except Exception as e:
                                print(f">>> [ID:{track_id}] 车牌识别失败: {str(e)[:50]}")

                    # 新增：可视化信息
                    best_plate = track_data[track_id]["formal"]["best_plate"]
                    if best_plate:
                        display_text = f"车牌: {best_plate} | 速度: {current_speed:.1f} km/h"
                        bg_color = (0, 0, 150)
                    else:
                        display_text = f"ID:{track_id} | 速度: {current_speed:.1f} km/h"
                        bg_color = (150, 0, 0)

                    text_pos = (x1 + 3, y1 - 10) if (y1 - 10) > 30 else (x1 + 3, y2 + 20)
                    annotated_frame = draw_text_with_background(
                        annotated_frame, display_text, text_pos,
                        font_ch, bg_color=bg_color, text_color=(255, 255, 255)
                    )

            # 清理 inactive 跟踪ID的滤波器（原有逻辑）
            for inactive_id in set(robust_depth_filters.keys()) - active_track_ids:
                del robust_depth_filters[inactive_id]

            out.write(annotated_frame)
            pbar.update(1)

    # 处理未完成投票的结果
    print(f"\n--- 视频处理完成，处理未投票结果 ---")
    for track_id in track_data:
        temp_detections = track_data[track_id]["temp"]["detections"]
        if len(temp_detections) > 0 and not track_data[track_id]["has_voted"]:
            print(f">>> [ID:{track_id}] 补充投票（{len(temp_detections)}次结果）")
            perform_voting(track_id, track_data, VOTE_THRESHOLD, is_supplementary=True)

    # 输出统计信息
    print(f"\n--- 最终统计:")
    for track_id in track_data:
        best_plate = track_data[track_id]["formal"]["best_plate"]
        final_speed = track_data[track_id]["current_speed"]
        total_frames_tracked = track_data[track_id]["frame_count"]
        speed_calc_times = track_data[track_id][
                               "last_speed_calc_frame"] // SPEED_CALC_INTERVAL if SPEED_CALC_INTERVAL > 0 else 0
        print(
            f"  [ID:{track_id}] 最优车牌: {best_plate or '无'} | 最终速度: {final_speed:.1f} km/h | 跟踪帧数: {total_frames_tracked} | 速度计算次数: {speed_calc_times}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n--- 视频处理完成！输出保存在: {output_path} ---")

# ==============================================================================
# 5. 运行主程序
# ==============================================================================
print(">>> [DEBUG] 步骤 5: 开始执行主程序...")
try:
    process_video_with_robust_depth_fusion(INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH)
except Exception as e:
    print(f"!!! [FATAL ERROR] 在视频处理过程中发生严重错误: {e}")
    import traceback

    traceback.print_exc()
print(">>> [DEBUG] 步骤 5: 主程序执行完毕。\n" + "=" * 60)