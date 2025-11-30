# ==============================================================================
# 0. 关键依赖库检查 (用于调试)
# ==============================================================================
print(">>> [DEBUG] 步骤 0: 检查关键库版本...")
try:
    import mmcv
    import timm
    import ultralytics
    import yaml  # 新增：用于读取配置文件
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
    from collections import Counter
    from PIL import ImageFont, Image, ImageDraw
    import hyperlpr3 as lpr3
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
INPUT_VIDEO_PATH = 'VIDEOS/3.mp4'
OUTPUT_VIDEO_PATH = 'output/output3.mp4'
FONT_PATH = 'platech.ttf'
CONFIG_PATH = 'bytetrack.yaml'  # 配置文件路径

paths_to_check = {
    "YOLOv8 权重": YOLO_MODEL_PATH,
    "Metric3D 权重": METRIC3D_MODEL_PATH,
    "Metric3D 配置": METRIC3D_CONFIG_PATH,
    "输入视频": INPUT_VIDEO_PATH,
    "中文字体": FONT_PATH,
    "参数配置": CONFIG_PATH
}
if not all(os.path.exists(p) for p in paths_to_check.values()):
    missing = [k for k, v in paths_to_check.items() if not os.path.exists(v)]
    raise FileNotFoundError(f"以下关键文件路径无效: {missing}")

# --- 读取 YAML 配置 ---
print(f">>> [INFO] 正在读取配置文件: {CONFIG_PATH}")
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    yaml_config = yaml.safe_load(f)

# 提取速度配置
speed_cfg = yaml_config.get('speed', {})
MIN_SPEED_THRESHOLD = speed_cfg.get('min_speed_threshold', 5.0)
MAX_SPEED_LIMIT = speed_cfg.get('max_speed_limit', 120.0)
SPEED_CALC_TIME_WINDOW = speed_cfg.get('calc_interval_seconds', 2.0)

print(
    f">>> [CONFIG] 速度计算配置: 静止阈值={MIN_SPEED_THRESHOLD}km/h, 上限={MAX_SPEED_LIMIT}km/h, 时间窗口={SPEED_CALC_TIME_WINDOW}s")

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



def ensure_writable_array(arr):
    if not arr.flags.writeable:
        return arr.copy()
    return arr


def draw_text_with_background(img, text, position, font, bg_color=(0, 100, 0), text_color=(255, 255, 255),
                              font_scale=0.6, thickness=2):
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

        if (bg_x2 - bg_x1) < 1: bg_x2 = bg_x1 + 10
        if (bg_y2 - bg_y1) < 1: bg_y2 = bg_y1 + 10

        draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=bg_color)
        draw.text((x, bg_y1 + 1), text, fill=text_color, font=font)
        return np.asarray(data)
    except:
        # Fallback to OpenCV
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

    PLATE_DETECT_INTERVAL = 3
    VOTE_THRESHOLD = 5

    if fps <= 0:
        fps = 30
        print(f">>> [WARNING] 视频FPS获取失败，默认按30FPS计算（每秒30帧）")

    # 速度计算间隔：使用配置的时间窗口
    SPEED_CALC_INTERVAL = int(round(fps * SPEED_CALC_TIME_WINDOW))
    MIN_DEPTH_HISTORY = SPEED_CALC_INTERVAL

    print(f">>> [INFO] 输入视频信息: {width}x{height} @ {fps:.2f} FPS, 共 {total_frames} 帧。")
    print(f">>> [INFO] 速度计算：每隔{SPEED_CALC_INTERVAL}帧（约{SPEED_CALC_TIME_WINDOW}秒）计算一次")

    # --- 使用 YAML 配置初始化跟踪器 ---
    # 过滤掉 speed 字段，将剩余字段转为 Namespace
    tracker_config_dict = {k: v for k, v in yaml_config.items() if k != 'speed'}
    tracker_args = SimpleNamespace(**tracker_config_dict)

    tracker = ByteTracker(args=tracker_args, frame_rate=fps)
    print(f">>> [INFO] ByteTracker 初始化参数: {tracker_config_dict}")

    robust_depth_filters = {}
    track_data = {}
    metric3d_input_size = (cfg.data_basic['vit_size'][1], cfg.data_basic['vit_size'][0])

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

            tip_text = f"速度计算: {SPEED_CALC_TIME_WINDOW}s间隔 | 静止阈值: {MIN_SPEED_THRESHOLD}km/h"
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

            # 步骤 3: 跟踪前
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

            # 步骤 4: 跟踪中
            tracks = tracker.update(np.array(detections_with_depth)) if len(detections_with_depth) > 0 else np.empty(
                (0, 8))

            # 步骤 5: 跟踪后
            active_track_ids = set()
            if tracks.shape[0] > 0:
                for track in tracks:
                    x1, y1, x2, y2 = map(int, track[:4])
                    track_id = int(track[4])
                    active_track_ids.add(track_id)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    if track_id not in track_data:
                        track_data[track_id] = {
                            "formal": {"plates": {}, "best_plate": None},
                            "temp": {"detections": []},
                            "has_voted": False,
                            "depth_history": [],
                            "current_speed": 0.0,
                            "frame_count": 0,
                            "last_speed_calc_frame": 0,
                            "speed_history": [],
                            "previous_speed": 0.0
                        }

                    track_data[track_id]["frame_count"] += 1
                    id_frame_count = track_data[track_id]["frame_count"]

                    # 深度获取逻辑 (保持不变)
                    box_w, box_h = x2 - x1, y2 - y1
                    if box_w <= 0 or box_h <= 0: continue
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

                    # --- 速度计算逻辑更新 ---
                    depth_history = track_data[track_id]["depth_history"]
                    depth_history.append(smoothed_depth)
                    last_speed_calc_frame = track_data[track_id]["last_speed_calc_frame"]
                    current_speed = track_data[track_id]["current_speed"]
                    speed_history = track_data[track_id]["speed_history"]

                    if track_data[track_id]["frame_count"] == 1:
                        track_data[track_id]["current_speed"] = 0.0
                        track_data[track_id]["last_speed_calc_frame"] = frame_count

                    max_history_length = SPEED_CALC_INTERVAL * 3
                    if len(depth_history) > max_history_length:
                        depth_history.pop(0)

                    if (frame_count - last_speed_calc_frame) >= SPEED_CALC_INTERVAL and len(
                            depth_history) >= MIN_DEPTH_HISTORY:
                        recent_depths = depth_history[-SPEED_CALC_INTERVAL:] if len(
                            depth_history) >= SPEED_CALC_INTERVAL else depth_history

                        if len(recent_depths) >= 2:
                            time_interval = len(recent_depths) / fps
                            start_depth = recent_depths[0]
                            end_depth = recent_depths[-1]
                            depth_diff = abs(end_depth - start_depth)

                            speed_mps = depth_diff / time_interval if time_interval > 0 else 0.0
                            raw_speed_kmh = speed_mps * 3.6

                            # 平滑处理
                            previous_speed = track_data[track_id]["previous_speed"]
                            if len(speed_history) == 0:
                                current_speed = raw_speed_kmh
                            else:
                                current_speed = 0.5 * raw_speed_kmh + 0.5 * previous_speed

                            # --- 关键修改：应用速度范围限制和静止阈值 ---
                            # 1. 解决静止显示5km/h问题：如果小于阈值，强制为0
                            if current_speed < MIN_SPEED_THRESHOLD:
                                current_speed = 0.0

                            # 2. 限制最大速度 (0 - 120 km/h)
                            current_speed = min(current_speed, MAX_SPEED_LIMIT)

                            speed_history.append(current_speed)
                            track_data[track_id]["previous_speed"] = current_speed
                            track_data[track_id]["current_speed"] = current_speed
                            track_data[track_id]["last_speed_calc_frame"] = frame_count

                            print(f">>> [ID:{track_id}] 速度更新: {current_speed:.1f} km/h (Raw: {raw_speed_kmh:.1f})")

                    if len(speed_history) > max_history_length:
                        speed_history.pop(0)

                    # 车牌识别逻辑 (保持不变)
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
                                        if len(temp_detections) >= VOTE_THRESHOLD:
                                            perform_voting(track_id, track_data, VOTE_THRESHOLD)
                            except Exception:
                                pass

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

            for inactive_id in set(robust_depth_filters.keys()) - active_track_ids:
                del robust_depth_filters[inactive_id]

            out.write(annotated_frame)
            pbar.update(1)

    # 结尾处理 (保持不变)
    print(f"\n--- 视频处理完成，处理未投票结果 ---")
    for track_id in track_data:
        temp_detections = track_data[track_id]["temp"]["detections"]
        if len(temp_detections) > 0 and not track_data[track_id]["has_voted"]:
            perform_voting(track_id, track_data, VOTE_THRESHOLD, is_supplementary=True)

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