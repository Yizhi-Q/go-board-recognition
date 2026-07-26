import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os

# The detector is configured by GoGameProcessor during application startup.
# Keeping model loading out of module import makes the project portable and
# produces a clear error when local weights have not been installed.
model = None

def configure_detector(detector):
    """Set the YOLO detector shared by stone-recognition functions."""
    global model
    model = detector

def detect_stones(image, conf_threshold=0.5):
    """ 使用 YOLOv8 检测棋子，并过滤低置信度检测 """
    if model is None:
        raise RuntimeError(
            "Stone detector is not configured. Place the model checkpoint in "
            "models/best_300epoch_onlybw.pt before starting the application."
        )
    results = model(image)
    detections = []

    for result in results:
        for box in result.boxes:
            x, y, w, h = box.xywh[0]  # 获取中心坐标
            original_label = int(box.cls[0])  
            # 转换标签: 0 = 黑子, 2 = 白子
            label = 0 if original_label == 0 else 2
            confidence = float(box.conf[0])  # 置信度
            
            # 过滤低置信度检测
            if confidence >= conf_threshold:
                detections.append((x.item(), y.item(), w.item(), h.item(), label, confidence))
    
    # 应用非极大值抑制(NMS)后处理
    detections = apply_nms(detections)
    
    return detections, results

def apply_nms(detections, iou_threshold=0.45):
    """ 应用非极大值抑制(NMS)，减少重复检测 """
    if not detections:
        return []
    
    # 按类别分组检测结果
    detections_by_class = {}
    for det in detections:
        label = det[4]  # 标签字段的索引
        if label not in detections_by_class:
            detections_by_class[label] = []
        detections_by_class[label].append(det)
    
    # 对每个类别应用NMS
    filtered_detections = []
    for label, dets in detections_by_class.items():
        # 按置信度排序（降序）
        dets.sort(key=lambda x: x[5], reverse=True)
        
        keep = []
        while len(dets) > 0:
            # 保留置信度最高的检测
            current = dets[0]
            keep.append(current)
            
            if len(dets) == 1:
                break
                
            # 移除已处理的检测
            dets = dets[1:]
            
            # 计算当前检测与剩余检测的IoU
            overlaps = []
            for i, other in enumerate(dets):
                iou = calculate_iou(current, other)
                if iou > iou_threshold:
                    overlaps.append(i)
            
            # 移除重叠的检测（从后往前移除，以保持索引有效）
            for idx in sorted(overlaps, reverse=True):
                if idx < len(dets):
                    dets.pop(idx)
        
        filtered_detections.extend(keep)
    
    return filtered_detections

def calculate_iou(box1, box2):
    """ 计算两个检测框的IoU """
    # 获取中心点坐标和尺寸
    x1, y1, w1, h1 = box1[0], box1[1], box1[2], box1[3]
    x2, y2, w2, h2 = box2[0], box2[1], box2[2], box2[3]
    
    # 计算框的坐标
    box1_x1, box1_y1 = x1 - w1/2, y1 - h1/2
    box1_x2, box1_y2 = x1 + w1/2, y1 + h1/2
    box2_x1, box2_y1 = x2 - w2/2, y2 - h2/2
    box2_x2, box2_y2 = x2 + w2/2, y2 + h2/2
    
    # 计算交集面积
    x_left = max(box1_x1, box2_x1)
    y_top = max(box1_y1, box2_y1)
    x_right = min(box1_x2, box2_x2)
    y_bottom = min(box1_y2, box2_y2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
        
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # 计算两个框的面积
    box1_area = w1 * h1
    box2_area = w2 * h2
    
    # 计算并集面积
    union_area = box1_area + box2_area - intersection_area
    
    # 返回IoU
    return intersection_area / union_area if union_area > 0 else 0

def draw_detection_result(image, detections):
    image_copy = image.copy()
    
    # 颜色映射 (0=黑子, 2=白子)
    colors = {0: (0, 0, 0), 2: (255, 255, 255)}  # 黑子和白子的颜色
    
    for (x, y, w, h, label, conf) in detections:
        # 计算边界框的左上角和右下角点
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)
        
        # 获取标签对应的颜色
        color = (0, 255, 0)  # 默认绿色
        if label in colors:
            color = colors[label]
        
        # 绘制边界框
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
        
        # 绘制中心点
        cv2.circle(image_copy, (int(x), int(y)), 3, (0, 0, 255), -1)
    
    # 保存检测结果图像
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "yolo_detection_result.jpg")
    cv2.imwrite(output_path, image_copy)
    print(f"YOLO检测结果已保存: {output_path}")
    
    return image_copy

def get_board_grid_positions(board_size=19, image_shape=None):
    """ 计算标准棋盘19x19交叉点的像素坐标，根据实际图像尺寸调整 """
    if image_shape is None:
        # 默认尺寸
        output_size = (512, 512)
    else:
        # 使用实际图像尺寸
        output_size = (image_shape[1], image_shape[0])  # width, height
    
    grid_positions = []
    step_x = output_size[0] / (board_size - 1)
    step_y = output_size[1] / (board_size - 1)
    
    for i in range(board_size):
        for j in range(board_size):
            x = int(j * step_x)
            y = int(i * step_y)
            grid_positions.append((x, y, i, j))  # (像素x, 像素y, 棋盘行i, 列j)
    
    return grid_positions

def map_stones_to_grid(detections, grid_positions):
    """ 将棋子像素坐标匹配到最近的棋盘交叉点，确保每个交叉点只有一个最高置信度的棋子 """
    # 创建一个字典来跟踪每个交叉点的最高置信度棋子
    grid_map = {}  # 键: (i, j), 值: (label, confidence)
    
    for (x, y, w, h, label, conf) in detections:
        min_dist = float("inf")
        best_match = None

        for (gx, gy, i, j) in grid_positions:
            dist = np.sqrt((x - gx) ** 2 + (y - gy) ** 2)
            if dist < min_dist:
                min_dist = dist
                best_match = (i, j, label, conf)

        if best_match:
            i, j, label, conf = best_match
            # 如果该交叉点尚未记录棋子或当前棋子置信度更高，则更新
            if (i, j) not in grid_map or conf > grid_map[(i, j)][1]:
                grid_map[(i, j)] = (label, conf)
    
    # 将字典转换为棋子位置列表
    stone_positions = []
    for (i, j), (label, conf) in grid_map.items():
        stone_positions.append((i, j, label))
    
    return stone_positions

def generate_sgf(stone_positions, output_file="game.sgf"):
    """ 根据棋子位置生成 SGF 文件 """
    sgf_content = "(;GM[1]FF[4]SZ[19]PB[Black]PW[White]\n"

    sgf_coords = lambda i, j: chr(ord('a') + j) + chr(ord('a') + i)

    for (i, j, label) in stone_positions:
        # 调整为新的标签系统: 0 = 黑子, 2 = 白子
        move = f";{'B' if label == 0 else 'W'}[{sgf_coords(i, j)}]\n"
        sgf_content += move

    sgf_content += ")"

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, output_file)
    
    with open(full_path, "w") as f:
        f.write(sgf_content)

    print(f"SGF文件已保存: {full_path}")

def generate_board_image(stone_positions, board_size=19):
    # 创建空棋盘列表，使用-1表示空位置
    board_list = [[-1 for _ in range(board_size)] for _ in range(board_size)]
    
    # 将stone_positions中的棋子位置填充到board_list
    for (i, j, label) in stone_positions:
        board_list[i][j] = label
    
    # 棋盘参数定义
    cell_size = 30
    stone_radius = 13
    padding = 30  # 增加边距

    # 计算图像大小
    image_size = 2 * padding + (board_size - 1) * cell_size

    # 创建一个新图像，使用更优质的木纹背景色
    img = Image.new('RGB', (image_size, image_size), color=(219, 181, 127))  # 木黄色背景
    draw = ImageDraw.Draw(img)

    # 尝试加载字体（如果操作系统中有）
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except IOError:
        font = ImageFont.load_default()

    # 绘制棋盘外边框
    draw.rectangle([(padding-2, padding-2), (image_size-padding+2, image_size-padding+2)], outline=(0, 0, 0), width=2)

    # 绘制棋盘网格
    for i in range(board_size):
        # 水平线
        draw.line([(padding, padding + i * cell_size),
                   (image_size - padding, padding + i * cell_size)],
                  fill=(0, 0, 0), width=1)
        # 垂直线
        draw.line([(padding + i * cell_size, padding),
                   (padding + i * cell_size, image_size - padding)],
                  fill=(0, 0, 0), width=1)

    # 绘制星位点（天元和星位）
    star_positions = [3, 9, 15]  # 按照19路棋盘的标准星位（从0开始计数）
    for x in star_positions:
        for y in star_positions:
            draw.ellipse([(padding + x * cell_size - 3, padding + y * cell_size - 3),
                          (padding + x * cell_size + 3, padding + y * cell_size + 3)],
                         fill=(0, 0, 0))

    # 标注坐标，使用更清晰的字体和定位
    for i in range(board_size):
        # 标注列坐标（字母，跳过I）
        if i < 8:
            letter = chr(65 + i)  # A-H
        else:
            letter = chr(65 + i + 1)  # J-T
        draw.text((padding + i * cell_size - 4, padding - 20), letter, fill=(0, 0, 0), font=font)

        # 标注行坐标（数字）
        num = str(i + 1)
        draw.text((padding - 20, padding + i * cell_size - 6), num, fill=(0, 0, 0), font=font)

    for y in range(board_size):
        for x in range(board_size):
            stone_type = board_list[y][x]
            if stone_type == 0 or stone_type == 2:  # 只绘制检测到的棋子
                center_x = padding + x * cell_size
                center_y = padding + y * cell_size
                if stone_type == 2:  # 白棋
                    # 简单的白色圆形
                    draw.ellipse([(center_x - stone_radius, center_y - stone_radius),
                                  (center_x + stone_radius, center_y + stone_radius)],
                                 fill=(255, 255, 255))
                elif stone_type == 0:  # 黑棋
                    # 简单的黑色圆形
                    draw.ellipse([(center_x - stone_radius, center_y - stone_radius),
                                  (center_x + stone_radius, center_y + stone_radius)],
                                 fill=(0, 0, 0))

    # 保存图像
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    img_save_path = os.path.join(output_dir, "recognized_board.png")
    img.save(img_save_path)
    print(f"模拟棋盘图像已保存到: {img_save_path}")

    # 转换为OpenCV格式的图像以便显示
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return cv_img

# def main(image_path):
#     """ 主函数，整合整个流程 """
#     print(f"处理图像: {image_path}")
    
#     # 检查文件是否存在
#     if not os.path.exists(image_path):
#         print(f"错误: 找不到图像文件 {image_path}")
#         return
    
#     # 创建输出目录
#     os.makedirs("output", exist_ok=True)
    
#     # 读取图像
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"错误: 无法读取图像 {image_path}")
#         return
    
#     print(f"图像尺寸: {image.shape[1]}x{image.shape[0]}")
    
#     # 1. 棋子检测（添加置信度阈值过滤）
#     detections_with_info, results = detect_stones(image, conf_threshold=0.5)
#     print(f"检测到 {len(detections_with_info)} 个棋子")
    
#     # 2. 绘制YOLO检测结果框图像
#     detection_image = draw_detection_result(image, detections_with_info)
    
#     # 3. 获取棋盘网格点，使用实际图像尺寸
#     grid_positions = get_board_grid_positions(board_size=19, image_shape=image.shape)
    
#     # 4. 匹配棋子到棋盘网格（优化版本，保留每个交叉点的最高置信度棋子）
#     stone_positions = map_stones_to_grid(detections_with_info, grid_positions)
#     print(f"成功匹配 {len(stone_positions)} 个棋子到棋盘")
    
#     # 5. 生成 SGF 文件
#     generate_sgf(stone_positions)
    
#     # 6. 生成模拟棋盘图像
#     board_image = generate_board_image(stone_positions)
    
#     # 显示检测结果和最终棋盘
#     cv2.imshow("YOLO Detection Result", detection_image)
#     cv2.imshow("Go Board", board_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
#     return detection_image, board_image

# if __name__ == "__main__":
#     image_path = "Test_Image\\pre1.png"
#     main(image_path)
