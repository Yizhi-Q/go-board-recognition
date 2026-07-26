import cv2
import numpy as np
import os

def load_and_gray(image_path):
    """读取图像并转换为灰度图"""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray

def calculate_mean_diff(gray1, gray2):
    """计算两张灰度图之间的平均像素差值"""
    diff = cv2.absdiff(gray1, gray2)
    mean_diff = np.mean(diff)
    return diff, mean_diff

def main(img1_path, img2_path, img3_path, diff_threshold=5.0):
    # 加载并转为灰度
    img1, gray1 = load_and_gray(img1_path)
    img2, gray2 = load_and_gray(img2_path)
    img3, gray3 = load_and_gray(img3_path)

    # 计算帧间差值
    diff1, mean1 = calculate_mean_diff(gray1, gray2)
    diff2, mean2 = calculate_mean_diff(gray2, gray3)

    # 输出差值结果
    print(f"图像1-2 平均灰度差值: {mean1:.2f}")
    print(f"图像2-3 平均灰度差值: {mean2:.2f}")

    # 判断是否稳定
    if mean1 < diff_threshold and mean2 < diff_threshold:
        print("结果：图像稳定 ✅")
    else:
        print("结果：图像不稳定 ❌")

    # 保存灰度图和差分图
    os.makedirs("output", exist_ok=True)
    cv2.imwrite("output/gray1.jpg", gray1)
    cv2.imwrite("output/gray2.jpg", gray2)
    cv2.imwrite("output/gray3.jpg", gray3)
    cv2.imwrite("output/diff1.jpg", diff1)
    cv2.imwrite("output/diff2.jpg", diff2)
    print("灰度图和差分图已保存到 ./output/ 目录")

if __name__ == "__main__":
    # 替换成你自己的三张图路径
    img1_path = "output\\20250421221234\\original_move_3.jpg"
    img2_path = "output\\20250421221234\\original_move_10.jpg"
    img3_path = "output\\20250421221234\\original_move_11.jpg"
    
    main(img1_path, img2_path, img3_path)
