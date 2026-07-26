import cv2
import numpy as np

def order_points(pts):
    """ 对四个角点进行排序，确保透视变换的正确性 """
    rect = np.zeros((4, 2), dtype="float32")

    # 按 x+y 升序（左上角最小，右下角最大）
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上角
    rect[2] = pts[np.argmax(s)]  # 右下角

    # 按 x-y 升序（右上角最大，左下角最小）
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上角
    rect[3] = pts[np.argmax(diff)]  # 左下角

    return rect

def warp_perspective(image_path, output_size=(512, 512)):
    """ 透视变换，使棋盘变为标准矩形，并避免旋转 """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. 边缘检测
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # 2. 轮廓检测
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_quad = None
    max_area = 0

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        # 多边形拟合
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:  # 只保留四边形
            area = cv2.contourArea(approx)
            if area > max_area:
                max_area = area
                largest_quad = approx

    if largest_quad is None:
        raise ValueError("未找到有效的棋盘四边形！")

    # 3. 确保角点顺序正确
    ordered_corners = order_points(np.array([point[0] for point in largest_quad], dtype=np.float32))

    # 4. 目标矩形的四个角点
    target_corners = np.array([
        [0, 0],
        [output_size[0] - 1, 0],
        [output_size[0] - 1, output_size[1] - 1],
        [0, output_size[1] - 1]
    ], dtype=np.float32)

    # 5. 计算透视变换矩阵并进行变换
    M = cv2.getPerspectiveTransform(ordered_corners, target_corners)
    warped = cv2.warpPerspective(image, M, output_size)

    return warped

# ex
# image_path = "Test_Image\\61.jpg"
# corrected_board = warp_perspective(image_path)

# cv2.imshow("Corrected Chessboard", corrected_board)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
