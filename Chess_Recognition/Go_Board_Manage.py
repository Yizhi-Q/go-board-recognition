import cv2
import numpy as np
from scipy.spatial import distance

class ReferenceBoardManager:
    def __init__(self):
        # 初始化参考图像和相关变量
        self.reference_image = None  # 参考图像
        self.grid_points = None      # 提取到的网格交点
        self.standard_grid = None    # 标准化的 19x19 网格坐标
        self.has_reference = False   # 是否已经设置参考图像

    def capture_reference(self, image):
        """
        捕获并处理参考棋盘图像
        Args:
            image: 一张清晰的空或接近空的棋盘图像
        Returns:
            bool: 若成功提取参考网格点返回 True
        """
        self.reference_image = image.copy()
        success = self.extract_grid_points()
        if success:
            self.has_reference = True
        return success

    def extract_grid_points(self):
        """
        使用 Shi-Tomasi 角点检测从参考图像中提取网格交叉点
        
        Returns:
            bool: 如果成功提取网格点则返回 True
        """
        if self.reference_image is None:
            return False
            
        # 转换为灰度图
        gray = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2GRAY)
        
        # 使用 Shi-Tomasi 角点检测
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=361,       # 最多检测19x19=361个交叉点
            qualityLevel=0.01,    # 角点质量阈值（越小越灵敏）
            minDistance=20        # 邻近角点的最小间距，避免重复检测
        )
        
        # 检查是否检测到足够的角点
        if corners is None or len(corners) < 300:  # 需要至少检测到300个角点
            print(f"检测到的角点数量不足: {0 if corners is None else len(corners)}")
            return False
        
        # 将角点转换为 float32 类型（cv2.cornerSubPix 需要）
        corners = np.float32(corners)
        
        # 定义亚像素角点精细化参数，提高角点位置精度
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cv2.cornerSubPix(gray, corners, winSize=(5,5), zeroZone=(-1,-1), criteria=criteria)
        
        # 生成可视化结果，保存参考图像和角点
        visualization = self.reference_image.copy()
        for pt in corners:
            x, y = pt.ravel()
            cv2.circle(visualization, (int(x), int(y)), 4, (0, 255, 0), -1)
        
        # 保存可视化结果以便调试
        if hasattr(self, 'output_dir') and self.output_dir:
            cv2.imwrite(f"{self.output_dir}/detected_corners.jpg", visualization)
        
        # 将角点转换为需要的格式
        intersection_points = []
        for corner in corners:
            x, y = corner.ravel()
            intersection_points.append((x, y))
        
        # 继续原有的聚类和排序处理
        try:
            from sklearn.cluster import DBSCAN
            
            # 转换为numpy数组
            points = np.array(intersection_points)
            
            # 应用DBSCAN聚类
            clustering = DBSCAN(eps=10, min_samples=1).fit(points)
            
            # 获取聚类中心
            unique_labels = set(clustering.labels_)
            clustered_points = []
            
            for label in unique_labels:
                cluster_points = points[clustering.labels_ == label]
                center = np.mean(cluster_points, axis=0)
                clustered_points.append(center)
        except ImportError:
            # 如果sklearn不可用，使用更简单的方法
            points = np.array(intersection_points)
            clustered_points = self._simple_clustering(points)
            
        # 将点排序成网格结构
        sorted_points = self._sort_points_to_grid(clustered_points)
        
        # 存储网格点
        self.grid_points = sorted_points
        
        # 创建标准网格（19x19归一化坐标）
        self.create_standard_grid()
        
        return True

    def _simple_clustering(self, points, threshold=10):
        """
        简易聚类算法（不依赖 sklearn），用于聚合相邻点
        """
        clusters = []
        for point in points:
            assigned = False
            for i, cluster in enumerate(clusters):
                if np.min(np.sqrt(np.sum((cluster - point) ** 2, axis=1))) < threshold:
                    clusters[i] = np.vstack([cluster, point])
                    assigned = True
                    break
            if not assigned:
                clusters.append(np.array([point]))
        # 计算每个簇的中心
        centers = [np.mean(cluster, axis=0) for cluster in clusters]
        return centers

    def create_standard_grid(self):
        """
        创建一个 19x19 的标准化网格，用于与检测到的棋子匹配
        """
        if self.grid_points is None or len(self.grid_points) == 0:
            return False

        # 获取网格边界
        points = np.array(self.grid_points)
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)

        # 在边界内插值生成标准网格坐标
        self.standard_grid = []
        for i in range(19):
            for j in range(19):
                standard_x = min_x + (max_x - min_x) * (j / 18.0)
                standard_y = min_y + (max_y - min_y) * (i / 18.0)
                self.standard_grid.append((standard_x, standard_y, i, j))  # (x, y, 行, 列)

        return True

    def align_frame(self, frame):
        """
        使用特征点匹配对齐当前图像到参考图像
        Args:
            frame: 当前帧图像
        Returns:
            aligned_frame: 对齐后的图像
            homography_matrix: 单应矩阵
        """
        if self.reference_image is None:
            return frame, None

        # 转灰度图
        ref_gray = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 使用 ORB 检测特征点
        orb = cv2.ORB_create(nfeatures=2000)
        kp1, des1 = orb.detectAndCompute(ref_gray, None)
        kp2, des2 = orb.detectAndCompute(frame_gray, None)

        if des1 is None or des2 is None:
            return frame, None

        # 特征匹配（汉明距离）
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # 取前50个最佳匹配
        good_matches = matches[:50]

        if len(good_matches) < 10:
            return frame, None

        # 提取匹配点位置
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 计算单应矩阵
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        if H is None:
            return frame, None

        # 应用变换进行图像对齐
        h, w = self.reference_image.shape[:2]
        aligned_frame = cv2.warpPerspective(frame, H, (w, h))

        return aligned_frame, H

    def get_nearest_grid_point(self, stone_center):
        """
        查找离检测到的棋子中心最近的网格点
        Args:
            stone_center: 棋子中心点坐标 (x, y)
        Returns:
            grid_point: 最近的网格点 (row, col)
        """
        if self.standard_grid is None:
            return None

        min_dist = float('inf')
        nearest_point = None
        for x, y, row, col in self.standard_grid:
            dist = np.sqrt((x - stone_center[0]) ** 2 + (y - stone_center[1]) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest_point = (row, col)

        return nearest_point

    def _line_intersection(self, line1, line2):
        """
        计算两条线段的交点
        Returns:
            若相交则返回交点坐标 (x, y)，否则返回 None
        """
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if denom == 0:
            return None  # 平行

        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom

        if 0 <= ua <= 1 and 0 <= ub <= 1:
            x = x1 + ua * (x2 - x1)
            y = y1 + ua * (y2 - y1)
            return (x, y)

        return None

    def _sort_points_to_grid(self, points):
        """
        将网格交点按照从上到下，从左到右排序成网格结构
        """
        if len(points) < 300:
            return points  # 太少无法排序为完整网格

        points = np.array(points)
        sorted_by_y = points[np.argsort(points[:, 1])]

        try:
            from sklearn.cluster import KMeans
            n_rows = min(19, len(sorted_by_y) // 15)
            kmeans = KMeans(n_clusters=n_rows).fit(sorted_by_y[:, 1].reshape(-1, 1))
            rows = [[] for _ in range(n_rows)]
            for i, label in enumerate(kmeans.labels_):
                rows[label].append(sorted_by_y[i])
        except ImportError:
            n_rows = min(19, len(sorted_by_y) // 15)
            rows = [[] for _ in range(n_rows)]
            points_per_row = len(sorted_by_y) // n_rows
            for i, point in enumerate(sorted_by_y):
                row_idx = min(i // points_per_row, n_rows - 1)
                rows[row_idx].append(point)

        # 每行按 x 排序
        sorted_grid = []
        for row in rows:
            row = sorted(row, key=lambda p: p[0])
            sorted_grid.extend(row)

        return sorted_grid

    def visualize_grid_points(self, image=None):
        """
        在图像上可视化提取到的网格交点
        Args:
            image: 要绘制的图像（可选，默认使用参考图像）
        Returns:
            标记了交点的图像
        """
        if image is None and self.reference_image is None:
            return None

        viz_image = self.reference_image.copy() if image is None else image.copy()

        if self.grid_points is not None:
            for x, y in self.grid_points:
                cv2.circle(viz_image, (int(x), int(y)), 3, (0, 255, 0), -1)

        if self.standard_grid is not None:
            for x, y, row, col in self.standard_grid:
                cv2.circle(viz_image, (int(x), int(y)), 2, (0, 0, 255), -1)

        return viz_image

    
# def test_reference_board_manager(image_path):
#         """
#         测试 ReferenceBoardManager 类的功能：
#         1. 加载图像
#         2. 提取参考网格
#         3. 可视化网格点
#         4. 显示结果

#         Args:
#             image_path: 棋盘参考图像的路径（清晰无遮挡的空棋盘）
#         """
#         # 创建参考棋盘管理器实例
#         manager = ReferenceBoardManager()

#         # 加载参考图像
#         image = cv2.imread(image_path)
#         if image is None:
#             print("无法加载图像，请检查路径是否正确。")
#             return

#         # 捕获参考图像并提取网格点
#         success = manager.capture_reference(image)
#         if not success:
#             print("参考图像网格提取失败，请确保图像清晰、无遮挡并包含完整棋盘。")
#             return
#         else:
#             print("参考图像网格提取成功。")

#         # 可视化提取的网格点
#         result_image = manager.visualize_grid_points()

#         # 显示结果
#         cv2.imshow("Reference Grid Points", result_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

# # 示例用法：替换为你的参考图像路径
# if __name__ == "__main__":
#         test_reference_board_manager("Test_Image\image.png")  # 例如一个清晰的空棋盘照片