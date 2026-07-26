import cv2
import os
import queue
import threading
import time
import numpy as np

class FrameExtractor:
    def __init__(self, output_dir="./Captured_Image"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)  # 确保存储目录存在
        self.frame_queue = queue.Queue(maxsize=30)  # 限制队列大小以控制内存使用
        self.is_extracting = False
        self.extraction_complete = False
        
        # 帧稳定性分析参数
        self.diff_threshold = 5.0  # 帧间差异阈值（平均灰度差）
        self.hand_model = None  # YOLO手部检测模型
        self.frame_buffer = []  # 存储连续3帧进行分析
    
    def start_extraction(self, video_path):
        """开始异步提取视频帧到队列中"""
        self.is_extracting = True
        self.extraction_complete = False
        self.frame_queue = queue.Queue(maxsize=30)
        self.frame_buffer = []  # 重置帧缓冲区
        
        # 启动提取线程
        extraction_thread = threading.Thread(
            target=self._extract_frames_to_queue, 
            args=(video_path,)
        )
        extraction_thread.daemon = True
        extraction_thread.start()
        
    def _extract_frames_to_queue(self, video_path):
        """将视频帧提取到队列中，每隔5帧取一帧，连续取3帧进行分析"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"无法打开视频: {video_path}")
                self.extraction_complete = True
                self.is_extracting = False
                return
                
            # 帧提取逻辑
            frame_count = 0
            saved_frame_count = 1
            frame_stride = 10  # 每5帧取1帧
            
            # 存储连续3帧用于稳定性分析
            stable_frames = []
            
            while self.is_extracting:
                ret, frame = cap.read()
                if not ret:
                    print("视频帧提取完成")
                    break

                # 每隔5帧采样一次
                if frame_count % frame_stride == 0:
                    # 保存当前帧用于分析
                    frame_path = f"{self.output_dir}/temp_{len(stable_frames)}.jpg"
                    cv2.imwrite(frame_path, frame)
                    stable_frames.append({"frame": frame, "path": frame_path})
                    
                    # 当收集到3帧后，分析稳定性
                    if len(stable_frames) == 3:
                        is_stable, has_hand = self._analyze_frame_stability(stable_frames)
                        
                        if is_stable and not has_hand:
                            # 如果稳定且没有手部遮挡，使用最后一帧
                            last_frame = stable_frames[2]["frame"]
                            frame_path = f"{self.output_dir}/{saved_frame_count}.jpg"
                            cv2.imwrite(frame_path, last_frame)
                            
                            frame_info = {
                                "frame": last_frame,
                                "path": frame_path,
                                "frame_number": saved_frame_count,
                                "is_stable": True
                            }
                            
                            self.frame_queue.put(frame_info, block=True)
                            print(f"提取到稳定帧 {saved_frame_count} 并加入队列")
                            saved_frame_count += 1
                        
                        # 移除第一帧，保留后两帧，为下一轮分析准备
                        stable_frames = stable_frames[1:]
                
                frame_count += 1
            
            cap.release()
        except Exception as e:
            print(f"视频帧提取出错: {e}")
        finally:
            self.extraction_complete = True
            self.is_extracting = False
    
    def _analyze_frame_stability(self, frames):
        """分析连续3帧的稳定性和手部遮挡
        
        Returns:
            tuple: (is_stable, has_hand_occlusion)
        """
        if len(frames) < 3:
            return False, True
        
        # 计算帧间差异
        diffs = []
        for i in range(len(frames)-1):
            diff = self._calculate_frame_difference(frames[i]["frame"], frames[i+1]["frame"])
            diffs.append(diff)
        
        # 检查是否有手部遮挡
        has_hand = False
        for frame in frames:
            if self._detect_hand_occlusion(frame["frame"]):
                has_hand = True
                break
        
        # 判断稳定性：所有帧间差异都低于阈值
        is_stable = all(diff < self.diff_threshold for diff in diffs)
        
        return is_stable, has_hand
    
    def _calculate_frame_difference(self, frame1, frame2):
        """计算两帧之间的差异（平均灰度差异）"""
        # 转为灰度图进行比较
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # 计算绝对差值
        diff = cv2.absdiff(gray1, gray2)
        
        # 计算平均差异
        mean_diff = np.mean(diff)
        return mean_diff
    
    def _detect_hand_occlusion(self, frame):
        """使用YOLO模型检测图像中是否有手部遮挡"""
        # 如果没有手部检测模型，则默认视为无手部遮挡
        if self.hand_model is None:
            return False
        
        try:
            # 使用YOLO检测手部或人
            results = self.hand_model(frame)
            
            # 检查是否有检测结果
            for result in results:
                for box in result.boxes:
                    # 获取置信度
                    confidence = float(box.conf[0])
                    
                    # 如果置信度足够高，判断为有手部
                    if confidence > 0.3:  # 置信度阈值
                        # 计算检测框面积占比
                        x, y, w, h = box.xywh[0]
                        box_area = w.item() * h.item()
                        frame_area = frame.shape[0] * frame.shape[1]
                        area_ratio = box_area / frame_area
                        
                        # 如果检测区域够大，认为有遮挡
                        if area_ratio > 0.05:  # 面积比例阈值
                            return True
            
            # 未检测到手部
            return False
        except Exception as e:
            print(f"手部检测出错: {e}")
            return False  # 出错时保守处理，视为没有手部
    
    def get_next_frame(self, timeout=1.0):
        """获取下一个可用的视频帧，如果没有更多帧则返回None"""
        try:
            # 非阻塞获取，避免无限等待
            frame_info = self.frame_queue.get(timeout=timeout)
            return frame_info
        except queue.Empty:
            # 如果提取已完成且队列为空，说明没有更多帧
            if self.extraction_complete:
                return None
            return None
        
    def stop_extraction(self):
        """停止帧提取过程"""
        self.is_extracting = False