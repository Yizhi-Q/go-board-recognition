import cv2
import numpy as np
import os
import time
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import json
from datetime import datetime
from pathlib import Path

from Chess_Recognition.Go_Board_Manage import *
from Preprocess.Capture_Video_Frames import FrameExtractor
from Preprocess.Board_Image_Correction import warp_perspective
from Chess_Recognition.Yolo_To_SGF import detect_stones, apply_nms, get_board_grid_positions, map_stones_to_grid, generate_sgf, generate_board_image, configure_detector

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_300epoch_onlybw.pt"
DEFAULT_HAND_MODEL_PATH = PROJECT_ROOT / "models" / "hand_s.pt"

class GoGameProcessor:
    def __init__(self, model_path=None,
                 hand_model_path=None,
                 temp_dir="./temp", output_dir="./output"):
        """
        围棋对局处理器主类
        
        参数:
            model_path: YOLO模型路径
            temp_dir: 临时文件存储目录
            output_dir: 输出文件存储目录
        """
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.hand_model_path = Path(hand_model_path) if hand_model_path else DEFAULT_HAND_MODEL_PATH
        self.temp_dir = temp_dir
        self.output_dir = output_dir

        # 按照棋局记录的时间创建子目录
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.output_dir = os.path.join(output_dir, timestamp)
        self.temp_dir = os.path.join(temp_dir, timestamp)

        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 确保目录存在
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载YOLO模型，默认使用CUDA
        from ultralytics import YOLO
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Stone-detection model not found: {self.model_path}. "
                "See models/README.md for setup instructions."
            )
        self.model = YOLO(str(self.model_path)).to('cuda' if self.is_cuda_available() else 'cpu')
        configure_detector(self.model)
        
        self.hand_model = None
        if self.hand_model_path.exists():
            try:
                self.hand_model = YOLO(str(self.hand_model_path)).to('cuda' if self.is_cuda_available() else 'cpu')
                print(f"已加载手部检测模型: {self.hand_model_path}")
            except Exception as e:
                print(f"加载手部检测模型失败: {e}")
        
        # 创建帧提取器
        self.frame_extractor = FrameExtractor(output_dir=temp_dir)
        self.frame_extractor.hand_model = self.hand_model
        
        # 初始化对局历史记录
        self.move_history = []
        self.board_states = []
        self.current_game_info = {
            "start_time": None,
            "end_time": None,
            "black_stones": 0,
            "white_stones": 0,
            "komi": 6.5
        }
        
        # 初始化棋盘状态
        self.current_board = [[-1 for _ in range(19)] for _ in range(19)]
        
        # 处理状态
        self.is_processing = False
        self.current_move = 0
        
        # 用于存储处理状态的变量
        self.processed_frames = 0
        self.total_frames = 0
        self.frames_with_board = 0

        self.reference_board_manager = ReferenceBoardManager()
        self.has_reference_board = False
        self.max_ref_attempts = 50  # 最多尝试x帧来获取参考棋盘
    
    def is_cuda_available(self):
        """检查是否可以使用CUDA"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def start_new_game(self):
        """开始新的对局记录"""
        self.move_history = []
        self.board_states = []
        self.current_board = [[-1 for _ in range(19)] for _ in range(19)]
        self.current_move = 0
        self.current_game_info = {
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": None,
            "black_stones": 0,
            "white_stones": 0,
            "komi": 6.5
        }
        
        # 记录初始状态
        self.board_states.append(self._deep_copy_board(self.current_board))
        
        # 重置处理状态
        self.processed_frames = 0
        self.total_frames = 0
        self.frames_with_board = 0
        
        return self.current_game_info
    
    def end_game(self):
        """结束当前对局记录"""
        if self.current_game_info["start_time"] is None:
            return None
            
        self.current_game_info["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 统计最终棋子数量
        black_count = sum(row.count(0) for row in self.current_board)
        white_count = sum(row.count(2) for row in self.current_board)
        
        self.current_game_info["black_stones"] = black_count
        self.current_game_info["white_stones"] = white_count
        
        # 保存对局信息和历史记录
        game_id = datetime.now().strftime("%Y%m%d%H%M%S")
        game_data = {
            "game_info": self.current_game_info,
            "move_history": self.move_history,
            "final_board": self.current_board,
            "processing_stats": {
                "total_frames": self.total_frames,
                "processed_frames": self.processed_frames,
                "frames_with_board": self.frames_with_board,
                "used_reference_board": self.has_reference_board
            }
        }
        
        with open(f"{self.output_dir}/game_{game_id}.json", "w") as f:
            json.dump(game_data, f, indent=2)
        
        return self.current_game_info
    
    def process_video(self, video_path, time_interval=1, callback=None, progress_callback=None):
        """处理视频文件，提取帧并检测棋盘变化"""
        self.is_processing = True
        
        # 启动帧提取器
        self.frame_extractor.start_extraction(video_path)
        
        # 处理提取的帧
        prev_board = self._deep_copy_board(self.current_board)
        self.processed_frames = 0
        self.total_frames = 0
        self.frames_with_board = 0
        
        # 参考棋盘捕获状态
        ref_attempt_count = 0
        
        # 处理循环
        while self.is_processing:
            # 获取下一帧
            frame_info = self.frame_extractor.get_next_frame(timeout=1.0)
            
            # 如果没有更多帧且提取已完成，退出循环
            if frame_info is None and self.frame_extractor.extraction_complete:
                break
            
            # 如果暂时没有帧，等待一下再尝试
            if frame_info is None:
                time.sleep(0.1)
                continue
            
            self.total_frames += 1
            
            # 更新进度
            if progress_callback:
                # 尝试估计总帧数
                progress = {
                    "current": self.processed_frames,
                    "total": self.total_frames,
                    "status": f"已处理 {self.processed_frames} 帧，检测到棋盘 {self.frames_with_board} 帧"
                }
                progress_callback(progress)
            
            # 处理当前帧
            try:
                print(f"处理帧 {frame_info['frame_number']}")
                
                # 获取帧和路径
                frame = frame_info['frame']
                frame_path = frame_info['path']
                
                # 尝试矫正棋盘图像
                corrected_image = warp_perspective(frame_path)
                corrected_path = f"{self.temp_dir}/corrected_{os.path.basename(frame_path)}"
                cv2.imwrite(corrected_path, corrected_image)
                
                # 成功矫正，检测棋子
                self.frames_with_board += 1
                
                # 如果尚未获取参考棋盘且处于对局初始阶段（前几帧），尝试捕获参考棋盘
                if not self.has_reference_board and self.frames_with_board > 3 :
                    ref_attempt_count += 1
                    print(f"尝试捕获参考棋盘 (尝试 {ref_attempt_count}/{self.max_ref_attempts})...")
                    
                    if self.capture_reference_board(corrected_image):
                        print("参考棋盘捕获成功！")
                    else:
                        print(f"参考棋盘捕获失败，将在后续帧再次尝试")
                
                # 如果有参考棋盘，将当前帧与参考帧对齐
                aligned_image = corrected_image
                if self.has_reference_board:
                    aligned_image, H = self.reference_board_manager.align_frame(corrected_image)
                    if H is not None:
                        aligned_path = f"{self.temp_dir}/aligned_{os.path.basename(frame_path)}"
                        cv2.imwrite(aligned_path, aligned_image)
                        print(f"帧已与参考棋盘对齐")
                    else:
                        print(f"无法对齐帧，使用原始矫正图像")
                
                # 检测棋子
                detections, _ = detect_stones(aligned_image, conf_threshold=0.5)
                
                # 使用不同的方法进行棋子到网格的映射
                stone_positions = []
                
                if self.has_reference_board:
                    # 使用参考棋盘的网格点进行映射
                    for (x, y, w, h, label, conf) in detections:
                        # 获取最近的网格交叉点
                        nearest_point = self.reference_board_manager.get_nearest_grid_point((x, y))
                        if nearest_point:
                            i, j = nearest_point
                            stone_positions.append((i, j, label))
                else:
                    # 使用原始方法
                    grid_positions = get_board_grid_positions(board_size=19, image_shape=aligned_image.shape)
                    stone_positions = map_stones_to_grid(detections, grid_positions)
                
                # 更新棋盘状态
                new_board = self._deep_copy_board(self.current_board)
                for i, j, label in stone_positions:
                    if 0 <= i < 19 and 0 <= j < 19:
                        new_board[i][j] = label
                
                # 检测变化并记录新的落子
                changes = self._detect_board_changes(prev_board, new_board)
                
                if changes:
                    for change in changes:
                        i, j, label = change
                        move_info = {
                            "move_number": self.current_move + 1,
                            "position": (i, j),
                            "color": "black" if label == 0 else "white",
                            "sgf_coord": chr(ord('a') + j) + chr(ord('a') + i),
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "frame_number": frame_info['frame_number']
                        }
                        
                        self.move_history.append(move_info)
                        self.current_move += 1
                        
                    # 更新当前棋盘状态
                    self.current_board = new_board
                    self.board_states.append(self._deep_copy_board(new_board))
                    
                    # 生成当前状态的棋盘图像
                    board_image = generate_board_image(stone_positions)
                    board_image_path = f"{self.output_dir}/board_state_{self.current_move}.png"
                    cv2.imwrite(board_image_path, board_image)
                    
                    # 保存当前帧的原始图像和矫正后的图像
                    original_copy_path = f"{self.output_dir}/original_move_{self.current_move}.jpg"
                    corrected_copy_path = f"{self.output_dir}/corrected_move_{self.current_move}.jpg"
                    cv2.imwrite(original_copy_path, frame)
                    cv2.imwrite(corrected_copy_path, corrected_image)
                    
                    # 如果有使用对齐，保存对齐后的图像
                    if self.has_reference_board:
                        aligned_copy_path = f"{self.output_dir}/aligned_move_{self.current_move}.jpg"
                        cv2.imwrite(aligned_copy_path, aligned_image)
                    
                    # 如果有回调函数，通知UI更新
                    if callback:
                        callback(self.current_move, board_image_path, changes)
                
                prev_board = self._deep_copy_board(new_board)
                
            except Exception as e:
                print(f"处理帧时出错: {e}")
                print(f"棋盘矫正失败或当前帧无棋盘，跳过当前帧")
            finally:
                self.processed_frames += 1
        
        # 停止帧提取
        self.frame_extractor.stop_extraction()
        
        # 生成SGF文件
        all_stones = []
        for i in range(19):
            for j in range(19):
                if self.current_board[i][j] in [0, 2]:
                    all_stones.append((i, j, self.current_board[i][j]))
        
        sgf_path = f"{self.output_dir}/game_{datetime.now().strftime('%Y%m%d%H%M%S')}.sgf"
        generate_sgf(all_stones, output_file=sgf_path)
        
        self.is_processing = False
        return self.move_history, self.board_states, sgf_path
    
    def stop_processing(self):
        """停止视频处理"""
        self.is_processing = False
        self.frame_extractor.stop_extraction()
    
    def get_move_history(self):
        """获取棋局的完整历史记录"""
        return self.move_history
    
    def get_board_state(self, move_number=None):
        """获取指定手数的棋盘状态"""
        if move_number is None or move_number >= len(self.board_states):
            return self.current_board
        
        return self.board_states[move_number]
    
    def export_sgf(self, output_path=None):
        """导出当前棋局的SGF文件"""
        if output_path is None:
            output_path = f"{self.output_dir}/game_{datetime.now().strftime('%Y%m%d%H%M%S')}.sgf"
        
        all_stones = []
        for i in range(19):
            for j in range(19):
                if self.current_board[i][j] in [0, 2]:
                    all_stones.append((i, j, self.current_board[i][j]))
        
        generate_sgf(all_stones, output_file=output_path)
        return output_path
    
    def _deep_copy_board(self, board):
        """深拷贝棋盘状态"""
        return [row[:] for row in board]
    
    def _detect_board_changes(self, prev_board, new_board):
        """检测两个棋盘状态之间的变化"""
        changes = []
        
        # 如果是第一次检测，初始化颜色变化历史记录
        if not hasattr(self, 'color_change_history'):
            self.color_change_history = [[[] for _ in range(19)] for _ in range(19)]
        
        for i in range(19):
            for j in range(19):
                # 如果有新增的棋子（空位变成有棋子）- 直接接受变化
                if prev_board[i][j] == -1 and new_board[i][j] in [0, 2]:
                    changes.append((i, j, new_board[i][j]))
                    # 更新历史记录，但不影响当前决策
                    self.color_change_history[i][j].append(new_board[i][j])
                    if len(self.color_change_history[i][j]) > 5:
                        self.color_change_history[i][j].pop(0)
                        
                # 如果有棋子被移除（提子）- 需要连续三次验证
                elif prev_board[i][j] in [0, 2] and new_board[i][j] == -1:
                    # 对于提子，需要连续判断以避免误检
                    self.color_change_history[i][j].append(-1)  # -1表示空点
                    if len(self.color_change_history[i][j]) > 5:
                        self.color_change_history[i][j].pop(0)
                    
                    if len(self.color_change_history[i][j]) >= 3 and all(c == -1 for c in self.color_change_history[i][j][-3:]):
                        # 连续三次检测到提子，认为是真实提子
                        changes.append((i, j, -1))
                        
                # 如果棋子颜色变化（例如黑子变白子）- 需要连续三次验证
                elif prev_board[i][j] != -1 and new_board[i][j] != -1 and prev_board[i][j] != new_board[i][j]:
                    # 记录新的颜色
                    self.color_change_history[i][j].append(new_board[i][j])
                    if len(self.color_change_history[i][j]) > 5:
                        self.color_change_history[i][j].pop(0)
                    
                    # 需要连续3次检测到相同的新颜色才认为有效
                    if len(self.color_change_history[i][j]) >= 3 and all(c == new_board[i][j] for c in self.color_change_history[i][j][-3:]):
                        changes.append((i, j, new_board[i][j]))
                
                # 如果检测到的棋子颜色与之前相同，只更新历史记录
                elif new_board[i][j] != -1:
                    self.color_change_history[i][j].append(new_board[i][j])
                    if len(self.color_change_history[i][j]) > 5:
                        self.color_change_history[i][j].pop(0)
        
        return changes
    
    def capture_reference_board(self, image):
        """
        捕获参考棋盘图像并提取交叉点
        
        参数:
            image: 清晰的棋盘图像（最好是空棋盘或棋子很少的状态）
            
        返回:
            bool: 参考棋盘是否成功处理
        """
        success = self.reference_board_manager.capture_reference(image)
        self.has_reference_board = success
        
        if success:
            # 保存参考棋盘图像和处理结果
            ref_output_path = f"{self.output_dir}/reference_board.jpg"
            cv2.imwrite(ref_output_path, image)
            
            # 保存可视化结果
            viz_image = self.reference_board_manager.visualize_grid_points()
            viz_output_path = f"{self.output_dir}/reference_grid_points.jpg"
            cv2.imwrite(viz_output_path, viz_image)
            
            print(f"参考棋盘已保存: {ref_output_path}")
            
        return success
