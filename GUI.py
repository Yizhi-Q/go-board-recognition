import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import cv2
import json
from PIL import Image, ImageTk
import time
from datetime import datetime
from GBR import GoGameProcessor
from Chess_Recognition.Yolo_To_SGF import *
from Chess_Recognition.End_Game_Decision import *

#! 识别逻辑优化
#! 识别准确度优化

class GoGameUI:
    def __init__(self, root,hand_model_path=None):
        self.root = root
        self.root.title("围棋记谱系统")
        self.root.geometry("1100x780")
        
        # 设置主处理器
        self.processor = GoGameProcessor(hand_model_path=hand_model_path)
        
        # 视频处理线程
        self.processing_thread = None
        
        # 当前显示的手数
        self.current_display_move = 0
        
        # 历史对局列表
        self.game_history = []
        self.load_game_history()
        
        # 创建UI组件
        self._create_ui_components()
        
        # 更新状态
        self.update_status("就绪")
    
    def _create_ui_components(self):
        """创建UI组件"""
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 上部分：左右控制面板
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 左侧控制面板（开始记谱、结束记谱）
        left_control = ttk.LabelFrame(control_frame, text="记谱控制")
        left_control.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        self.start_btn = ttk.Button(left_control, text="开始记谱", command=self.start_recording)
        self.start_btn.grid(row=0, column=0, padx=5, pady=5)
        
        self.stop_btn = ttk.Button(left_control, text="结束记谱", command=self.stop_recording, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=1, padx=5, pady=5)
        
        self.recording_label = ttk.Label(left_control, text="记谱中...", foreground="red")
        self.recording_label.grid(row=0, column=4, padx=5, pady=5)
        self.recording_label.grid_remove()  # 初始隐藏
        
        # 右侧控制面板（棋子数量信息）
        right_control = ttk.LabelFrame(control_frame, text="对局信息")
        right_control.pack(side=tk.RIGHT, padx=5, pady=5, fill=tk.X, expand=True)
        
        ttk.Label(right_control, text="当前目数:").grid(row=0, column=0, padx=5, pady=5)
        self.score_label = ttk.Label(right_control, text="")
        self.score_label.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(right_control, text="黑子:").grid(row=0, column=2, padx=5, pady=5)
        self.black_count = ttk.Label(right_control, text="0")
        self.black_count.grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Label(right_control, text="白子:").grid(row=0, column=4, padx=5, pady=5)
        self.white_count = ttk.Label(right_control, text="0")
        self.white_count.grid(row=0, column=5, padx=5, pady=5)
        
        ttk.Label(right_control, text="贴目:").grid(row=0, column=6, padx=5, pady=5)
        self.komi_var = tk.StringVar(value="6.5")
        self.komi_entry = ttk.Entry(right_control, textvariable=self.komi_var, width=4)
        self.komi_entry.grid(row=0, column=7, padx=5, pady=5)
        
        # 胜负结果显示
        ttk.Label(right_control, text="胜负:").grid(row=0, column=8, padx=5, pady=5)
        self.result_label = ttk.Label(right_control, text="--", foreground="blue", font=("Arial", 10, "bold"))
        self.result_label.grid(row=0, column=9, padx=5, pady=5)
        
        # 中间部分：棋盘和侧边栏
        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左侧：棋盘显示
        self.board_frame = ttk.LabelFrame(middle_frame, text="棋盘")
        self.board_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.board_canvas = tk.Canvas(self.board_frame, bg="#E6C079", width=500, height=500)
        self.board_canvas.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        
        # 初始化一个空棋盘
        self._draw_empty_board()
        
        # 创建一个水平分割的右侧容器
        right_container = ttk.Frame(middle_frame)
        right_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 右侧左方：SGF显示区域
        sgf_frame = ttk.LabelFrame(right_container, text="SGF棋谱")
        sgf_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 添加滚动条到SGF文本区域
        sgf_scroll = ttk.Scrollbar(sgf_frame)
        sgf_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.sgf_text = tk.Text(sgf_frame, height=25, width=30, font=("Courier", 9), yscrollcommand=sgf_scroll.set)
        self.sgf_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        sgf_scroll.config(command=self.sgf_text.yview)
        
        # 配置SGF文本样式
        self.sgf_text.tag_configure("highlight", background="yellow", foreground="red")
        
        # 右侧右方：功能区和历史记录
        control_sidebar = ttk.Frame(right_container)
        control_sidebar.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 右侧右上方：棋局历史
        history_frame = ttk.LabelFrame(control_sidebar, text="棋局历史")
        history_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.history_frame = ttk.Frame(history_frame)
        self.history_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        history_scroll = ttk.Scrollbar(self.history_frame)
        history_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.history_listbox = tk.Listbox(self.history_frame, yscrollcommand=history_scroll.set)
        self.history_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        history_scroll.config(command=self.history_listbox.yview)
        
        self.history_listbox.bind('<<ListboxSelect>>', self.on_history_select)
        
        # 右侧右下方：功能区
        functions_frame = ttk.LabelFrame(control_sidebar, text="功能区")
        functions_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 视频上传部分
        ttk.Label(functions_frame, text="视频处理").pack(anchor=tk.W, padx=5, pady=5)
        
        video_frame = ttk.Frame(functions_frame)
        video_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.video_path_var = tk.StringVar()
        self.video_path_entry = ttk.Entry(video_frame, textvariable=self.video_path_var, width=20)
        self.video_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5))
        
        self.browse_btn = ttk.Button(video_frame, text="浏览", command=self.browse_video)
        self.browse_btn.pack(side=tk.RIGHT)
        
        process_frame = ttk.Frame(functions_frame)
        process_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.process_btn = ttk.Button(process_frame, text="处理视频", command=self.process_video)
        self.process_btn.pack(side=tk.RIGHT)
        
        # 进度条
        progress_frame = ttk.Frame(functions_frame)
        progress_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", length=200, mode="determinate")
        self.progress_bar.pack(fill=tk.X, expand=True, padx=5, pady=5)
        
        self.progress_label = ttk.Label(progress_frame, text="")
        self.progress_label.pack(pady=2)
        
        # 终局判断按钮
        self.calculate_btn = ttk.Button(functions_frame, text="计算胜负", command=self.calculate_result)
        self.calculate_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # 胜负详情按钮
        self.show_details_btn = ttk.Button(functions_frame, text="查看详细结果", command=self.show_result_details)
        self.show_details_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # 棋局导出部分
        ttk.Separator(functions_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5, pady=10)
        
        export_frame = ttk.Frame(functions_frame)
        export_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.export_sgf_btn = ttk.Button(export_frame, text="导出SGF", command=self.export_sgf)
        self.export_sgf_btn.pack(side=tk.LEFT, padx=(0,5))
        
        self.export_img_btn = ttk.Button(export_frame, text="导出图像", command=self.export_image)
        self.export_img_btn.pack(side=tk.RIGHT)
        
        # 底部：状态栏和历史浏览
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(bottom_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 历史浏览控制
        history_controls = ttk.Frame(bottom_frame)
        history_controls.pack(fill=tk.X, pady=5)
        
        self.first_move_btn = ttk.Button(history_controls, text="<<", command=lambda: self.navigate_moves("first"))
        self.first_move_btn.pack(side=tk.LEFT, padx=2)
        
        self.prev_move_btn = ttk.Button(history_controls, text="<", command=lambda: self.navigate_moves("prev"))
        self.prev_move_btn.pack(side=tk.LEFT, padx=2)
        
        self.move_var = tk.StringVar(value="第 0 手")
        self.move_label = ttk.Label(history_controls, textvariable=self.move_var, width=9)
        self.move_label.pack(side=tk.LEFT, padx=5)
        
        # 添加落子位置显示
        ttk.Label(history_controls, text="|").pack(side=tk.LEFT)
        self.position_var = tk.StringVar(value="位置: --")
        self.position_label = ttk.Label(history_controls, textvariable=self.position_var, width=8)
        self.position_label.pack(side=tk.LEFT, padx=5)
        
        # 添加当前落子颜色显示
        ttk.Label(history_controls, text="|").pack(side=tk.LEFT)
        self.stone_color_var = tk.StringVar(value="颜色: --")
        self.stone_color_label = ttk.Label(history_controls, textvariable=self.stone_color_var)
        self.stone_color_label.pack(side=tk.LEFT, padx=5)

        self.next_move_btn = ttk.Button(history_controls, text=">", command=lambda: self.navigate_moves("next"))
        self.next_move_btn.pack(side=tk.LEFT, padx=2)
        
        self.last_move_btn = ttk.Button(history_controls, text=">>", command=lambda: self.navigate_moves("last"))
        self.last_move_btn.pack(side=tk.LEFT, padx=2)
        
        # 存储结果数据
        self.result_data = None
        
        # 更新历史对局列表
        self.update_history_list()
        
        # 初始化SGF显示
        self.update_sgf_display()
    
    def _draw_empty_board(self):
        """绘制空棋盘"""
        # 清空画布
        self.board_canvas.delete("all")
        
        # 获取画布尺寸
        canvas_width = self.board_canvas.winfo_width()
        canvas_height = self.board_canvas.winfo_height()
        
        # 如果画布尺寸为1，使用默认值
        if canvas_width <= 1:
            canvas_width = 500
        if canvas_height <= 1:
            canvas_height = 500
        
        # 计算棋盘大小和边距
        board_size = min(canvas_width, canvas_height) - 40
        margin = 20
        grid_size = board_size / 18
        
        # 绘制棋盘网格
        for i in range(19):
            # 水平线
            self.board_canvas.create_line(
                margin, margin + i * grid_size,
                margin + board_size, margin + i * grid_size,
                fill="black", width=1
            )
            # 垂直线
            self.board_canvas.create_line(
                margin + i * grid_size, margin,
                margin + i * grid_size, margin + board_size,
                fill="black", width=1
            )
        
        # 绘制星位点
        for x in [3, 9, 15]:
            for y in [3, 9, 15]:
                self.board_canvas.create_oval(
                    margin + x * grid_size - 3, margin + y * grid_size - 3,
                    margin + x * grid_size + 3, margin + y * grid_size + 3,
                    fill="black"
                )
    
    def update_board_display(self, board_state=None, highlight_move=None):
        """更新棋盘显示"""
        # 如果没有提供棋盘状态，使用当前处理器中的状态
        if board_state is None:
            board_state = self.processor.get_board_state(self.current_display_move)
        
        # 清空画布并重绘空棋盘
        self._draw_empty_board()
        
        # 获取画布尺寸
        canvas_width = self.board_canvas.winfo_width()
        canvas_height = self.board_canvas.winfo_height()
        
        # 如果画布尺寸为1，使用默认值
        if canvas_width <= 1:
            canvas_width = 500
        if canvas_height <= 1:
            canvas_height = 500
        
        # 计算棋盘大小和边距
        board_size = min(canvas_width, canvas_height) - 40
        margin = 20
        grid_size = board_size / 18
        stone_radius = grid_size / 2 - 1
        
        # 黑白子计数
        black_count = 0
        white_count = 0
        
        # 绘制棋子
        for i in range(19):
            for j in range(19):
                stone_type = board_state[i][j]
                if stone_type == 0 or stone_type == 2:  # 0=黑子, 2=白子
                    x = margin + j * grid_size
                    y = margin + i * grid_size
                    
                    # 绘制棋子
                    if stone_type == 0:  # 黑子
                        self.board_canvas.create_oval(
                            x - stone_radius, y - stone_radius,
                            x + stone_radius, y + stone_radius,
                            fill="black", outline="black"
                        )
                        black_count += 1
                    else:  # 白子
                        self.board_canvas.create_oval(
                            x - stone_radius, y - stone_radius,
                            x + stone_radius, y + stone_radius,
                            fill="white", outline="black"
                        )
                        white_count += 1
                    
                    # 高亮最新落子
                    if highlight_move and highlight_move[0] == i and highlight_move[1] == j:
                        self.board_canvas.create_rectangle(
                            x - stone_radius - 2, y - stone_radius - 2,
                            x + stone_radius + 2, y + stone_radius + 2,
                            outline="red", width=2
                        )
        
        # 更新棋子计数
        self.black_count.config(text=str(black_count))
        self.white_count.config(text=str(white_count))
        
        # 更新目数
        self.score_label.config(text=f"{black_count+white_count}")
    
    def calculate_result(self):
        """计算胜负结果"""
        if not self.processor.current_board or all(all(cell == -1 for cell in row) for row in self.processor.current_board):
            messagebox.showinfo("提示", "没有棋盘数据可供计算")
            return
        
        try:
            komi = float(self.komi_var.get())
        except ValueError:
            messagebox.showerror("错误", "贴目必须是一个有效的数字")
            return
        
        # 计算胜负
        game_result = GameResult(self.processor, komi)
        game_result.calculate()
        
        # 保存结果数据
        self.result_data = game_result.get_detailed_result()
        
        # 显示结果
        self.result_label.config(text=game_result.result)
        
        # 更新状态
        self.update_status(f"终局判断完成: {game_result.result}")
        
        return game_result
    
    def show_result_details(self):
        """显示胜负详情"""
        if not self.result_data:
            messagebox.showinfo("提示", "请先计算胜负结果")
            return
        
        # 创建详情窗口
        details_win = tk.Toplevel(self.root)
        details_win.title("胜负详情")
        details_win.geometry("400x300")
        details_win.resizable(False, False)
        
        # 详情内容
        content_frame = ttk.Frame(details_win, padding=20)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # 黑方信息
        ttk.Label(content_frame, text="黑方", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        ttk.Label(content_frame, text=f"棋子数: {self.result_data['black_stones']}").grid(row=1, column=0, sticky=tk.W)
        ttk.Label(content_frame, text=f"领地: {self.result_data['black_territory']}").grid(row=2, column=0, sticky=tk.W)
        ttk.Label(content_frame, text=f"总计: {self.result_data['black_total']}").grid(row=3, column=0, sticky=tk.W, pady=(0, 15))
        
        # 白方信息
        ttk.Label(content_frame, text="白方", font=("Arial", 12, "bold")).grid(row=4, column=0, sticky=tk.W, pady=(0, 10))
        ttk.Label(content_frame, text=f"棋子数: {self.result_data['white_stones']}").grid(row=5, column=0, sticky=tk.W)
        ttk.Label(content_frame, text=f"领地: {self.result_data['white_territory']}").grid(row=6, column=0, sticky=tk.W)
        ttk.Label(content_frame, text=f"贴目: {self.result_data['komi']}").grid(row=7, column=0, sticky=tk.W)
        ttk.Label(content_frame, text=f"总计: {self.result_data['white_total']}").grid(row=8, column=0, sticky=tk.W, pady=(0, 15))
        
        # 中立点
        ttk.Label(content_frame, text=f"中立点: {self.result_data['neutral_points']}").grid(row=9, column=0, sticky=tk.W, pady=(0, 15))
        
        # 结果
        result_label = ttk.Label(content_frame, text=f"结果: {self.result_data['result']}", 
                                 font=("Arial", 12, "bold"), foreground="blue")
        result_label.grid(row=10, column=0, sticky=tk.W)
        
        # 关闭按钮
        ttk.Button(content_frame, text="关闭", command=details_win.destroy).grid(row=11, column=0, pady=(15, 0))
    
    def browse_video(self):
        """浏览并选择视频文件"""
        video_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=(("视频文件", "*.mp4 *.avi *.mov"), ("所有文件", "*.*"))
        )
        if video_path:
            self.video_path_var.set(video_path)
    
    def process_video(self):
        """处理视频文件"""
        video_path = self.video_path_var.get()
        if not video_path or not os.path.exists(video_path):
            messagebox.showerror("错误", "请选择有效的视频文件")
            return
        
        # 禁用处理按钮，避免重复点击
        self.process_btn.config(state=tk.DISABLED)
        self.update_status("正在处理视频...")
        
        # 重置进度条
        self.progress_bar["value"] = 0
        self.progress_label.config(text="准备处理...")
        
        # 清除上次的结果
        self.result_label.config(text="--")
        self.result_data = None
        
        # 开始新游戏
        self.processor.start_new_game()
        
        # 在新线程中处理视频
        def process_thread():
            try:
                self.processor.process_video(
                    video_path, 
                    callback=self.update_from_processor,
                    progress_callback=self.update_progress
                )
                
                # 处理完成后在主线程更新UI
                self.root.after(0, self.process_complete)
            except Exception as e:
                # 修复作用域问题：使用默认参数捕获变量e
                error_msg = str(e)
                # self.root.after(0, lambda: self.show_error(f"处理视频时出错: {error_msg}"))
            finally:
                # 恢复处理按钮
                self.root.after(0, lambda: self.process_btn.config(state=tk.NORMAL))
        
        self.processing_thread = threading.Thread(target=process_thread)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def update_sgf_display(self):
        """更新SGF显示"""
        if not self.processor.board_states:
            self.sgf_text.delete("1.0", tk.END)
            self.sgf_text.insert(tk.END, "(;GM[1]FF[4]SZ[19]PB[Black]PW[White])")
            return
        
        # 生成当前状态的SGF
        sgf = "(;GM[1]FF[4]SZ[19]PB[Black]PW[White]\n"
        
        # 添加当前状态的所有棋子
        board = self.processor.get_board_state(self.current_display_move)
        for i in range(19):
            for j in range(19):
                stone_type = board[i][j]
                if stone_type == 0:  # 黑子
                    sgf += f";B[{chr(ord('a') + j)}{chr(ord('a') + i)}]\n"
                elif stone_type == 2:  # 白子
                    sgf += f";W[{chr(ord('a') + j)}{chr(ord('a') + i)}]\n"
        
        sgf += ")"
        
        # 更新SGF文本框
        self.sgf_text.delete("1.0", tk.END)
        self.sgf_text.insert(tk.END, sgf)
        
        # 高亮显示最后一步
        if self.current_display_move > 0 and len(self.processor.move_history) >= self.current_display_move:
            move_info = self.processor.move_history[self.current_display_move - 1]
            sgf_coord = move_info["sgf_coord"]
            color_code = "B" if move_info["color"] == "black" else "W"
            
            # 查找并高亮最后一步
            search_text = f";{color_code}[{sgf_coord}]"
            start_idx = "1.0"
            while True:
                start_idx = self.sgf_text.search(search_text, start_idx, tk.END)
                if not start_idx:
                    break
                end_idx = f"{start_idx}+{len(search_text)}c"
                self.sgf_text.tag_add("highlight", start_idx, end_idx)
                break
    
    def update_progress(self, progress_info):
        """更新进度条和进度信息"""
        self.root.after(0, lambda: self._update_progress_ui(progress_info))
    
    def _update_progress_ui(self, progress_info):
        """在主线程中更新进度UI"""
        current = progress_info.get("current", 0)
        total = progress_info.get("total", 100)
        status = progress_info.get("status", "")
        
        # 更新进度条
        if total > 0:
            progress_value = (current / total) * 100
            self.progress_bar["value"] = progress_value
        
        # 更新状态文本
        self.progress_label.config(text=status)
    
    def update_from_processor(self, move_number, board_image_path, changes):
        """处理器回调函数，用于更新UI"""
        self.root.after(0, lambda: self.update_ui_from_processor(move_number, board_image_path, changes))
    
    def update_ui_from_processor(self, move_number, board_image_path, changes):
        """在主线程中更新UI"""
        self.current_display_move = move_number
        self.move_var.set(f"第 {move_number} 手")
        
        # 更新棋盘显示
        if changes and len(changes) > 0:
            # 高亮显示最新的变化
            latest_change = changes[0]
            self.update_board_display(highlight_move=(latest_change[0], latest_change[1]))
            self.update_sgf_display()
            
            # 更新落子位置和颜色信息
            if move_number > 0 and move_number <= len(self.processor.move_history):
                move_info = self.processor.move_history[move_number - 1]
                # 转换位置为围棋坐标 (A-T除I, 1-19)
                x, y = move_info["position"]
                column = chr(ord('A') + (y if y < 8 else y + 1))  # 跳过"I"
                row = 19 - x  # 翻转行号
                position_str = f"{column}{row}"
                self.position_var.set(f"位置: {position_str}")
                
                # 更新颜色信息
                color_str = "黑" if move_info["color"] == "black" else "白"
                self.stone_color_var.set(f"颜色: {color_str}")
            else:
                # 重置信息
                self.position_var.set("位置: --")
                self.stone_color_var.set("颜色: --")
        else:
            self.update_board_display()
            # 重置信息
            self.position_var.set("位置: --")
            self.stone_color_var.set("颜色: --")
        
        # 更新状态
        self.update_status(f"已处理至第 {move_number} 手")
    
    def process_complete(self):
        """视频处理完成后的操作"""
        # 更新状态
        self.update_status("视频处理完成")
        
        # 更新信息显示
        game_info = self.processor.current_game_info
        self.black_count.config(text=str(game_info["black_stones"]))
        self.white_count.config(text=str(game_info["white_stones"]))
        self.score_label.config(text=str(game_info["black_stones"] + game_info["white_stones"]))
        
        # 更新进度显示
        self.progress_bar["value"] = 100
        self.progress_label.config(text=f"完成! 共处理 {self.processor.processed_frames} 帧，检测到棋盘 {self.processor.frames_with_board} 帧")
        
        # 保存并更新历史
        self.processor.end_game()
        self.load_game_history()
        self.update_history_list()
        
        # 自动计算胜负
        self.calculate_result()
        
        # 显示完成消息
        messagebox.showinfo("处理完成", "视频处理完成，已生成棋谱")
    
    def start_recording(self):
        """开始记谱"""
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.recording_label.grid()  # 显示记谱中标签
        
        # 清除上次的结果
        self.result_label.config(text="--")
        self.result_data = None
        
        self.processor.start_new_game()
        self.current_display_move = 0
        self.update_board_display()
        
        # 重置位置和颜色信息
        self.position_var.set("位置: --")
        self.stone_color_var.set("颜色: --")
        
        self.update_status("记谱已开始")
    
    def stop_recording(self):
        """结束记谱"""
        if self.processing_thread and self.processing_thread.is_alive():
            self.processor.stop_processing()
            self.processing_thread.join(timeout=1.0)
        
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.recording_label.grid_remove()  # 隐藏记谱中标签
        
        game_info = self.processor.end_game()
        if game_info:
            self.black_count.config(text=str(game_info["black_stones"]))
            self.white_count.config(text=str(game_info["white_stones"]))
            self.score_label.config(text=str(game_info["black_stones"] + game_info["white_stones"]))
        
        self.load_game_history()
        self.update_history_list()
        
        # 计算胜负
        self.calculate_result()
        
        self.update_status("记谱已结束")
    
    def navigate_moves(self, direction):
        """浏览棋局历史"""
        if not self.processor.board_states:
            return
        
        if direction == "first":
            self.current_display_move = 0
        elif direction == "prev" and self.current_display_move > 0:
            self.current_display_move -= 1
        elif direction == "next" and self.current_display_move < len(self.processor.board_states) - 1:
            self.current_display_move += 1
        elif direction == "last":
            self.current_display_move = len(self.processor.board_states) - 1
        
        self.move_var.set(f"第 {self.current_display_move} 手")
        
        # 高亮显示当前手的变化
        highlight_move = None
        
        # 更新落子位置和颜色信息
        if self.current_display_move == 0:
            # 第0手没有落子信息
            self.position_var.set("位置: --")
            self.stone_color_var.set("颜色: --")
        elif self.current_display_move <= len(self.processor.move_history):
            move_info = self.processor.move_history[self.current_display_move - 1]
            highlight_move = move_info["position"]
            
            # 转换位置为围棋坐标 (A-T除I, 1-19)
            x, y = move_info["position"]
            # Go坐标从左上角开始，但我们的数组是从左上角(0,0)开始
            column = chr(ord('A') + (y if y < 8 else y + 1))  # 跳过"I"
            row = 19 - x  # 翻转行号
            position_str = f"{column}{row}"
            self.position_var.set(f"位置: {position_str}")
            
            # 更新颜色信息
            color_str = "黑" if move_info["color"] == "black" else "白"
            self.stone_color_var.set(f"颜色: {color_str}")
        
        self.update_board_display(self.processor.get_board_state(self.current_display_move), highlight_move)
        # 更新SGF显示
        self.update_sgf_display()
    
    def export_sgf(self):
        """导出SGF文件"""
        if not self.processor.move_history:
            messagebox.showinfo("提示", "没有可导出的棋谱")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="导出SGF文件",
            defaultextension=".sgf",
            filetypes=(("SGF文件", "*.sgf"), ("所有文件", "*.*"))
        )
        
        if file_path:
            try:
                output_path = self.processor.export_sgf(file_path)
                messagebox.showinfo("导出成功", f"SGF文件已导出至:\n{output_path}")
            except Exception as e:
                messagebox.showerror("导出失败", f"导出SGF文件时出错:\n{str(e)}")
    
    def export_image(self):
        """导出当前棋盘图像"""
        if not self.processor.board_states:
            messagebox.showinfo("提示", "没有可导出的棋盘")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="导出棋盘图像",
            defaultextension=".png",
            filetypes=(("PNG图像", "*.png"), ("所有文件", "*.*"))
        )
        
        if file_path:
            try:
                # 创建当前棋盘的图像
                board_state = self.processor.get_board_state(self.current_display_move)
                
                # 转换为stone_positions格式
                stone_positions = []
                for i in range(19):
                    for j in range(19):
                        if board_state[i][j] in [0, 2]:
                            stone_positions.append((i, j, board_state[i][j]))
                
                # 生成棋盘图像
                board_image = generate_board_image(stone_positions)
                cv2.imwrite(file_path, board_image)
                
                messagebox.showinfo("导出成功", f"棋盘图像已导出至:\n{file_path}")
            except Exception as e:
                messagebox.showerror("导出失败", f"导出棋盘图像时出错:\n{str(e)}")
    
    def load_game_history(self):
        """加载历史对局记录"""
        self.game_history = []
        
        if not os.path.exists(self.processor.output_dir):
            return
        
        for file in os.listdir(self.processor.output_dir):
            if file.endswith(".json") and file.startswith("game_"):
                try:
                    file_path = os.path.join(self.processor.output_dir, file)
                    with open(file_path, "r") as f:
                        game_data = json.load(f)
                    
                    # 提取关键信息
                    game_info = {
                        "file_path": file_path,
                        "start_time": game_data.get("game_info", {}).get("start_time", "未知"),
                        "black_stones": game_data.get("game_info", {}).get("black_stones", 0),
                        "white_stones": game_data.get("game_info", {}).get("white_stones", 0),
                        "move_count": len(game_data.get("move_history", [])),
                        "board": game_data.get("final_board", []),
                        "result": game_data.get("game_info", {}).get("result", "")
                    }
                    
                    self.game_history.append(game_info)
                except Exception as e:
                    print(f"加载历史对局记录失败: {e}")
    
    def update_history_list(self):
        """更新历史对局列表"""
        self.history_listbox.delete(0, tk.END)
        
        # 按时间倒序排列
        sorted_history = sorted(self.game_history, key=lambda x: x.get("start_time", ""), reverse=True)
        
        for game in sorted_history:
            result_text = f" [{game['result']}]" if game.get('result') else ""
            display_text = f"{game['start_time']} - 黑:{game['black_stones']}子 白:{game['white_stones']}子{result_text}"
            self.history_listbox.insert(tk.END, display_text)
    
    def on_history_select(self, event):
        """当历史对局被选中时"""
        if not self.history_listbox.curselection():
            return
        
        index = self.history_listbox.curselection()[0]
        if index < 0 or index >= len(self.game_history):
            return
        
        selected_game = self.game_history[index]
        
        # 加载选中的对局
        try:
            with open(selected_game["file_path"], "r") as f:
                game_data = json.load(f)
            
            # 更新处理器状态
            self.processor.current_game_info = game_data.get("game_info", {})
            self.processor.move_history = game_data.get("move_history", [])
            self.processor.current_board = game_data.get("final_board", [[-1 for _ in range(19)] for _ in range(19)])
            
            # 重建棋盘状态历史
            self.processor.board_states = []
            current_board = [[-1 for _ in range(19)] for _ in range(19)]
            self.processor.board_states.append(self._deep_copy_board(current_board))
            
            for move in self.processor.move_history:
                i, j = move["position"]
                label = 0 if move["color"] == "black" else 2
                current_board[i][j] = label
                self.processor.board_states.append(self._deep_copy_board(current_board))
            
            # 更新显示
            self.current_display_move = len(self.processor.board_states) - 1
            self.move_var.set(f"第 {self.current_display_move} 手")
            
            # 更新落子位置和颜色信息 (如果有落子)
            if self.current_display_move > 0 and self.processor.move_history:
                last_move = self.processor.move_history[-1]
                # 转换位置为围棋坐标
                x, y = last_move["position"]
                column = chr(ord('A') + (y if y < 8 else y + 1))  # 跳过"I"
                row = 19 - x
                self.position_var.set(f"位置: {column}{row}")
                
                # 更新颜色信息
                color_str = "黑" if last_move["color"] == "black" else "白"
                self.stone_color_var.set(f"颜色: {color_str}")
            else:
                # 如果没有落子，重置信息
                self.position_var.set("位置: --")
                self.stone_color_var.set("颜色: --")
                
            self.update_board_display()
            self.update_sgf_display()
            
            # 更新统计信息
            self.black_count.config(text=str(selected_game["black_stones"]))
            self.white_count.config(text=str(selected_game["white_stones"]))
            self.score_label.config(text=str(selected_game["black_stones"] + selected_game["white_stones"]))
            
            # 显示结果信息
            if "result" in game_data.get("game_info", {}) and game_data["game_info"]["result"]:
                self.result_label.config(text=game_data["game_info"]["result"])
                # 尝试还原详细结果数据
                if "result_details" in game_data.get("game_info", {}):
                    self.result_data = game_data["game_info"]["result_details"]
            else:
                self.result_label.config(text="--")
                self.result_data = None
            
            self.update_status(f"已加载对局: {selected_game['start_time']}")
        except Exception as e:
            # 捕获错误并显示详细信息
            error_msg = str(e)
            messagebox.showerror("加载失败", f"加载对局记录时出错:\n{error_msg}")
    
    def _deep_copy_board(self, board):
        """深拷贝棋盘状态"""
        return [row[:] for row in board]
    
    def update_status(self, message):
        """更新状态栏消息"""
        self.status_var.set(message)
    
    def show_error(self, message):
        """显示错误消息"""
        messagebox.showerror("错误", message)
        self.update_status("出错: " + message)