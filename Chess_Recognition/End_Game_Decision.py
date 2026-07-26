class GameResult:
    def __init__(self, processor, komi=6.5):
        self.processor = processor
        self.komi = komi  # 贴目
        self.black_territory = 0  # 黑方领地
        self.white_territory = 0  # 白方领地
        self.black_stones = 0  # 黑方棋子数
        self.white_stones = 0  # 白方棋子数
        self.neutral_points = 0  # 中立点（双方都能进入的空点）
        self.result = ""  # 结果文本
        
    def calculate(self):
        """计算胜负"""
        # 获取终局棋盘
        board = self.processor.current_board
        
        # 计算棋子数
        for i in range(19):
            for j in range(19):
                if board[i][j] == 0:  # 黑子
                    self.black_stones += 1
                elif board[i][j] == 2:  # 白子
                    self.white_stones += 1
        
        # 分析领地
        self._analyze_territory(board)
        
        # 计算总分
        black_total = self.black_stones + self.black_territory
        white_total = self.white_stones + self.white_territory + self.komi
        
        # 确定胜负
        if black_total > white_total:
            self.result = f"黑胜 {black_total - white_total:.1f} 目"
        elif white_total > black_total:
            self.result = f"白胜 {white_total - black_total:.1f} 目"
        else:
            self.result = "和棋"
            
        return self
    
    def _analyze_territory(self, board):
        """分析领地"""
        visited = [[False for _ in range(19)] for _ in range(19)]
        
        for i in range(19):
            for j in range(19):
                if board[i][j] == -1 and not visited[i][j]:  # 空点且未访问过
                    # 对该连通空点区域进行分析
                    territory_type, count = self._analyze_empty_region(board, i, j, visited)
                    
                    if territory_type == "black":
                        self.black_territory += count
                    elif territory_type == "white":
                        self.white_territory += count
                    else:  # neutral
                        self.neutral_points += count
    
    def _analyze_empty_region(self, board, i, j, visited):
        """分析连通的空点区域，确定归属"""
        # 使用广度优先搜索 (BFS)
        queue = [(i, j)]
        visited[i][j] = True
        region_points = []
        black_border = False
        white_border = False
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while queue:
            x, y = queue.pop(0)
            region_points.append((x, y))
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                if 0 <= nx < 19 and 0 <= ny < 19:
                    if board[nx][ny] == -1 and not visited[nx][ny]:  # 空点
                        queue.append((nx, ny))
                        visited[nx][ny] = True
                    elif board[nx][ny] == 0:  # 黑子
                        black_border = True
                    elif board[nx][ny] == 2:  # 白子
                        white_border = True
        
        # 确定该区域的归属
        if black_border and not white_border:
            return "black", len(region_points)
        elif white_border and not black_border:
            return "white", len(region_points)
        else:
            return "neutral", len(region_points)  # 中立区域或双方都有边界
        
    def get_detailed_result(self):
        """获取详细结果信息"""
        return {
            "black_stones": self.black_stones,
            "white_stones": self.white_stones,
            "black_territory": self.black_territory,
            "white_territory": self.white_territory,
            "neutral_points": self.neutral_points,
            "komi": self.komi,
            "black_total": self.black_stones + self.black_territory,
            "white_total": self.white_stones + self.white_territory + self.komi,
            "result": self.result
        }
