import numpy as np
from typing import List
from utils import TreeNode
from simuScene import PlanningMap


### 定义一些你需要的变量和函数 ###
STEP_DISTANCE = 0.6403         # RRT 每次扩展的步长 (auto-tuned)
TARGET_THREHOLD = 0.25      # 到达目标的距离阈值
MAX_ITER = 4509             # RRT 的最大迭代次数 (auto-tuned)
GOAL_BIAS = 0.2315             # RRT 采样时朝向目标的概率 (auto-tuned)
### 定义一些你需要的变量和函数 ###


class RRT:
    def __init__(self, walls) -> None:
        """
        输入包括地图信息，你需要按顺序吃掉的一列事物位置 
        注意：只有按顺序吃掉上一个食物之后才能吃下一个食物，在吃掉上一个食物之前Pacman经过之后的食物也不会被吃掉
        """
        self.map = PlanningMap(walls)
        self.walls = walls
        
        # 其他需要的变量
        ### 你的代码 ###      
        self.path = []  # 用于存储规划的路径
        self.current_path_index = 0  # 当前目标路径点的索引
        # 从 walls 计算地图边界，用于RRT采样
        self.x_size = np.max(walls[:, 0]) + 1
        self.y_size = np.max(walls[:, 1]) + 1
        self.path_start_pos = None # 初始化变量
        ### 你的代码 ###
 
        
        
    def find_path(self, current_position, next_food):
        """
        在程序初始化时，以及每当 pacman 吃到一个食物时，主程序会调用此函数
        current_position: pacman 当前的仿真位置
        next_food: 下一个食物的位置
        
        本函数的默认实现是调用 build_tree，并记录生成的 path 信息。你可以在此函数增加其他需要的功能
        """
        
        ### 你的代码 ###      
        self.path = self.build_bi_rrt(current_position, next_food)
        # 判断是否找到了路径，索引设为0表示有效路径，-1表示无路径
        self.current_path_index = 0 if self.path else -1
        ### 你的代码 ###

        
        
    def get_target(self, current_position, current_velocity):
        """
        主程序将在每个仿真步内调用此函数，并用返回的位置计算 PD 控制力来驱动 pacman 移动
        current_position: pacman 当前的仿真位置
        current_velocity: pacman 当前的仿真速度
        一种可能的实现策略是，仅作为参考：
        （1）记录该函数的调用次数
        （2）假设当前 path 中每个节点需要作为目标 n 次
        （3）记录目前已经执行到当前 path 的第几个节点，以及它的执行次数，如果超过 n，则将目标改为下一节点
        
        你也可以考虑根据当前位置与 path 中节点位置的距离来决定如何选择 target
        
        同时需要注意，仿真中的 pacman 并不能准确到达 path 中的节点。你可能需要考虑在什么情况下重新规划 path
        """
        target_pose = np.zeros_like(current_position)
        ### 你的代码 ###
        if not self.path or self.current_path_index == -1:
            return current_position
        
        current_target = self.path[self.current_path_index]
        distance = np.linalg.norm(current_target - current_position)
        
        # 如果离当前目标点很近，则切换到路径中的下一个点
        if distance < STEP_DISTANCE:
            if self.current_path_index < len(self.path) - 1:
                self.current_path_index += 1
                
        target_pose = self.path[self.current_path_index]
        ### 你的代码 ###
        return target_pose
        
    ### 以下是RRT中一些可能用到的函数框架，全部可以修改，当然你也可以自己实现 ###
    def build_bi_rrt(self, start, goal):
        """
        实现双向RRT算法
        """
        # 修正点：在开始规划前，将起点保存到实例变量中
        self.path_start_pos = start
        
        graph_start = [TreeNode(-1, start[0], start[1])]
        graph_goal = [TreeNode(-1, goal[0], goal[1])]

        for _ in range(MAX_ITER):
            # 1. 交替扩展两棵树
            # 扩展从起点开始的树
            path = self.extend_and_connect(graph_start, graph_goal, goal)
            if path: return path

            # 扩展从终点开始的树
            path = self.extend_and_connect(graph_goal, graph_start, start)
            if path: return path

        return [] # 未找到路径

    def extend_and_connect(self, tree_to_extend, other_tree, target):
        """
        扩展一棵树，并尝试连接到另一棵树
        """
        # 随机采样，带有目标偏向
        if np.random.rand() < GOAL_BIAS:
            sample_point = target
        else:
            sample_point = np.random.rand(2) * np.array([self.x_size, self.y_size])

        # 在待扩展树中找到最近的节点
        nearest_idx, _ = self.find_nearest_point(sample_point, tree_to_extend)
        nearest_node = tree_to_extend[nearest_idx]

        # 从最近节点向采样点扩展一步
        is_empty, new_point = self.connect_a_to_b(nearest_node.pos, sample_point)

        if is_empty:
            # 将新节点加入树中
            new_node = TreeNode(nearest_idx, new_point[0], new_point[1])
            tree_to_extend.append(new_node)

            # 尝试连接到另一棵树
            other_nearest_idx, _ = self.find_nearest_point(new_point, other_tree)
            other_node = other_tree[other_nearest_idx]

            # 检查新节点和另一棵树最近节点之间是否可直连
            hit, _ = self.map.checkline(new_point.tolist(), other_node.pos.tolist())
            if not hit:
                # 连接成功，构建并返回完整路径
                path1 = self.reconstruct_path(new_node, tree_to_extend)
                path2 = self.reconstruct_path(other_node, other_tree)
                
                # 确保路径方向正确
                if np.array_equal(tree_to_extend[0].pos, self.path_start_pos):
                    path1.reverse() # path1是从叶子到根，需要翻转
                    return path1 + path2
                else:
                    path2.reverse() # path2是从叶子到根，需要翻转
                    return path2 + path1
        return None

    def reconstruct_path(self, leaf_node, graph):
        """
        从叶子节点回溯到根节点，构建部分路径
        """
        path = []
        current_node = leaf_node
        while current_node.parent_idx != -1:
            path.append(current_node.pos)
            current_node = graph[current_node.parent_idx]
        path.append(current_node.pos)
        return path

    @staticmethod
    def find_nearest_point(point, graph):
        """
        找到图中离目标位置最近的节点，返回该节点的编号和到目标位置距离、
        输入：
        point：维度为(2,)的np.array, 目标位置
        graph: List[TreeNode]节点列表
        输出：
        nearest_idx, nearest_distance 离目标位置距离最近节点的编号和距离
        """
        nearest_idx = -1
        nearest_distance = float('inf')
        ### 你的代码 ###
        for i, node in enumerate(graph):
            node_pos = node.pos
            dist = np.linalg.norm(point - node_pos)
            if dist < nearest_distance:
                nearest_distance = dist
                nearest_idx = i
        ### 你的代码 ###
        return nearest_idx, nearest_distance
    
    def connect_a_to_b(self, point_a, point_b):
        """
        以A点为起点，沿着A到B的方向前进STEP_DISTANCE的距离，并用self.map.checkline函数检查这段路程是否可以通过
        输入：
        point_a, point_b: 维度为(2,)的np.array，A点和B点位置，注意是从A向B方向前进
        输出：
        is_empty: bool，True表示从A出发前进STEP_DISTANCE这段距离上没有障碍物
        newpoint: 从A点出发向B点方向前进STEP_DISTANCE距离后的新位置，如果is_empty为真，之后的代码需要把这个新位置添加到图中
        """
        is_empty = False
        newpoint = np.zeros(2)
        ### 你的代码 ###
        direction = point_b - point_a
        dist = np.linalg.norm(direction)

        if dist < 1e-6:
            return False, point_a

        unit_vector = direction / dist
        
        if dist < STEP_DISTANCE:
            newpoint = point_b
        else:
            newpoint = point_a + unit_vector * STEP_DISTANCE

        # checkline返回(是否碰撞, 碰撞点)，所以我们需要检查第一个布尔值是否为False
        # docstring提示输入为list，进行转换
        hit, _ = self.map.checkline(point_a.tolist(), newpoint.tolist())
        if not hit:
            is_empty = True
        ### 你的代码 ###
        return is_empty, newpoint
