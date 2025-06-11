from typing import List
import numpy as np
from utils import Particle

### 可以在这里写下一些你需要的变量和函数 ###
COLLISION_DISTANCE = 1
MAX_ERROR = 50000

W = 0.3857   # 权重常数 (auto-tuned)
ALPHA = 1.0314  # 每个点重采样比占比例多的倍数 (auto-tuned)
sigma_P = 0.1145  # 位置方差 (auto-tuned)
sigma_T = 0.0355  # 角度方差 (auto-tuned)
K = 3  # 采样点数量 (auto-tuned)
### 可以在这里写下一些你需要的变量和函数 ###


def generate_uniform_particles(walls, N):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    N: int, 采样点数量
    输出：
    particles: List[Particle], 返回在空地上均匀采样出的N个采样点的列表，每个点的权重都是1/N
    """
    all_particles: List[Particle] = []
    for _ in range(N):
        all_particles.append(Particle(1.0, 1.0, 1.0, 0.0))
    ### 你的代码 ###

    # 确定范围
    MIN_X = walls[:, 0].min()
    MAX_X = walls[:, 0].max()
    MIN_Y = walls[:, 1].min()
    MAX_Y = walls[:, 1].max()
    for i in range(N):
        while True:
            all_particles[i].position[0] = np.random.uniform(MIN_X, MAX_X)
            all_particles[i].position[1] = np.random.uniform(MIN_Y, MAX_Y)
            all_particles[i].theta = np.random.uniform(-np.pi, np.pi)
            test = [int(all_particles[i].position[0]+0.5),
                    int(all_particles[i].position[1]+0.5)]
            if not np.any(np.all(walls == test, axis=1)):
                break
    ### 你的代码 ###
    return all_particles


def calculate_particle_weight(estimated, gt):
    """
    输入：
    estimated: np.array, 该采样点的距离传感器数据
    gt: np.array, Pacman实际位置的距离传感器数据
    输出：
    weight, float, 该采样点的权重
    """
    weight = 1.0
    ### 你的代码 ###
    dis = np.linalg.norm(estimated - gt) # 计算距离
    # 计算权重
    weight = np.exp(-W*dis)
    ### 你的代码 ###
    return weight


def resample_particles(walls, particles: List[Particle]):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息
    particles: List[Particle], 上一次采样得到的粒子，注意是按权重从大到小排列的
    输出：
    particles: List[Particle], 返回重采样后的N个采样点的列表
    """
    resampled_particles: List[Particle] = []
    # 这里先创建 N 个占位粒子
    for _ in range(len(particles)):
        resampled_particles.append(Particle(1.0, 1.0, 1.0, 0.0))
    ### 你的代码 ###
    N = len(particles)

    index = 0

    # 1. 按权重多次复制
    for i in range(N):
        times = int(particles[i].weight * N * ALPHA)
        for _ in range(times):
            if index >= N:
                break
            resampled_particles[index].position[0] = particles[i].position[0]
            resampled_particles[index].position[1] = particles[i].position[1]
            resampled_particles[index].theta = particles[i].theta
            resampled_particles[index].weight = 1.0 / N
            index += 1

    # 2. 如果粒子数量不足N就用均匀分布补足
    if index < N:
        count_left = N - index
        tmp_particles = generate_uniform_particles(walls, count_left)
        for i in range(count_left):
            resampled_particles[index].position[0] = tmp_particles[i].position[0]
            resampled_particles[index].position[1] = tmp_particles[i].position[1]
            resampled_particles[index].theta = tmp_particles[i].theta
            resampled_particles[index].weight = 1.0 / N
            index += 1

    # 3. 给所有粒子加高斯噪声
    for i in range(N):
        resampled_particles[i].position[0] += np.random.normal(0, sigma_P)
        resampled_particles[i].position[1] += np.random.normal(0, sigma_P)
        resampled_particles[i].theta += np.random.normal(0, sigma_T)
        resampled_particles[i].theta = (resampled_particles[i].theta + np.pi) % (2 * np.pi) - np.pi
    ### 你的代码 ###
    return resampled_particles

def apply_state_transition(p: Particle, traveled_distance, dtheta):
    """
    输入：
    p: 采样的粒子
    traveled_distance, dtheta: ground truth的Pacman这一步相对于上一步运动方向改变了dtheta，并移动了traveled_distance的距离
    particle: 按照相同方式进行移动后的粒子
    """
    ### 你的代码 ###

    p.theta += dtheta
    p.theta = (p.theta+np.pi) % (2*np.pi)-np.pi
    dx = traveled_distance*np.cos(p.theta)
    dy = traveled_distance*np.sin(p.theta)
    # 更新位置
    p.position[0] += dx
    p.position[1] += dy

    ### 你的代码 ###
    return p

def get_estimate_result(particles: List[Particle]):
    """
    输入：
    particles: List[Particle], 全部采样粒子
    输出：
    final_result: Particle, 最终的猜测结果
    """
    final_result = Particle()
    ### 你的代码 ###

    particles.sort(key=lambda x: x.weight, reverse=True)
    # 寻找权重最大的前k个粒子，计算平均
    final_result = Particle()
    for i in range(K):
        final_result.position[0] += particles[i].position[0] / K
        final_result.position[1] += particles[i].position[1] / K
        final_result.theta += particles[i].theta/K
    
    ### 你的代码 ###
    return final_result
