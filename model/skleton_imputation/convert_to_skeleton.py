import json
import os
import numpy as np

# JSON 文件夹路径
folder_path = '/home/huaizhenhao/codes/GD/GAIN-master/batterymodule'
# 输出 npy 文件路径
output_npy_path = '/home/huaizhenhao/codes/GD/GAIN-master/batterymodule.npy'

# 获取文件夹中所有 JSON 文件的路径
json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

# 初始化数据列表
all_videos = []

# 遍历每个 JSON 文件
for json_file in json_files:
    file_path = os.path.join(folder_path, json_file)
    
    # 读取 JSON 文件
    with open(file_path, 'r') as f:
        video_info = json.load(f)
    
    # 初始化视频数据列表
    video_data = []
    
    # 遍历每一帧数据
    for frame_info in video_info['data']:
        frame_index = frame_info['frame_index']
        
        # 提取第一个 skeleton 的 pose 数据
        if frame_info["skeleton"]:
            skeleton_info = frame_info["skeleton"][0]
            pose = skeleton_info['pose']
            
            # 将 pose 数据添加到视频数据中
            video_data.append(pose)
    
    # 将视频数据添加到所有视频列表中
    all_videos.append(video_data)

# 确定 N, T, V, C 的值
N = len(all_videos)
T = max(len(video) for video in all_videos)
V = len(all_videos[0][0]) // 2  # 假设每个 pose 包含 V 个节点的 (x, y) 坐标
C = 2  # 假设每个节点有两个坐标 (x, y)

# 创建一个形状为 [N, T, V, C] 的 numpy 数组
pose_data = np.zeros((N, T, V, C))

# 填充 numpy 数组
for i, video in enumerate(all_videos):
    for j, pose in enumerate(video):
        for k in range(V):
            pose_data[i, j, k, 0] = pose[2*k]
            pose_data[i, j, k, 1] = pose[2*k + 1]

# 保存为 npy 文件
np.save(output_npy_path, pose_data)

print(f"Pose 数据已保存为 {output_npy_path}，形状为 {pose_data.shape}")
