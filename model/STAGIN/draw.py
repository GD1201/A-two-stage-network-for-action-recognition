import matplotlib.pyplot as plt

# 读取txt文件
file_path = '/home/huaizhenhao/codes/GD/GAIN-master/1.txt'
with open(file_path, 'r') as file:
    lines = file.readlines()

# 定义关节点连接顺序 (以COCO格式为例)
connections = [
    (4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
    (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
    (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)   # 从肩到右腿
]

for index, line in enumerate(lines):
    data = line.strip()
    # 将数据转换为列表
    coordinates = list(map(float, data.split(',')))

    # 提取x和y坐标
    # 提取x和y坐标并取整
    x_coords = list(map(lambda x: int(float(x)), coordinates[0::2]))
    y_coords = list(map(lambda y: int(float(y)), coordinates[1::2]))
    #     x_coords = coordinates[0::2]
    #     y_coords = coordinates[1::2]
    # 绘制关节坐标
    plt.figure(figsize=(8, 6))
    plt.scatter(x_coords, y_coords, color='blue')
    # 连线
    for (i, j) in connections:
        if x_coords[i] != 0.0 and y_coords[i] != 0.0 and x_coords[j] != 0.0 and y_coords[j] != 0.0:  # 过滤掉无效坐标
            plt.plot([x_coords[i], x_coords[j]], [y_coords[i], y_coords[j]], 'r-')

    # 标注关节点
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        if x != 0.0 and y != 0.0:  # 过滤掉无效坐标
            plt.text(x, y, str(i), fontsize=12, color='red')

    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)
#     plt.title(f'OpenPose Keypoint Coordinates with Connections (Line {index + 1})')
    # plt.title(f'第 ({index + 1}) 帧')
    plt.gca().invert_yaxis()

    # 保存图像
    output_path = f'/home/huaizhenhao/codes/GD/GAIN-master/output_image_{index + 1}.png'
    plt.savefig(output_path)
    plt.close()

    print(f"图像已保存到 {output_path}")
