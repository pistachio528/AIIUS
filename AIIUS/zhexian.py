import matplotlib.pyplot as plt

# 横坐标标签（聚类数量）
x_labels = ['1', '2', '3', '3.5', 'No-Cluster']
x_pos = range(len(x_labels))  # 对应位置索引0,1,2,3,4

# 各分类器在不同聚类数量下的MCC均值（顺序与x_labels一致）
data = {
    'KNN': [0.296, 0.291, 0.316, 0.314, 0.326],
    'LR':  [0.316, 0.322, 0.330, 0.329, 0.298],
    'NB':  [0.299, 0.297, 0.299, 0.299, 0.291],
    'RF':  [0.344, 0.372, 0.387, 0.374, 0.324],
    'TREE':[0.223, 0.258, 0.274, 0.266, 0.293]
}

# 创建图形
plt.figure(figsize=(8, 5))

# 为每条折线设置不同的标记样式
markers = ['o', 's', '^', 'D', 'v']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for i, (clf, values) in enumerate(data.items()):
    plt.plot(x_pos, values, marker=markers[i], color=colors[i], linewidth=2, markersize=8, label=clf)

# 设置横纵坐标轴标签和标题
plt.xticks(x_pos, x_labels)
plt.xlabel('Cluster Number (× $|\mathcal{S}_{\mathrm{min}}|$)', fontsize=12)
plt.ylabel('MCC (Average)', fontsize=12)
# plt.title('MCC (Average) across Different Cluster Numbers for Each Classifier', fontsize=14)

# 添加图例和网格
plt.legend(loc='best', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)

# 优化布局并显示
plt.tight_layout()
plt.savefig('mcc_clusters.png')
plt.show()




import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 原有的折线图数据（保持不变）
x_labels = ['1', '2', '3', '3.5', 'No-Cluster']
x_pos = np.arange(len(x_labels))

# 各分类器在不同聚类数量下的MCC均值
data_means = {
    'KNN': [0.296, 0.291, 0.316, 0.314, 0.326],
    'LR': [0.316, 0.322, 0.330, 0.329, 0.298],
    'NB': [0.299, 0.297, 0.299, 0.299, 0.291],
    'RF': [0.344, 0.372, 0.387, 0.374, 0.324],
    'TREE': [0.223, 0.258, 0.274, 0.266, 0.293]
}


# 从CSV文件中读取标准差数据的函数
def read_std_data_from_csv():
    # 聚类配置与路径映射
    cluster_configs = {
        '1': ('a1-cluster', 'AIIUS-onlyCluster-1times'),
        '2': ('a2-cluster', 'AIIUS-1.0'),
        '3': ('AIIUS-4.0', 'AIIUS-4.0'),  # 注意：这是我们的主要方法
        '3.5': ('a3.5-cluster', 'AIIUS-5.0'),
        'No-Cluster': ('no-cluster', 'AIIUS-4.0-noCluster')
    }

    # 分类器名称映射
    classifier_names = ['KNN', 'LR', 'NB', 'RF', 'TREE']
    classifier_keys = ['knn', 'lr', 'nb', 'rf', 'tree']

    # 存储标准差数据
    data_stds = {clf: [] for clf in classifier_names}

    for x_label, (config_key, path_key) in cluster_configs.items():
        for clf_name, clf_key in zip(classifier_names, classifier_keys):
            # 构建文件路径
            if x_label == '3':
                # 3-cluster是我们的主要方法
                file_path = f"AIIUS-4.0/mean/{clf_key}_mean_results.csv"
            else:
                file_path = f"{path_key}/mean/{clf_key}_mean_results.csv"

            try:
                # 读取CSV文件
                df = pd.read_csv(file_path, comment='#')

                # 查找MCC列
                mcc_column = None
                for col in df.columns:
                    if col.lower() == 'mcc':
                        mcc_column = col
                        break

                if mcc_column:
                    mcc_values = df[mcc_column].dropna().values
                    if len(mcc_values) > 0:
                        std_value = np.std(mcc_values)
                        data_stds[clf_name].append(std_value)
                    else:
                        print(f"警告: {file_path} 中没有MCC数据")
                        data_stds[clf_name].append(0)
                else:
                    print(f"警告: {file_path} 中没有找到MCC列")
                    data_stds[clf_name].append(0)

            except Exception as e:
                print(f"读取文件 {file_path} 时出错: {e}")
                data_stds[clf_name].append(0)

    return data_stds


# 读取标准差数据
data_stds = read_std_data_from_csv()

# 打印标准差数据以供验证
print("标准差数据:")
for clf, stds in data_stds.items():
    print(f"{clf}: {stds}")

# 创建图形
plt.figure(figsize=(12, 7))

# 首先绘制原有的折线图
markers = ['o', 's', '^', 'D', 'v']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for i, (clf, values) in enumerate(data_means.items()):
    plt.plot(x_pos, values, marker=markers[i], color=colors[i],
             linewidth=2, markersize=8, label=clf)

# 在每个聚类数量点处添加柱状图表示标准差
# 设置柱状图的宽度和位置
bar_width = 0.15  # 每个柱子的宽度
x_positions = np.arange(len(x_labels))  # 聚类数量的位置

# 为每个分类器计算柱状图的位置
for i, clf in enumerate(data_stds.keys()):
    # 计算当前分类器柱状图的x位置
    offset = (i - 2) * bar_width  # 使中间的分类器在中心位置
    bar_x = x_positions + offset

    # 绘制柱状图
    plt.bar(bar_x, data_stds[clf], width=bar_width,
            color=colors[i], alpha=0.5, label=f'{clf} (Std)')

# 设置横纵坐标轴标签和标题
plt.xticks(x_pos, x_labels)
plt.xlabel('Cluster Number (× $|\mathcal{S}_{\mathrm{min}}|$)', fontsize=12)
plt.ylabel('MCC (Average) and Standard Deviation', fontsize=12)

# 添加图例（合并折线和柱状图的图例）
# 创建自定义图例项
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = []
for i, (clf, color) in enumerate(zip(data_means.keys(), colors)):
    # 添加折线图图例
    line = Line2D([0], [0], marker=markers[i], color=color,
                  linewidth=2, markersize=8, label=f'{clf} (Mean)')
    legend_elements.append(line)

    # 添加柱状图图例
    patch = Patch(facecolor=color, alpha=0.5, label=f'{clf} (Std)')
    legend_elements.append(patch)

# 添加图例
plt.legend(handles=legend_elements, loc='best', fontsize=9)

# 添加网格
plt.grid(True, linestyle='--', alpha=0.6)

# 优化布局并显示
plt.tight_layout()
plt.savefig('mcc_clusters_with_std_bars.png', dpi=300, bbox_inches='tight')
plt.show()


# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
#
# # 原有的折线图数据（保持不变）
# x_labels = ['1', '2', '3', '3.5', 'No-Cluster']
# x_pos = np.arange(len(x_labels))
#
# # 各分类器在不同聚类数量下的MCC均值
# data_means = {
#     'KNN': [0.296, 0.291, 0.316, 0.314, 0.326],
#     'LR': [0.316, 0.322, 0.330, 0.329, 0.298],
#     'NB': [0.299, 0.297, 0.299, 0.299, 0.291],
#     'RF': [0.344, 0.372, 0.387, 0.374, 0.324],
#     'TREE': [0.223, 0.258, 0.274, 0.266, 0.293]
# }
#
#
# # 从CSV文件中读取标准差数据的函数
# def read_std_data_from_csv():
#     # 聚类配置与路径映射
#     cluster_configs = {
#         '1': ('a1-cluster', 'AIIUS-onlyCluster-1times'),
#         '2': ('a2-cluster', 'AIIUS-1.0'),
#         '3': ('AIIUS-4.0', 'AIIUS-4.0'),  # 注意：这是我们的主要方法
#         '3.5': ('a3.5-cluster', 'AIIUS-5.0'),
#         'No-Cluster': ('no-cluster', 'AIIUS-4.0-noCluster')
#     }
#
#     # 分类器名称映射
#     classifier_names = ['KNN', 'LR', 'NB', 'RF', 'TREE']
#     classifier_keys = ['knn', 'lr', 'nb', 'rf', 'tree']
#
#     # 存储标准差数据
#     data_stds = {clf: [] for clf in classifier_names}
#
#     for x_label, (config_key, path_key) in cluster_configs.items():
#         for clf_name, clf_key in zip(classifier_names, classifier_keys):
#             # 构建文件路径
#             if x_label == '3':
#                 # 3-cluster是我们的主要方法
#                 file_path = f"AIIUS-4.0/mean/{clf_key}_mean_results.csv"
#             else:
#                 file_path = f"{path_key}/mean/{clf_key}_mean_results.csv"
#
#             try:
#                 # 读取CSV文件
#                 df = pd.read_csv(file_path, comment='#')
#
#                 # 查找MCC列
#                 mcc_column = None
#                 for col in df.columns:
#                     if col.lower() == 'mcc':
#                         mcc_column = col
#                         break
#
#                 if mcc_column:
#                     mcc_values = df[mcc_column].dropna().values
#                     if len(mcc_values) > 0:
#                         std_value = np.std(mcc_values)
#                         data_stds[clf_name].append(std_value)
#                     else:
#                         print(f"警告: {file_path} 中没有MCC数据")
#                         data_stds[clf_name].append(0)
#                 else:
#                     print(f"警告: {file_path} 中没有找到MCC列")
#                     data_stds[clf_name].append(0)
#
#             except Exception as e:
#                 print(f"读取文件 {file_path} 时出错: {e}")
#                 data_stds[clf_name].append(0)
#
#     return data_stds
#
#
# # 读取标准差数据
# data_stds = read_std_data_from_csv()
#
# # 以表格形式输出标准差值
# print("\n" + "=" * 70)
# print("标准差表格")
# print("=" * 70)
# print(f"{'分类器/聚类':<10} | {'1':<8} | {'2':<8} | {'3':<8} | {'3.5':<8} | {'No-Cluster':<8}")
# print("-" * 70)
#
# for clf in data_stds.keys():
#     std_values = data_stds[clf]
#     print(
#         f"{clf:<10} | {std_values[0]:<8.4f} | {std_values[1]:<8.4f} | {std_values[2]:<8.4f} | {std_values[3]:<8.4f} | {std_values[4]:<8.4f}")
#
# print("=" * 70)
# print("\n")
#
# # 创建图形
# plt.figure(figsize=(8, 5))
#
# # 首先绘制原有的折线图
# markers = ['o', 's', '^', 'D', 'v']
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
#
# for i, (clf, values) in enumerate(data_means.items()):
#     plt.plot(x_pos, values, marker=markers[i], color=colors[i],
#              linewidth=2, markersize=8, label=clf)
#
# # 在每个聚类数量点处添加柱状图表示标准差
# # 设置柱状图的宽度和位置
# bar_width = 0.15  # 每个柱子的宽度
# x_positions = np.arange(len(x_labels))  # 聚类数量的位置
#
# # 为每个分类器计算柱状图的位置
# for i, clf in enumerate(data_stds.keys()):
#     # 计算当前分类器柱状图的x位置
#     offset = (i - 2) * bar_width  # 使中间的分类器在中心位置
#     bar_x = x_positions + offset
#
#     # 绘制柱状图
#     plt.bar(bar_x, data_stds[clf], width=bar_width,
#             color=colors[i], alpha=0.5, label=f'{clf} (Std)')
#
# # 设置横纵坐标轴标签和标题
# plt.xticks(x_pos, x_labels)
# plt.xlabel('Cluster Number (× $|\mathcal{S}_{\mathrm{min}}|$)', fontsize=12)
# plt.ylabel('MCC (Average) and Standard Deviation', fontsize=12)
#
# # 添加图例（合并折线和柱状图的图例）
# # 创建自定义图例项
# from matplotlib.patches import Patch
# from matplotlib.lines import Line2D
#
# legend_elements = []
# for i, (clf, color) in enumerate(zip(data_means.keys(), colors)):
#     # 添加折线图图例
#     line = Line2D([0], [0], marker=markers[i], color=color,
#                   linewidth=2, markersize=8, label=f'{clf} (Mean)')
#     legend_elements.append(line)
#
#     # 添加柱状图图例
#     patch = Patch(facecolor=color, alpha=0.5, label=f'{clf} (Std)')
#     legend_elements.append(patch)
#
# # 添加图例
# plt.legend(handles=legend_elements, loc='best', fontsize=9)
#
# # 添加网格
# plt.grid(True, linestyle='--', alpha=0.6)
#
# # 优化布局并显示
# plt.tight_layout()
# plt.savefig('mcc_clusters_with_std_bars.png', dpi=300, bbox_inches='tight')
# plt.show()