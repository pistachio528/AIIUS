# 全局替代模型
# 全局差分进化
# 预计算聚类
import torch
from scipy.optimize import differential_evolution
import os
import numpy as np
import pandas as pd
import time
import warnings
import math
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import neighbors
from sklearn.metrics import roc_auc_score, recall_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

# 可视化相关导入
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

np.set_printoptions(suppress=True)
warnings.filterwarnings('ignore')

# 分类器定义
classifier_for_selection = {
    "knn": neighbors.KNeighborsClassifier(n_neighbors=5),
    "svm": svm.SVC(probability=True),
    "rf": RandomForestClassifier(random_state=0),
    "tree": tree.DecisionTreeClassifier(random_state=0),
    "lr": LogisticRegression(random_state=0),
    "nb": GaussianNB()
}
classifier = "tree"

# 交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# 替代模型
class SurrogateModel(nn.Module):
    def __init__(self, input_dim):
        super(SurrogateModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# 改进的可视化函数
def visualize_datasets(original_data, clustered_data, balanced_data, dataset_name, output_dir, fold_idx=0):
    """
    可视化原始数据集、聚类后数据集和平衡数据集的分布

    参数说明：
    - 图像中括号内的百分比（如45.05%）表示PCA主成分的解释方差比例
    - PC1 (45.05%): 第一主成分解释了数据中45.05%的方差
    - PC2 (12.34%): 第二主成分解释了数据中12.34%的方差
    - 这些百分比越高，说明该主成分包含的信息越多
    """

    print(f"\n开始可视化: {dataset_name} (Fold {fold_idx})")

    # 1. 类别分布饼图
    print("生成类别分布饼图...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Class Distribution Comparison: {dataset_name} (Fold {fold_idx})', fontsize=16)

    datasets_info = [
        ("Original Dataset", original_data),
        ("Clustered Dataset", clustered_data),  # 修改：显示聚类后的数据，而不是ROCT
        ("Balanced Dataset", balanced_data)
    ]

    for i, (name, data) in enumerate(datasets_info):
        if data is not None and len(data) > 0:
            if isinstance(data, pd.DataFrame):
                # 查找标签列
                if 'Defective' in data.columns:
                    class_counts = data['Defective'].value_counts()
                else:
                    # 假设最后一列是标签
                    class_counts = data.iloc[:, -1].value_counts()
            else:
                # numpy数组，假设最后一列是标签
                class_counts = pd.Series(data[:, -1]).value_counts()

            # 计算各类别比例
            clean_pct = class_counts.get(0, 0) / class_counts.sum() * 100
            defect_pct = class_counts.get(1, 0) / class_counts.sum() * 100

            # 使用饼图
            axes[i].pie(class_counts.values, labels=['Clean (0)', 'Defect (1)'],
                        autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
            axes[i].set_title(f'{name}\nTotal: {len(data)} samples\nClean: {clean_pct:.1f}%, Defect: {defect_pct:.1f}%')
        else:
            axes[i].text(0.5, 0.5, 'No data', horizontalalignment='center',
                         verticalalignment='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{name}')

    plt.tight_layout()
    pie_chart_path = os.path.join(output_dir, f'{dataset_name}_class_distribution_fold{fold_idx}.png')
    plt.savefig(pie_chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"饼图已保存: {pie_chart_path}")

    # 2. PCA可视化
    print("生成PCA可视化...")

    # 准备所有数据
    all_datasets = []
    all_labels = []
    dataset_names = []

    for name, data in datasets_info:
        if data is not None and len(data) > 0:
            # 提取特征和标签
            if isinstance(data, pd.DataFrame):
                # 查找标签列
                if 'Defective' in data.columns:
                    features = data.drop(columns=['Defective']).values
                    labels = data['Defective'].values
                else:
                    # 假设最后一列是标签
                    features = data.iloc[:, :-1].values
                    labels = data.iloc[:, -1].values
            else:
                # numpy数组
                features = data[:, :-1]
                labels = data[:, -1]

            all_datasets.append(features)
            all_labels.append(labels)
            dataset_names.append(name)

    if len(all_datasets) == 0:
        print("没有有效数据进行可视化")
        return

    # 检查所有特征维度是否相同
    feature_dims = [data.shape[1] for data in all_datasets]
    print(f"特征维度: {feature_dims}")

    # 如果维度不同，使用最小的维度
    if len(set(feature_dims)) > 1:
        min_dim = min(feature_dims)
        print(f"特征维度不一致，使用最小维度: {min_dim}")
        # 截取所有数据到最小维度
        for i in range(len(all_datasets)):
            if all_datasets[i].shape[1] > min_dim:
                all_datasets[i] = all_datasets[i][:, :min_dim]

    # 合并所有数据用于统一的PCA
    try:
        all_features = np.vstack(all_datasets)

        # 标准化和PCA
        scaler = StandardScaler()
        all_features_scaled = scaler.fit_transform(all_features)

        pca = PCA(n_components=2, random_state=42)
        all_pca = pca.fit_transform(all_features_scaled)

        # 获取解释方差比例
        explained_var_ratio = pca.explained_variance_ratio_
        print(f"PCA解释方差比例: PC1={explained_var_ratio[0]:.2%}, PC2={explained_var_ratio[1]:.2%}")

        # 创建PCA可视化
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'PCA Visualization: {dataset_name} (Fold {fold_idx})', fontsize=16)

        start_idx = 0
        for i, (name, data) in enumerate(datasets_info):
            if data is not None and len(data) > 0:
                # 提取当前数据集的特征
                if isinstance(data, pd.DataFrame):
                    if 'Defective' in data.columns:
                        features = data.drop(columns=['Defective']).values
                        labels = data['Defective'].values
                    else:
                        features = data.iloc[:, :-1].values
                        labels = data.iloc[:, -1].values
                else:
                    features = data[:, :-1]
                    labels = data[:, -1]

                # 确保特征维度一致
                if features.shape[1] > all_features.shape[1]:
                    features = features[:, :all_features.shape[1]]

                # 标准化
                features_scaled = scaler.transform(features)
                pca_result = pca.transform(features_scaled)

                # 绘制
                scatter = axes[i].scatter(pca_result[:, 0], pca_result[:, 1],
                                          c=labels, cmap='coolwarm', alpha=0.7, s=50)
                axes[i].set_title(f'{name}')

                # 设置坐标轴标签，显示解释方差比例
                pc1_var = explained_var_ratio[0] * 100
                pc2_var = explained_var_ratio[1] * 100
                axes[i].set_xlabel(f'PC1 ({pc1_var:.2f}%)')
                axes[i].set_ylabel(f'PC2 ({pc2_var:.2f}%)')

                if i == 0:
                    # 添加图例
                    handles, _ = scatter.legend_elements()
                    legend = axes[i].legend(handles, ['Clean (0)', 'Defect (1)'],
                                            title="Classes", loc='upper right')
                    axes[i].add_artist(legend)

                start_idx += len(features)
            else:
                axes[i].text(0.5, 0.5, 'No data', horizontalalignment='center',
                             verticalalignment='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{name}')

        plt.tight_layout()
        pca_chart_path = os.path.join(output_dir, f'{dataset_name}_pca_fold{fold_idx}.png')
        plt.savefig(pca_chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"PCA图已保存: {pca_chart_path}")

        print(f"\nPCA解释说明:")
        print(
            f"1. PC1 ({explained_var_ratio[0]:.2%}): 第一主成分，解释了数据中 {explained_var_ratio[0] * 100:.2f}% 的方差")
        print(
            f"2. PC2 ({explained_var_ratio[1]:.2%}): 第二主成分，解释了数据中 {explained_var_ratio[1] * 100:.2f}% 的方差")
        print(f"3. 合计解释方差: {(explained_var_ratio[0] + explained_var_ratio[1]) * 100:.2f}%")
        print("注：百分比越高，说明该主成分包含的信息越多")

    except Exception as e:
        print(f"PCA可视化失败: {e}")
        import traceback
        traceback.print_exc()

    print("可视化完成！")


def train_global_surrogate_model(dataset):
    """
    在整个数据集上训练一个全局替代模型
    """
    print("训练全局替代模型...")

    # 准备数据
    x = dataset[:, 0:-1]
    y = dataset[:, -1]

    # 标准化
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # 训练替代模型
    input_dim = x_scaled.shape[1]
    surrogate_model = SurrogateModel(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(surrogate_model.parameters(), lr=0.001)

    # 转换为PyTorch张量
    X_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    train_dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 训练模型
    epochs = 30
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = surrogate_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    print("全局替代模型训练完成")
    return surrogate_model, scaler


def perform_clustering(train_data):
    """
    对多数类聚类为 2 × 少数类数量，返回聚类中心
    """
    # 获取列名
    if isinstance(train_data, pd.DataFrame):
        columns = train_data.columns.tolist()
    else:
        # 如果是numpy数组，创建列名
        columns = [f'feature_{i}' for i in range(train_data.shape[1] - 1)] + ['Defective']

    train_df = pd.DataFrame(train_data, columns=columns)
    train_df = train_df.reset_index(drop=True)
    train_df['global_index'] = train_df.index

    majority = train_df[train_df['Defective'] == 0]
    minority = train_df[train_df['Defective'] == 1]

    num_minority = len(minority)
    num_majority = len(majority)

    if num_majority <= num_minority:
        return {
            'center_global_indices': [],
            'minority_data': minority,
            'train_df': train_df,
            'remaining_majority': majority,
            'clustered_dataset': train_df  # 如果不需要聚类，返回原始数据
        }

    # 簇数 = 2 × 少数类数量
    n_clusters = min(num_majority, int(3 * num_minority))

    print(f"聚类: 多数类={num_majority}, 少数类={num_minority}, 聚类数量={n_clusters}")

    majority_x = majority.drop(columns=["Defective", "global_index"]).values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(majority_x)

    # 找每簇最近中心点
    center_indices = []
    for i in range(n_clusters):
        cluster_points = majority_x[labels == i]
        if len(cluster_points) == 0:
            continue
        distances = np.linalg.norm(cluster_points - kmeans.cluster_centers_[i], axis=1)
        closest_idx = np.argmin(distances)
        original_idx = np.where(labels == i)[0][closest_idx]
        center_indices.append(majority.iloc[original_idx]["global_index"])

    # 构建聚类后的数据集（聚类中心 + 少数类）
    center_df = train_df[train_df['global_index'].isin(center_indices)]
    clustered_df = pd.concat([center_df, minority], axis=0)

    return {
        'center_global_indices': center_indices,
        'minority_data': minority,
        'train_df': train_df,
        'remaining_majority': majority.iloc[
            [i for i in range(len(majority)) if majority.iloc[i]["global_index"] not in center_indices]],
        'clustered_dataset': clustered_df  # 添加聚类后的数据集
    }


def perform_adversarial_selection(clustering_result, epsilon, alpha, num_iter,
                                  classifier_name, surrogate, scaler):
    center_indices = clustering_result['center_global_indices']
    train_df = clustering_result['train_df']
    minority = clustering_result['minority_data']

    if len(center_indices) == 0:
        return []

    # 获取聚类中心在原始数据空间的特征
    center_df = train_df[train_df['global_index'].isin(center_indices)]
    original_x = center_df.drop(columns=["Defective", "global_index"]).values

    # 标准化用于 surrogate 模型
    scaled_x = scaler.transform(original_x)
    tensor_x = torch.tensor(scaled_x, dtype=torch.float32)

    adversarial_x = []

    for i in range(len(tensor_x)):
        x = tensor_x[i].clone().detach()
        adv = x.clone().detach()

        for _ in range(num_iter):
            adv.requires_grad_(True)
            prob = surrogate(adv.unsqueeze(0))
            loss = -prob
            loss.backward()

            with torch.no_grad():
                adv = adv + alpha * adv.grad.sign()
                delta = torch.clamp(adv - x, -epsilon, epsilon)
                adv = x + delta

        adversarial_x.append(adv.detach().numpy())

    adversarial_x = np.array(adversarial_x)

    # 反标准化回原始空间
    adv_orig = scaler.inverse_transform(adversarial_x)

    # 用原始特征训练真实 classifier
    clf = classifier_for_selection[classifier_name]
    clf.fit(train_df.drop(columns=["Defective", "global_index"]).values,
            train_df['Defective'].values)

    # 计算重要性
    orig_prob = clf.predict_proba(original_x)[:, 1]
    adv_prob = clf.predict_proba(adv_orig)[:, 1]
    importance = (adv_prob - orig_prob) ** 2

    best_k = len(minority)
    sorted_indices = [center_indices[i] for i in np.argsort(-importance)[:best_k]]

    return sorted_indices


def build_balanced_dataset(clustering_result, retained_indices):
    """
    构建最终的平衡数据集
    """
    train_df = clustering_result['train_df']
    minority_data = clustering_result['minority_data']

    # 构建平衡数据集
    retained_majority = train_df[train_df['global_index'].isin(retained_indices)]
    final_balanced_df = pd.concat([retained_majority, minority_data], axis=0)
    final_train_x = final_balanced_df.drop(columns=['Defective', 'global_index']).values
    final_train_y = final_balanced_df['Defective'].values

    print(f"  最终数据集: 多数类={len(retained_majority)}, 少数类={len(minority_data)}, 总计={len(final_balanced_df)}")

    return final_train_x, final_train_y, final_balanced_df


def apply_clustering_and_adversarial_selection(train_data, epsilon, alpha, num_iter, classifier_name,
                                               global_surrogate_model, global_scaler, precomputed_clustering):
    """
    应用聚类和对抗攻击的完整流程（使用预计算的聚类结果）
    """
    # 使用预计算的聚类结果
    clustering_result = precomputed_clustering

    if len(clustering_result['center_global_indices']) == 0:
        # 如果没有多数类，直接返回原始数据
        return train_data[:, 0:-1], train_data[:, -1], clustering_result['train_df']

    # 对抗攻击
    retained_indices = perform_adversarial_selection(
        clustering_result, epsilon, alpha, num_iter, classifier_name,
        global_surrogate_model, global_scaler
    )

    # 构建平衡数据集
    final_train_x, final_train_y, final_balanced_df = build_balanced_dataset(clustering_result, retained_indices)

    return final_train_x, final_train_y, final_balanced_df


def precompute_clustering_for_all_folds(dataset):
    """
    为所有fold预计算聚类结果
    """
    print("预计算所有fold的聚类结果...")
    clustering_results = {}

    x = dataset
    y = dataset[:, -1]

    for fold_idx, (train_idx, _) in enumerate(skf.split(x, y)):
        train_data = x[train_idx]
        print(f"Fold {fold_idx}:")
        clustering_results[fold_idx] = perform_clustering(train_data)

    print("所有fold聚类预计算完成")
    return clustering_results


def single_cv_objective(params, dataset, classifier_name, global_surrogate_model, global_scaler,
                        precomputed_clustering):
    """
    单层五折交叉验证的目标函数
    """
    epsilon, alpha, num_iter = params
    num_iter = int(num_iter)

    x = dataset
    y = dataset[:, -1]

    total_mcc = 0
    fold_count = 0

    # 单层五折交叉验证
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(x, y)):
        train_data = x[train_idx]
        val_data = x[val_idx]

        val_y = val_data[:, -1]
        val_x = val_data[:, 0:-1]

        # 使用预计算的聚类结果和当前参数应用对抗攻击选择
        processed_train_x, processed_train_y, _ = apply_clustering_and_adversarial_selection(
            train_data, epsilon, alpha, num_iter, classifier_name,
            global_surrogate_model, global_scaler, precomputed_clustering[fold_idx]
        )

        # 训练分类器
        clf = classifier_for_selection[classifier_name]
        clf.fit(processed_train_x, processed_train_y)

        # 在验证集上评估
        pred_y = clf.predict(val_x)
        mcc = matthews_corrcoef(val_y, pred_y)

        total_mcc += mcc
        fold_count += 1

    avg_mcc = total_mcc / fold_count
    print(f"参数: ε={epsilon:.4f}, α={alpha:.4f}, iter={num_iter}, MCC={avg_mcc:.4f}")
    return -avg_mcc  # 最小化负MCC


def run_experiment_with_visualization():
    """
    运行实验并生成可视化图像
    """
    # 创建输出目录
    output_dir = 'AIIUS-4.0/Visualization_Results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"输出目录: {output_dir}")

    # 只处理PC4数据集
    target_file = "PC4.csv"

    for inputfile in os.listdir("datasets\\nasa dataset_csv"):
        if inputfile != target_file:
            continue

        print(f"\n{'=' * 60}")
        print(f"处理文件: {inputfile}")
        print(f"{'=' * 60}")

        start_time = time.time()

        # 读取数据
        dataset_path = os.path.join("datasets\\nasa dataset_csv", inputfile)
        print(f"读取文件: {dataset_path}")

        try:
            dataset = pd.read_csv(dataset_path)
            print(f"数据集形状: {dataset.shape}")
            print(f"数据集列名: {dataset.columns.tolist()}")
        except Exception as e:
            print(f"读取文件失败: {e}")
            continue

        # 保存原始数据集用于可视化
        original_dataset = dataset.copy()

        # 检查是否有Defective列
        if 'Defective' not in dataset.columns:
            print("错误: 数据集中没有Defective列")
            print(f"可用列: {dataset.columns.tolist()}")
            continue

        # 数据预处理 - 处理Defective列
        total_number = len(dataset)
        print(f"\n数据预处理... (总样本数: {total_number})")

        # 统计初始分布
        original_defective_count = (dataset['Defective'].astype(str).str.upper() == 'Y').sum()
        print(f"原始缺陷样本数: {original_defective_count}")
        print(f"原始干净样本数: {total_number - original_defective_count}")

        for z in range(total_number):
            if str(dataset.loc[z, "Defective"]).strip().upper() == "Y":
                dataset.loc[z, "Defective"] = 1
            else:
                dataset.loc[z, "Defective"] = 0

        dataset['Defective'] = dataset['Defective'].astype(int)

        # 检查是否有非数值列
        feature_columns = [col for col in dataset.columns if col != 'Defective']
        non_numeric_cols = []
        for col in feature_columns:
            if dataset[col].dtype not in ['int64', 'float64']:
                non_numeric_cols.append(col)

        if non_numeric_cols:
            print(f"发现非数值列: {non_numeric_cols}")
            # 尝试转换为数值类型
            for col in non_numeric_cols:
                try:
                    dataset[col] = pd.to_numeric(dataset[col], errors='coerce')
                except:
                    print(f"无法转换列 {col} 为数值类型")

        # 对数变换 - 只对数值列进行
        print("进行对数变换...")
        for col in feature_columns:
            if dataset[col].dtype in ['int64', 'float64']:
                # 检查是否有负值
                min_val = dataset[col].min()
                if min_val < 0:
                    dataset[col] = dataset[col] - min_val + 1

                # 进行对数变换
                dataset[col] = np.log(dataset[col] + 1)

        # 转换为numpy数组
        dataset_np = dataset.values

        # 检查是否有NaN值
        nan_count = np.isnan(dataset_np).sum()
        if nan_count > 0:
            print(f"处理NaN值: {nan_count} 个")
            # 用列均值填充NaN
            for i in range(dataset_np.shape[1]):
                col_data = dataset_np[:, i]
                nan_mask = np.isnan(col_data)
                if nan_mask.any():
                    col_mean = np.nanmean(col_data)
                    dataset_np[nan_mask, i] = col_mean

        print(f"处理后数据集形状: {dataset_np.shape}")

        # 训练全局替代模型
        global_surrogate_model, global_scaler = train_global_surrogate_model(dataset_np)

        # 预计算所有fold的聚类结果
        precomputed_clustering = precompute_clustering_for_all_folds(dataset_np)

        # 差分进化优化参数
        print("\n正在进行差分进化优化...")
        bounds = [
            (0.01, 0.3),  # epsilon
            (0.001, 0.05),  # alpha
            (5, 20)  # num_iter
        ]

        result = differential_evolution(
            single_cv_objective,
            bounds,
            args=(dataset_np, classifier, global_surrogate_model, global_scaler, precomputed_clustering),
            strategy='best1bin',
            maxiter=3,
            popsize=6,
            tol=0.01,
            seed=42,
            disp=True
        )

        best_epsilon, best_alpha, best_num_iter = result.x
        best_num_iter = int(best_num_iter)

        print(f"\n最优参数: epsilon={best_epsilon:.4f}, alpha={best_alpha:.4f}, num_iter={best_num_iter}")
        print(f"最优MCC: {-result.fun:.4f}")

        # 只处理第一个fold并生成可视化
        print(f"\n{'=' * 60}")
        print("只处理第一个fold并生成可视化...")
        print(f"{'=' * 60}")

        x = dataset_np
        y = dataset_np[:, -1]

        # 获取第一个fold的划分
        fold_generator = skf.split(x, y)
        train_idx, test_idx = next(fold_generator)
        fold_idx = 0  # 第一个fold

        print(f"\n处理 Fold {fold_idx}")
        print(f"{'=' * 50}")

        train_data = x[train_idx]
        test_data = x[test_idx]

        test_y = test_data[:, -1]
        test_x = test_data[:, 0:-1]

        # 使用最优参数处理训练数据
        processed_train_x, processed_train_y, balanced_df = apply_clustering_and_adversarial_selection(
            train_data, best_epsilon, best_alpha, best_num_iter, classifier,
            global_surrogate_model, global_scaler, precomputed_clustering[fold_idx]
        )

        # 训练分类器并评估
        clf = classifier_for_selection[classifier]
        clf.fit(processed_train_x, processed_train_y)
        pred_y = clf.predict(test_x)

        # 计算指标（仅打印，不保存）
        recall = recall_score(test_y, pred_y)
        tn, fp, fn, tp = confusion_matrix(test_y, pred_y).ravel()
        pf = fp / (tn + fp) if (tn + fp) > 0 else 0
        balance = 1 - (((0 - pf) ** 2 + (1 - recall) ** 2) / 2) ** 0.5
        auc = roc_auc_score(test_y, pred_y)
        mcc = matthews_corrcoef(test_y, pred_y)

        print(f"Fold {fold_idx} 指标:")
        print(f"  AUC: {auc:.4f}")
        print(f"  MCC: {mcc:.4f}")
        print(f"  Balance: {balance:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  PF: {pf:.4f}")

        # 准备数据用于可视化
        clustering_result = precomputed_clustering[fold_idx]

        # 1. 原始数据（当前fold的训练集）
        original_df = pd.DataFrame(train_data, columns=dataset.columns.tolist())

        # 2. 聚类后的数据
        clustered_df = clustering_result['clustered_dataset']

        # 3. 平衡后的数据
        balanced_df_for_viz = balanced_df.copy()

        print(f"\nFold {fold_idx} 数据统计:")
        print(f"  原始数据: {len(original_df)} 个样本")
        print(f"  聚类后数据: {len(clustered_df)} 个样本")
        print(f"  平衡后数据: {len(balanced_df_for_viz)} 个样本")

        # 生成可视化
        visualize_datasets(
            original_data=original_df,
            clustered_data=clustered_df,  # 使用聚类后的数据
            balanced_data=balanced_df_for_viz,
            dataset_name=inputfile.replace('.csv', ''),
            output_dir=output_dir,
            fold_idx=fold_idx
        )

        processing_time = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"处理完成!")
        print(f"第一个fold的可视化图像已保存")
        print(f"总处理时间: {processing_time:.2f}秒")
        print(f"可视化结果保存在: {output_dir}")
        print(f"{'=' * 60}")

        # 保存好可视化图像后直接退出
        print("\n可视化完成，程序停止运行。")
        return  # 直接返回，停止程序运行


if __name__ == "__main__":
    run_experiment_with_visualization()