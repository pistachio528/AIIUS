# 全局替代模型
# 全局差分进化
# 预计算聚类
# 多架构替代模型支持
import torch
from scipy.optimize import differential_evolution
import os
import numpy as np
import pandas as pd
import time
import csv
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
import torch.nn.functional as F

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
classifier = "lr"

# 交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ========== 多架构替代模型定义 ==========

# 残差块（用于ResNet）
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

        # 如果输入输出维度不一致，使用1x1卷积进行维度匹配
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.linear1(x)))
        out = self.bn2(self.linear2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        return out


# 1. 基础MLP替代模型
class BaseSurrogateModel(nn.Module):
    def __init__(self, input_dim):
        super(BaseSurrogateModel, self).__init__()
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


# 2. 更深层的MLP替代模型
class DeepMLPSurrogateModel(nn.Module):
    def __init__(self, input_dim):
        super(DeepMLPSurrogateModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# 3. CNN替代模型（适用于表格数据）
class CNNSurrogateModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

        # 方案A：极简但有效的设计
        self.conv_layers = nn.Sequential(
            # 第一层：宽卷积核，捕获特征间全局关系
            nn.Conv1d(1, 8, kernel_size=5, padding=2),
            nn.ReLU(),

            # 第二层：中等卷积核
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            # 全局池化，减少过拟合风险
            nn.AdaptiveAvgPool1d(1)
        )

        # 简单的分类头
        self.classifier = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 重塑为 (batch, 1, features)
        x = x.view(-1, 1, self.input_dim)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# 4. ResNet替代模型（使用残差块）
class ResNetSurrogateModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # 固定维度设计（平衡容量和效率）
        self.hidden_dim = 32  # 固定隐藏维度

        # 1. 输入投影层
        self.input_proj = nn.Linear(input_dim, self.hidden_dim)

        # 2. 两个简化的残差块
        self.res_block1 = SimpleResBlock(self.hidden_dim)
        self.res_block2 = SimpleResBlock(self.hidden_dim)

        # 3. 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 输入投影
        x = torch.relu(self.input_proj(x))

        # 残差块
        x = self.res_block1(x)
        x = self.res_block2(x)

        # 输出
        return self.output_layer(x)

    # def __init__(self, input_dim):
    #     super(ResNetSurrogateModel, self).__init__()
    #
    #     self.initial_layer = nn.Sequential(
    #         nn.Linear(input_dim, 64),
    #         nn.BatchNorm1d(64),
    #         nn.ReLU()
    #     )
    #
    #     self.residual_blocks = nn.Sequential(
    #         ResidualBlock(64, 64),
    #         ResidualBlock(64, 64),
    #         ResidualBlock(64, 32)
    #     )
    #
    #     self.final_layers = nn.Sequential(
    #         nn.Linear(32, 16),
    #         nn.ReLU(),
    #         nn.Dropout(0.1),
    #         nn.Linear(16, 1),
    #         nn.Sigmoid()
    #     )
    #
    # def forward(self, x):
    #     x = self.initial_layer(x)
    #     x = self.residual_blocks(x)
    #     x = self.final_layers(x)
    #     return x


class SimpleResBlock(nn.Module):
    """简化的残差块，没有BatchNorm和Dropout"""

    def __init__(self, dim):
        super().__init__()
        # 两个线性层，保持维度不变
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x):
        residual = x  # 保留输入

        # 残差路径
        out = torch.relu(self.linear1(x))
        out = self.linear2(out)

        # 残差连接 + ReLU激活
        out = out + residual
        return torch.relu(out)


# 模型创建函数
def create_surrogate_model(model_type, input_dim):
    """根据类型创建替代模型"""
    if model_type == "mlp":
        return BaseSurrogateModel(input_dim)
    elif model_type == "deep_mlp":
        return DeepMLPSurrogateModel(input_dim)
    elif model_type == "cnn":
        return CNNSurrogateModel(input_dim)
    elif model_type == "resnet":
        return ResNetSurrogateModel(input_dim)
    else:
        raise ValueError(f"未知模型类型: {model_type}")


# ========== 替代模型配置 ==========
SURROGATE_MODEL_TYPES = ["mlp", "deep_mlp", "cnn", "resnet"]
# 选择要测试的替代模型
selected_surrogate_model = "resnet"  # 可以改为 "deep_mlp", "cnn", "resnet"


def train_global_surrogate_model(dataset, selected_surrogate_model):
    """
    在整个数据集上训练一个全局替代模型
    """
    print(f"训练全局替代模型 (类型: {selected_surrogate_model})...")

    # 准备数据
    x = dataset[:, 0:-1]
    y = dataset[:, -1]

    # 标准化
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # 训练替代模型
    input_dim = x_scaled.shape[1]
    surrogate_model = create_surrogate_model(selected_surrogate_model, input_dim)
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

    print(f"全局替代模型 ({selected_surrogate_model}) 训练完成")
    return surrogate_model, scaler


def perform_clustering(train_data):
    """
    对多数类聚类为 2 × 少数类数量，返回聚类中心
    """
    columns = [f'col_{i}' for i in range(train_data.shape[1] - 1)] + ['class']
    train_df = pd.DataFrame(train_data, columns=columns)
    train_df = train_df.reset_index(drop=True)
    train_df['global_index'] = train_df.index

    majority = train_df[train_df['class'] == 0]
    minority = train_df[train_df['class'] == 1]

    num_minority = len(minority)
    num_majority = len(majority)

    if num_majority <= num_minority:
        return {
            'center_global_indices': [],
            'minority_data': minority,
            'train_df': train_df
        }

    # ★★★ 新逻辑：簇数 = 2 × 少数类数量
    n_clusters = min(num_majority, int(3 * num_minority))

    print(f"聚类: 多数类={num_majority}, 少数类={num_minority}, 聚类数量={n_clusters}")

    majority_x = majority.drop(columns=["class", "global_index"]).values
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

    return {
        'center_global_indices': center_indices,
        'minority_data': minority,
        'train_df': train_df
    }


def perform_adversarial_selection(clustering_result, epsilon, alpha, num_iter,
                                  classifier_name, surrogate, scaler):
    center_indices = clustering_result['center_global_indices']
    train_df = clustering_result['train_df']
    minority = clustering_result['minority_data']

    if len(center_indices) == 0:
        return []

    # ★★★ 获取聚类中心在原始数据空间的特征
    center_df = train_df[train_df['global_index'].isin(center_indices)]
    original_x = center_df.drop(columns=["class", "global_index"]).values

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

    # ★★★ 反标准化回原始空间
    adv_orig = scaler.inverse_transform(adversarial_x)

    # 用原始特征训练真实 classifier
    clf = classifier_for_selection[classifier_name]
    clf.fit(train_df.drop(columns=["class", "global_index"]).values,
            train_df['class'].values)

    # ★★★ 计算重要性
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
    final_train_x = final_balanced_df.drop(columns=['class', 'global_index']).values
    final_train_y = final_balanced_df['class'].values

    print(f"  最终数据集: 多数类={len(retained_majority)}, 少数类={len(minority_data)}, 总计={len(final_balanced_df)}")

    return final_train_x, final_train_y


def apply_clustering_and_adversarial_selection(train_data, epsilon, alpha, num_iter, classifier_name,
                                               global_surrogate_model, global_scaler, precomputed_clustering):
    """
    应用聚类和对抗攻击的完整流程（使用预计算的聚类结果）
    """
    # 使用预计算的聚类结果
    clustering_result = precomputed_clustering

    if len(clustering_result['center_global_indices']) == 0:
        # 如果没有多数类，直接返回原始数据
        return train_data[:, 0:-1], train_data[:, -1]

    # 对抗攻击
    retained_indices = perform_adversarial_selection(
        clustering_result, epsilon, alpha, num_iter, classifier_name,
        global_surrogate_model, global_scaler
    )

    # 构建平衡数据集
    final_train_x, final_train_y = build_balanced_dataset(clustering_result, retained_indices)

    return final_train_x, final_train_y


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
        processed_train_x, processed_train_y = apply_clustering_and_adversarial_selection(
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


def run_single_cv_experiment(selected_surrogate_model):
    """
    运行单层五折交叉验证实验
    """
    output_dir = f'AIIUS-4.0-{selected_surrogate_model}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for iteration in range(3):  # 运行3次以减少时间
        print(f"\n=== 单层五折实验 - 第 {iteration + 1} 次运行 (替代模型: {selected_surrogate_model}) ===")

        single_file = open(os.path.join(output_dir, f'{classifier}_single_cv_{iteration}.csv'), 'a', newline='')
        single_writer = csv.writer(single_file)
        single_writer.writerow(
            ["inputfile", "mcc", "auc", "balance", "pd", "pf", "final_ratio", "processing_time", "surrogate_model"])

        for inputfile in os.listdir("datasets\\aeeem dataset"):  # 只处理前2个文件以节省时间
            print(f"\n处理文件: {inputfile}")
            start_time = time.time()

            # 读取和预处理数据
            dataset = pd.read_csv("datasets\\aeeem dataset\\" + inputfile)
            total_number = len(dataset)
            # 数据预处理
            for z in range(total_number):
                if dataset.loc[z, "class"] == "buggy":
                    dataset.loc[z, "class"] = 1
                else:
                    dataset.loc[z, "class"] = 0
            dataset['class'] = dataset['class'].astype(int)

            cols = list(dataset.columns)
            for col in cols:
                if col == "class":
                    continue
                dataset[col] = np.log(dataset[col] + 1)

            dataset = np.array(dataset)

            # 训练全局替代模型（使用指定类型）
            global_surrogate_model, global_scaler = train_global_surrogate_model(dataset, selected_surrogate_model)

            # 预计算所有fold的聚类结果
            precomputed_clustering = precompute_clustering_for_all_folds(dataset)

            # 差分进化优化参数
            print("正在进行差分进化优化...")
            bounds = [
                (0.01, 0.3),  # epsilon
                (0.001, 0.05),  # alpha
                (5, 20)  # num_iter
            ]

            result = differential_evolution(
                single_cv_objective,
                bounds,
                args=(dataset, classifier, global_surrogate_model, global_scaler, precomputed_clustering),
                strategy='best1bin',
                maxiter=3,  # 减少迭代次数以节省时间
                popsize=6,
                tol=0.01,
                seed=42,
                disp=True,
                workers=1
            )

            best_epsilon, best_alpha, best_num_iter = result.x
            best_num_iter = int(best_num_iter)

            print(f"最优参数: epsilon={best_epsilon:.4f}, alpha={best_alpha:.4f}, num_iter={best_num_iter}")
            print(f"最优MCC: {-result.fun:.4f}")

            # 使用最优参数进行最终评估
            total_recall, total_pf, total_balance = 0, 0, 0
            total_auc, total_mcc, total_ratio = 0, 0, 0

            x = dataset
            y = dataset[:, -1]

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(x, y)):
                train_data = x[train_idx]
                test_data = x[test_idx]

                test_y = test_data[:, -1]
                test_x = test_data[:, 0:-1]

                # 使用最优参数处理训练数据
                processed_train_x, processed_train_y = apply_clustering_and_adversarial_selection(
                    train_data, best_epsilon, best_alpha, best_num_iter, classifier,
                    global_surrogate_model, global_scaler, precomputed_clustering[fold_idx]
                )

                # 训练分类器并评估
                clf = classifier_for_selection[classifier]
                clf.fit(processed_train_x, processed_train_y)
                pred_y = clf.predict(test_x)

                # 计算指标
                recall = recall_score(test_y, pred_y)
                total_recall += recall

                tn, fp, fn, tp = confusion_matrix(test_y, pred_y).ravel()
                pf = fp / (tn + fp) if (tn + fp) > 0 else 0
                total_pf += pf

                balance = 1 - (((0 - pf) ** 2 + (1 - recall) ** 2) / 2) ** 0.5
                total_balance += balance

                auc = roc_auc_score(test_y, pred_y)
                total_auc += auc

                mcc = matthews_corrcoef(test_y, pred_y)
                total_mcc += mcc

                # 计算比例
                ratio = len(processed_train_y[processed_train_y == 1]) / len(processed_train_y)
                total_ratio += ratio

            # 计算平均值
            avg_balance = total_balance / 5
            avg_recall = total_recall / 5
            avg_pf = total_pf / 5
            avg_auc = total_auc / 5
            avg_mcc = total_mcc / 5
            final_ratio = total_ratio / 5

            processing_time = time.time() - start_time  # 计算处理时间

            # 写入结果
            single_writer.writerow([
                inputfile, avg_mcc, avg_auc, avg_balance, avg_recall,
                avg_pf, final_ratio, processing_time, selected_surrogate_model
            ])

            print(f"最终结果 - AUC: {avg_auc:.4f}, MCC: {avg_mcc:.4f}, Balance: {avg_balance:.4f}")
            print(f"处理时间: {time.time() - start_time:.2f}秒")

        single_file.close()


def run_all_surrogate_model_experiments():
    """
    运行所有替代模型架构的实验
    """
    for model_type in SURROGATE_MODEL_TYPES:
        print(f"\n{'=' * 60}")
        print(f"开始实验: {model_type} 替代模型")
        print(f"{'=' * 60}")
        run_single_cv_experiment(model_type)


if __name__ == "__main__":
    # 运行单个模型类型的实验
    run_single_cv_experiment(selected_surrogate_model)

    # 或者运行所有模型类型的实验
    # run_all_surrogate_model_experiments()