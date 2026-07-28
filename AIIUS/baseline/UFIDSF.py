# 安装依赖库（首次运行时执行，执行一次后可注释）
# !pip install numpy pandas scikit-learn scipy

# 导入核心库
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier  # KNN（论文对比分类器扩展）
from sklearn.naive_bayes import GaussianNB  # NB（高斯朴素贝叶斯）
from sklearn.ensemble import RandomForestClassifier  # RF（随机森林）
from sklearn.svm import SVC  # SVM（支持向量机）
from sklearn.tree import DecisionTreeClassifier  # Tree（决策树）
from sklearn.linear_model import LogisticRegression  # LR（逻辑回归）
from scipy.spatial.distance import cityblock  # 曼哈顿距离（论文S1D-MFVNN指定）
from sklearn.neighbors import NearestNeighbors


def calculate_metrics(y_true, y_pred, y_prob):
    """
    计算PD（少数类召回率）、PF（多数类假正率）、Balance、AUC、MCC
    :param y_true: 真实标签（0=多数类，1=少数类，需与PC5.csv标签逻辑一致）
    :param y_pred: 预测标签
    :param y_prob: 少数类预测概率（用于AUC）
    :return: 指标字典
    """
    # 计算混淆矩阵组件（以少数类为正例）
    TP = np.sum((y_true == 1) & (y_pred == 1))  # 少数类正确预测
    TN = np.sum((y_true == 0) & (y_pred == 0))  # 多数类正确预测
    FP = np.sum((y_true == 0) & (y_pred == 1))  # 多数类错误预测为少数类
    FN = np.sum((y_true == 1) & (y_pred == 0))  # 少数类错误预测为多数类

    # PD（论文核心关注的少数类识别率，对应Recall）
    PD = TP / (TP + FN) if (TP + FN) != 0 else 0.0
    # PF（多数类被误判的比例，衡量对多数类的影响）
    PF = FP / (FP + TN) if (FP + TN) != 0 else 0.0
    # Balance（PD与(1-PF)的均值，平衡两类性能）
    Balance = (PD + (1 - PF)) / 2.0
    # AUC（论文核心指标，反映整体区分能力）
    AUC = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else 0.0
    # MCC（马修斯相关系数，综合评估二分类稳健性）
    MCC = matthews_corrcoef(y_true, y_pred)

    return {
        "PD": round(PD, 4),
        "PF": round(PF, 4),
        "Balance": round(Balance, 4),
        "AUC": round(AUC, 4),
        "MCC": round(MCC, 4)
    }


def calculate_s1d_mfvnn(X_maj, k=5):
    """
    计算多数类样本的S1D-MFVNN（特征值最近邻的一维曼哈顿距离之和）
    :param X_maj: 多数类样本特征矩阵（n_samples × n_features）
    :param k: 最近邻数量（论文5.1节验证k=2~10性能稳定，默认k=5）
    :return: 每个多数类样本的S1D-MFVNN值数组
    """
    n_samples, n_features = X_maj.shape
    s1d_mfvnn_list = []

    for i in range(n_samples):
        sample_total_dist = 0.0
        # 逐特征计算一维距离（论文核心：单独处理每个特征的近邻）
        for j in range(n_features):
            curr_feature_val = X_maj[i, j]  # 当前样本第j个特征值
            all_feature_vals = X_maj[:, j]  # 所有多数类样本第j个特征值
            # 计算曼哈顿距离（排除自身，避免距离为0）
            distances = [cityblock([curr_feature_val], [val]) for val in all_feature_vals]
            valid_distances = [d for d in distances if d > 1e-8]  # 浮点误差容错
            # 取前k个最小距离求和
            top_k_dist = sorted(valid_distances)[:k] if len(valid_distances) >= k else valid_distances
            sample_total_dist += sum(top_k_dist)
        s1d_mfvnn_list.append(sample_total_dist)

    return np.array(s1d_mfvnn_list)


def calculate_feature_importance(X, y, n_estimators=100):
    """
    计算特征重要性（Gini Importance，论文3.2节指定）
    :param X: 过滤后的完整特征矩阵（含多数类+少数类）
    :param y: 过滤后的完整标签
    :param n_estimators: 随机森林树数量（论文默认无特殊设置，取100）
    :return: 特征重要性数组（长度=特征数）
    """
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf.fit(X, y)
    # 论文3.2节：基尼重要性直接反映特征对分类的贡献，无需额外归一化
    return rf.feature_importances_


def double_side_filter_optimized(X, y, k=5, danger_ratio=0.7):
    """
    优化的双边过滤（DSF）：去除多数类和少数类中的噪声
    :param X: 原始特征矩阵
    :param y: 原始标签
    :param k: 最近邻数量
    :param danger_ratio: 危险特征比例阈值
    :return: 过滤后的特征矩阵X_filtered、标签y_filtered
    """
    n_samples, n_features = X.shape
    keep_mask = np.ones(n_samples, dtype=bool)  # 标记是否保留样本

    for i in range(n_samples):
        curr_label = y[i]
        danger_count = 0  # "危险"特征计数器

        for j in range(n_features):
            # 当前样本第j个特征值
            curr_feature_val = X[i, j]
            # 全数据集第j个特征值与对应标签
            all_feature_vals = X[:, j]
            all_labels = y

            # 计算当前特征值的k近邻
            distances = [cityblock([curr_feature_val], [val]) for val in all_feature_vals]
            # 排序取前k个近邻（排除自身，避免干扰）
            neighbor_indices = np.argsort(distances)[1:k + 1]
            neighbor_labels = all_labels[neighbor_indices]

            # 统计近邻中同类（k+）与异类（k-）数量
            k_plus = np.sum(neighbor_labels == curr_label)
            k_minus = k - k_plus

            # 若异类数量>同类，标记该特征为"危险"
            if k_minus > k_plus:
                danger_count += 1

        # 使用比例阈值：如果危险特征比例超过阈值，则剔除
        if danger_count >= n_features * danger_ratio:
            keep_mask[i] = False

    return X[keep_mask], y[keep_mask]


def calculate_dynamic_k(n_minority):
    """
    根据少数类样本数量动态选择k值
    """
    if n_minority <= 10:
        return 2
    elif n_minority <= 30:
        return 3
    elif n_minority <= 50:
        return 4
    else:
        return 5


def remove_boundary_noise(X, y, threshold=0.1):
    """
    移除边界噪声样本
    """
    if len(X) == 0:
        return X, y

    nbrs = NearestNeighbors(n_neighbors=min(5, len(X))).fit(X)
    distances, indices = nbrs.kneighbors(X)

    keep_mask = np.ones(len(X), dtype=bool)

    for i in range(len(X)):
        neighbor_labels = y[indices[i]]
        same_class_ratio = np.sum(neighbor_labels == y[i]) / len(neighbor_labels)

        # 如果邻居中同类比例过低，可能是边界噪声
        if same_class_ratio < threshold:
            keep_mask[i] = False

    return X[keep_mask], y[keep_mask]


def run_optimized_experiment(X, y, dataset_name):
    """
    运行单个数据集的优化UFIDSF实验（10折交叉验证）
    :param X: 特征矩阵
    :param y: 标签
    :param dataset_name: 数据集名称（用于结果保存）
    :return: 各分类器平均性能DataFrame
    """
    # 初始化分类器
    classifiers = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "NB": GaussianNB(),
        "RF": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Tree": DecisionTreeClassifier(random_state=42),
        "LR": LogisticRegression(max_iter=1000, random_state=42)
    }

    # 初始化10折交叉验证
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # 存储各分类器的折次性能
    fold_results = {clf_name: [] for clf_name in classifiers.keys()}

    # 遍历每折数据
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n=== 第{fold_idx}/10折交叉验证 ===")

        # 划分训练集/测试集
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 数据标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 应用优化的UFIDSF欠采样
        ufidsf = OptimizedUFIDSF(k=5, danger_ratio=0.7, noise_threshold=0.1)
        X_train_balanced, y_train_balanced = ufidsf.fit_resample(X_train_scaled, y_train)
        print(
            f"训练集欠采样后：总样本数={len(X_train_balanced)}（多数类={len(y_train_balanced[y_train_balanced == 0])}，少数类={len(y_train_balanced[y_train_balanced == 1])}）")

        # 每个分类器训练与评估
        for clf_name, clf in classifiers.items():
            try:
                # 训练分类器
                clf.fit(X_train_balanced, y_train_balanced)
                # 预测（标签+概率）
                y_pred = clf.predict(X_test_scaled)
                y_prob = clf.predict_proba(X_test_scaled)[:, 1]  # 少数类（标签1）的概率
                # 计算指标
                metrics = calculate_metrics(y_test, y_pred, y_prob)
                fold_results[clf_name].append(metrics)
                print(f"  {clf_name}：PD={metrics['PD']}, PF={metrics['PF']}, AUC={metrics['AUC']}")
            except Exception as e:
                print(f"  {clf_name} 训练或预测出错: {str(e)}")
                # 添加默认指标
                fold_results[clf_name].append({"PD": 0.0, "PF": 0.0, "Balance": 0.0, "AUC": 0.0, "MCC": 0.0})

    # 计算各分类器的平均性能
    avg_metrics = []
    for clf_name, results in fold_results.items():
        avg = {
            "Classifier": clf_name,
            "Mean_PD": np.mean([r["PD"] for r in results]),
            "Mean_PF": np.mean([r["PF"] for r in results]),
            "Mean_Balance": np.mean([r["Balance"] for r in results]),
            "Mean_AUC": np.mean([r["AUC"] for r in results]),
            "Mean_MCC": np.mean([r["MCC"] for r in results])
        }
        avg_metrics.append(avg)

    # 转换为DataFrame
    avg_df = pd.DataFrame(avg_metrics)

    # 保存当前数据集的结果
    result_save_path = f"Optimized_UFIDSF_{dataset_name}_results.csv"
    avg_df.to_csv(result_save_path, index=False)
    print(f"\n{dataset_name} 优化结果已保存至：{os.path.abspath(result_save_path)}")

    return avg_df


class OptimizedUFIDSF:
    def __init__(self, k=5, danger_ratio=0.7, noise_threshold=0.1):
        """
        初始化优化的UFIDSF欠采样器
        :param k: 最近邻数量
        :param danger_ratio: 危险特征比例阈值
        :param noise_threshold: 边界噪声阈值
        """
        self.k = k
        self.danger_ratio = danger_ratio
        self.noise_threshold = noise_threshold
        self.feature_importance_ = None
        self.s1d_mfvnn_ = None

    def fit_resample(self, X, y):
        """
        执行优化的UFIDSF欠采样
        :param X: 输入特征矩阵
        :param y: 输入标签（0=多数类，1=少数类）
        :return: 欠采样后的X_resampled（平衡特征矩阵）、y_resampled（平衡标签）
        """
        # 动态调整k值
        n_min = np.sum(y == 1)
        dynamic_k = calculate_dynamic_k(n_min)

        # 步骤1：分离多数类与少数类
        X_maj = X[y == 0]
        y_maj = y[y == 0]
        X_min = X[y == 1]
        y_min = y[y == 1]
        n_min = len(X_min)

        # 如果少数类样本数为0，直接返回
        if n_min == 0:
            return X, y

        # 步骤2：优化的双边过滤
        X_filtered, y_filtered = double_side_filter_optimized(
            X, y, k=dynamic_k, danger_ratio=self.danger_ratio
        )

        # 过滤后重新分离多数类与少数类
        X_maj_filtered = X_filtered[y_filtered == 0]
        y_maj_filtered = y_filtered[y_filtered == 0]
        X_min_filtered = X_filtered[y_filtered == 1]
        y_min_filtered = y_filtered[y_filtered == 1]

        # 如果过滤后少数类样本数为0，返回原始少数类
        if len(X_min_filtered) == 0:
            X_min_filtered = X_min
            y_min_filtered = y_min

        # 步骤3：计算多数类S1D-MFVNN
        if len(X_maj_filtered) > 0:
            self.s1d_mfvnn_ = calculate_s1d_mfvnn(X_maj_filtered, k=dynamic_k)
        else:
            self.s1d_mfvnn_ = np.array([])

        # 步骤4：计算特征重要性
        self.feature_importance_ = calculate_feature_importance(X_filtered, y_filtered)

        # 如果多数类样本数为0，直接返回少数类
        if len(X_maj_filtered) == 0:
            return X_min_filtered, y_min_filtered

        # 步骤5：改进的特征重要性加权
        n_maj, n_features = X_maj_filtered.shape
        weighted_s1d = []

        # 归一化特征重要性
        if np.sum(self.feature_importance_) > 0:
            normalized_importance = self.feature_importance_ / np.sum(self.feature_importance_)
        else:
            normalized_importance = np.ones(n_features) / n_features

        for i in range(n_maj):
            sample_weight = 0.0
            for j in range(n_features):
                # 重新计算单特征S1D-MFVNN
                curr_val = X_maj_filtered[i, j]
                all_vals = X_maj_filtered[:, j]
                dists = [cityblock([curr_val], [v]) for v in all_vals]
                valid_dists = [d for d in dists if d > 1e-8]
                top_k_dists = sorted(valid_dists)[:dynamic_k] if len(valid_dists) >= dynamic_k else valid_dists
                feat_s1d = sum(top_k_dists)

                # 改进的加权公式：使用对数变换避免极端值
                weight = 1 + np.log1p(normalized_importance[j] * 10)
                sample_weight += feat_s1d * weight

            weighted_s1d.append(sample_weight)

        weighted_s1d = np.array(weighted_s1d)

        # 步骤6：按加权S1D降序剔除样本，直至与少数类平衡
        sorted_indices = np.argsort(weighted_s1d)[::-1]
        n_to_select = min(n_min, len(X_maj_filtered))
        selected_maj_idx = sorted_indices[:n_to_select]
        X_maj_selected = X_maj_filtered[selected_maj_idx]
        y_maj_selected = y_maj_filtered[selected_maj_idx]

        # 步骤7：合并多数类（筛选后）与少数类（过滤后）
        X_resampled = np.vstack([X_maj_selected, X_min_filtered])
        y_resampled = np.hstack([y_maj_selected, y_min_filtered])

        # 步骤8：后处理 - 移除边界噪声
        if len(X_resampled) > 0:
            X_resampled, y_resampled = remove_boundary_noise(
                X_resampled, y_resampled, threshold=self.noise_threshold
            )

        return X_resampled, y_resampled


def load_and_process_data(data_path):
    """
    加载并处理单个数据集
    :param data_path: 数据集文件路径
    :return: 特征矩阵X，标签y，数据集名称
    """
    print(f"正在处理数据集: {data_path}")

    try:
        # 读取CSV文件
        dataset = pd.read_csv(data_path)

        # 删除不需要的列（如果存在）
        columns_to_drop = ["name", "version", "name.1"]
        for col in columns_to_drop:
            if col in dataset.columns:
                dataset = dataset.drop(columns=col)

        # 处理bug列，将大于0的值转换为1
        if "bug" in dataset.columns:
            dataset["bug"] = (dataset["bug"] > 0).astype(int)

        # 对数值特征进行log变换（排除bug列）
        cols = list(dataset.columns)
        for col in cols:
            if col == "bug":
                continue
            if dataset[col].dtype in ['int64', 'float64']:
                dataset[col] = np.log(dataset[col].abs() + 1)  # 使用绝对值避免负数

        print("预处理完成！")

        # 分离特征与标签
        if "global_index" in dataset.columns:
            X = dataset.drop(columns=['bug', 'global_index']).values
        else:
            X = dataset.drop(columns=['bug']).values

        y = dataset['bug'].values

        # 获取数据集名称（不含扩展名）
        dataset_name = os.path.splitext(os.path.basename(data_path))[0]

        # 打印数据集信息
        print(f"数据集 {dataset_name} 加载完成：")
        print(f"- 总样本数：{len(X)}")
        print(f"- 特征数：{X.shape[1]}")
        print(f"- 多数类样本数（标签0）：{len(y[y == 0])}")
        print(f"- 少数类样本数（标签1）：{len(y[y == 1])}")
        if len(y[y == 1]) > 0:
            print(f"- 不平衡比例（多数类/少数类）：{round(len(y[y == 0]) / len(y[y == 1]), 2)}")
        else:
            print(f"- 不平衡比例：无穷大（无少数类样本）")

        return X, y, dataset_name

    except Exception as e:
        print(f"处理数据集 {data_path} 时出错: {str(e)}")
        return None, None, None


def process_all_datasets_optimized(data_dir="datasets\\promise dataset"):
    """
    处理目录中的所有数据集（优化版本）
    :param data_dir: 数据集目录路径
    """
    # 检查路径是否存在
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据集目录不存在：{data_dir}")

    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    if not csv_files:
        print(f"在目录 {data_dir} 中未找到CSV文件")
        return

    print(f"找到 {len(csv_files)} 个数据集文件")

    all_results = {}

    for csv_file in csv_files:
        try:
            data_path = os.path.join(data_dir, csv_file)
            X, y, dataset_name = load_and_process_data(data_path)

            if X is None or y is None:
                continue

            # 检查是否有少数类样本
            if len(y[y == 1]) == 0:
                print(f"警告：数据集 {dataset_name} 没有少数类样本，跳过处理")
                continue

            print(f"\n{'=' * 50}")
            print(f"开始优化处理数据集: {dataset_name}")
            print(f"{'=' * 50}")

            # 运行优化实验
            results = run_optimized_experiment(X, y, dataset_name)
            all_results[dataset_name] = results

            print(f"\n数据集 {dataset_name} 优化处理完成！")

        except Exception as e:
            print(f"处理数据集 {csv_file} 时出错: {str(e)}")
            continue

    return all_results


if __name__ == "__main__":
    # 运行所有数据集的优化UFIDSF实验
    print("=" * 50)
    print("优化UFIDSF方法实验（处理目录中所有数据集）")
    print("=" * 50)

    # 执行优化实验
    all_results = process_all_datasets_optimized()

    # 输出总结信息
    if all_results:
        print("\n" + "=" * 50)
        print("所有数据集优化处理完成！")
        print("=" * 50)
        print(f"成功处理 {len(all_results)} 个数据集")

        # 创建汇总文件
        summary_data = []
        for dataset_name, results in all_results.items():
            for _, row in results.iterrows():
                summary_data.append({
                    "Dataset": dataset_name,
                    "Classifier": row["Classifier"],
                    "Mean_PD": row["Mean_PD"],
                    "Mean_PF": row["Mean_PF"],
                    "Mean_Balance": row["Mean_Balance"],
                    "Mean_AUC": row["Mean_AUC"],
                    "Mean_MCC": row["Mean_MCC"]
                })

        summary_df = pd.DataFrame(summary_data)
        summary_save_path = "Optimized_UFIDSF_All_Datasets_Summary.csv"
        summary_df.to_csv(summary_save_path, index=False)
        print(f"\n优化汇总结果已保存至：{os.path.abspath(summary_save_path)}")
    else:
        print("\n没有成功处理任何数据集")