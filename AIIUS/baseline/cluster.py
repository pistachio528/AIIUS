import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix
import warnings

warnings.filterwarnings('ignore')


class ClusterUndersamplingStrategy1:
    """基于聚类中心策略的欠采样方法（策略一）"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.classifiers = {
            'knn': KNeighborsClassifier(n_neighbors=5),
            'svm': SVC(probability=True, random_state=42),
            'rf': RandomForestClassifier(random_state=42),
            'tree': DecisionTreeClassifier(random_state=42),
            'lr': LogisticRegression(random_state=42),
            'nb': GaussianNB()
        }

    def load_datasets(self, directory_path):
        """从指定目录加载所有CSV数据集"""
        datasets = {}
        for file in os.listdir(directory_path):
            if file.endswith('.csv'):
                file_path = os.path.join(directory_path, file)
                try:
                    df = pd.read_csv(file_path)
                    total_number = len(df)
                    for z in range(total_number):
                        if df.loc[z, "isDefective"] == "buggy" :
                            df.loc[z, "isDefective"] = 1
                        else:
                            df.loc[z, "isDefective"] = 0
                    df['isDefective'] = df['isDefective'].astype(int)

                    cols = list(df.columns)
                    for col in cols:
                        if col == "isDefective":
                            continue
                        df[col] = np.log(df[col] + 1)

                    df = np.array(df)

                        # 假设最后一列是目标变量
                    X = df[:, :-1]
                    y = df[:, -1]

                    # 确保是二分类问题
                    if len(np.unique(y)) == 2:
                        datasets[file] = (X, y)
                        print(f"Loaded dataset: {file}, Shape: {X.shape}, Classes: {np.unique(y)}")
                    else:
                        print(f"Skipping {file}: Not binary classification")
                except Exception as e:
                    print(f"Error loading {file}: {e}")

        return datasets

    def cluster_undersampling_strategy1(self, X_majority, n_minority):
        """策略一：使用聚类中心代表多数类"""
        # 使用K-means聚类
        kmeans = KMeans(n_clusters=n_minority, random_state=42, n_init=10)
        kmeans.fit(X_majority)

        # 使用聚类中心作为新的多数类样本
        X_majority_sampled = kmeans.cluster_centers_

        return X_majority_sampled

    def evaluate_metrics(self, y_true, y_pred, y_prob):
        """计算评估指标"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # PD (Recall/True Positive Rate)
        pd = tp / (tp + fn) if (tp + fn) > 0 else 0

        # PF (False Positive Rate)
        pf = fp / (fp + tn) if (fp + tn) > 0 else 0

        # Balance (G-mean)
        balance = np.sqrt(pd * (1 - pf)) if (pd > 0 and pf < 1) else 0

        # AUC
        auc = roc_auc_score(y_true, y_prob[:, 1]) if len(np.unique(y_true)) == 2 else 0.5

        # MCC
        mcc = matthews_corrcoef(y_true, y_pred)

        return {
            'pd': pd,
            'pf': pf,
            'balance': balance,
            'auc': auc,
            'mcc': mcc
        }

    def experiment(self, directory_path, n_splits=5):
        """主实验函数"""
        # 加载数据集
        datasets = self.load_datasets(directory_path)

        if not datasets:
            print("No valid datasets found!")
            return

        # 存储结果
        results = {}

        for dataset_name, (X, y) in datasets.items():
            print(f"\nProcessing dataset: {dataset_name}")

            # 分层K折交叉验证
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

            # 存储每个数据集的结果
            dataset_results = {clf_name: [] for clf_name in self.classifiers.keys()}

            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # 数据标准化
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)

                # 分离多数类和少数类
                X_train_majority = X_train_scaled[y_train == 0]
                X_train_minority = X_train_scaled[y_train == 1]
                y_train_minority = y_train[y_train == 1]

                n_minority = len(X_train_minority)

                if len(X_train_majority) > n_minority:
                    # 应用策略一：聚类欠采样
                    X_majority_sampled = self.cluster_undersampling_strategy1(
                        X_train_majority, n_minority
                    )

                    # 构建平衡的训练集
                    X_balanced = np.vstack([X_majority_sampled, X_train_minority])
                    y_balanced = np.hstack([np.zeros(len(X_majority_sampled)), y_train_minority])
                else:
                    # 如果多数类样本数已经小于等于少数类，直接使用原始数据
                    X_balanced = X_train_scaled
                    y_balanced = y_train

                # 训练和评估每个分类器
                for clf_name, classifier in self.classifiers.items():
                    try:
                        # 训练分类器
                        classifier.fit(X_balanced, y_balanced)

                        # 预测
                        y_pred = classifier.predict(X_test_scaled)
                        y_prob = classifier.predict_proba(X_test_scaled)

                        # 计算指标
                        metrics = self.evaluate_metrics(y_test, y_pred, y_prob)
                        dataset_results[clf_name].append(metrics)

                    except Exception as e:
                        print(f"Error with {clf_name} on fold {fold}: {e}")
                        # 添加默认值
                        dataset_results[clf_name].append({
                            'pd': 0, 'pf': 0, 'balance': 0, 'auc': 0.5, 'mcc': 0
                        })

            # 计算每个分类器的平均指标
            avg_results = {}
            for clf_name, fold_results in dataset_results.items():
                if fold_results:
                    avg_metrics = {}
                    for metric in ['pd', 'pf', 'balance', 'auc', 'mcc']:
                        values = [result[metric] for result in fold_results]
                        avg_metrics[metric] = np.mean(values)
                    avg_results[clf_name] = avg_metrics

            results[dataset_name] = avg_results

            # 打印当前数据集结果
            self.print_results(dataset_name, avg_results)

        return results

    def print_results(self, dataset_name, results):
        """打印结果"""
        print(f"\n{'=' * 60}")
        print(f"Results for {dataset_name}")
        print(f"{'=' * 60}")
        print(f"{'Classifier':<10} {'PD':<8} {'PF':<8} {'Balance':<8} {'AUC':<8} {'MCC':<8}")
        print(f"{'-' * 60}")

        for clf_name, metrics in results.items():
            print(f"{clf_name:<10} {metrics['pd']:.4f}  {metrics['pf']:.4f}  "
                  f"{metrics['balance']:.4f}  {metrics['auc']:.4f}  {metrics['mcc']:.4f}")

    def save_results(self, results, output_file='cluster\\relink_results.csv'):
        """保存结果到CSV文件"""
        rows = []
        for dataset_name, classifiers in results.items():
            for clf_name, metrics in classifiers.items():
                row = {
                    'Dataset': dataset_name,
                    'Classifier': clf_name,
                    'PD': metrics['pd'],
                    'PF': metrics['pf'],
                    'Balance': metrics['balance'],
                    'AUC': metrics['auc'],
                    'MCC': metrics['mcc']
                }
                rows.append(row)

        df_results = pd.DataFrame(rows)
        df_results.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        return df_results


# 主程序
if __name__ == "__main__":
    # 初始化聚类欠采样器
    cluster_sampler = ClusterUndersamplingStrategy1()

    # 指定包含CSV数据集的目录路径
    data_directory = "datasets\\relink dataset_csv"  # 请替换为你的数据集目录路径

    # 运行实验
    results = cluster_sampler.experiment(data_directory, n_splits=5)

    # 保存结果
    if results:
        cluster_sampler.save_results(results)

        # 打印总体平均结果
        print(f"\n{'=' * 80}")
        print("OVERALL AVERAGE RESULTS ACROSS ALL DATASETS")
        print(f"{'=' * 80}")

        overall_avg = {}
        for clf_name in cluster_sampler.classifiers.keys():
            clf_results = []
            for dataset_results in results.values():
                if clf_name in dataset_results:
                    clf_results.append(dataset_results[clf_name])

            if clf_results:
                avg_metrics = {}
                for metric in ['pd', 'pf', 'balance', 'auc', 'mcc']:
                    values = [result[metric] for result in clf_results]
                    avg_metrics[metric] = np.mean(values)
                overall_avg[clf_name] = avg_metrics

        print(f"{'Classifier':<10} {'PD':<8} {'PF':<8} {'Balance':<8} {'AUC':<8} {'MCC':<8}")
        print(f"{'-' * 80}")
        for clf_name, metrics in overall_avg.items():
            print(f"{clf_name:<10} {metrics['pd']:.4f}  {metrics['pf']:.4f}  "
                  f"{metrics['balance']:.4f}  {metrics['auc']:.4f}  {metrics['mcc']:.4f}")