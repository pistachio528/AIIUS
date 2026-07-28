import numpy as np
import pandas as pd
from scipy import stats
import itertools


def wilcoxon_signed_rank_test(data1, data2):
    """
    执行Wilcoxon符号秩检验

    参数:
    data1, data2: 两个配对样本的数据

    返回:
    dict: 包含检验统计量和p值的字典
    """
    # 确保数据长度相同

    assert len(data1) == len(data2), "数据长度必须相同"

    # 执行Wilcoxon符号秩检验
    stat, p_value = stats.wilcoxon(data1, data2)

    return {
        'wilcoxon_statistic': stat,
        'p_value': p_value,
        'significant': p_value < 0.05  # 只保留0.05显著性水平
    }


def cliffs_delta(data1, data2, lower_better=False, paired=False):
    """
    计算克利夫德δ效应大小

    参数:
    data1, data2: 两个样本的数据
    lower_better: 对于当前指标，值越小是否越好
    paired: 是否为配对数据

    返回:
    dict: 包含克利夫德δ值和相关统计量的字典
    """
    if paired:
        # 配对数据比较
        assert len(data1) == len(data2), "配对数据长度必须相同"
        wins = 0
        losses = 0
        ties = 0

        for i in range(len(data1)):
            if lower_better:
                # 对于PF等指标：值越小越好
                if data1[i] < data2[i]:
                    wins += 1
                elif data1[i] > data2[i]:
                    losses += 1
                else:
                    ties += 1
            else:
                # 对于准确率等指标：值越大越好
                if data1[i] > data2[i]:
                    wins += 1
                elif data1[i] < data2[i]:
                    losses += 1
                else:
                    ties += 1

        total_comparisons = len(data1)
        delta = (wins - losses) / total_comparisons

    else:
        # 独立数据比较（所有配对比较）
        n1, n2 = len(data1), len(data2)
        wins = 0
        losses = 0

        for x, y in itertools.product(data1, data2):
            if lower_better:
                if x < y:
                    wins += 1
                elif x > y:
                    losses += 1
            else:
                if x > y:
                    wins += 1
                elif x < y:
                    losses += 1

        total_comparisons = n1 * n2
        delta = (wins - losses) / total_comparisons
        ties = total_comparisons - wins - losses

    return {
        'cliffs_delta': delta,
        'wins': wins,
        'losses': losses,
        'ties': ties,
        'interpretation': interpret_cliffs_delta(delta)
    }


def interpret_cliffs_delta(delta):
    """解释克利夫德δ的大小"""
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        return "微不足道的效应"
    elif abs_delta < 0.33:
        return "小效应"
    elif abs_delta < 0.474:
        return "中等效应"
    else:
        return "大效应"


def win_tie_loss_analysis(data1, data2, paired=True, lower_better=False):
    """
    胜负平策略分析

    参数:
    data1, data2: 两个样本的数据
    paired: 是否为配对数据
    lower_better: 对于当前指标，值越小是否越好
    """
    if paired:
        # 配对数据比较
        assert len(data1) == len(data2), "配对数据长度必须相同"

        if lower_better:
            # 对于PF等指标：值越小越好
            wins = sum(1 for a, b in zip(data1, data2) if a < b)  # data1更小 → 胜
            losses = sum(1 for a, b in zip(data1, data2) if a > b)  # data1更大 → 负
        else:
            # 对于准确率等指标：值越大越好
            wins = sum(1 for a, b in zip(data1, data2) if a > b)  # data1更大 → 胜
            losses = sum(1 for a, b in zip(data1, data2) if a < b)  # data1更小 → 负

        ties = len(data1) - wins - losses

        win_rate = wins / len(data1)
        loss_rate = losses / len(data1)
        tie_rate = ties / len(data1)

    else:
        # 独立数据比较
        wins = 0
        losses = 0

        for x, y in itertools.product(data1, data2):
            if lower_better:
                if x < y:
                    wins += 1
                elif x > y:
                    losses += 1
            else:
                if x > y:
                    wins += 1
                elif x < y:
                    losses += 1

        ties = len(data1) * len(data2) - wins - losses
        total_comparisons = len(data1) * len(data2)
        win_rate = wins / total_comparisons
        loss_rate = losses / total_comparisons
        tie_rate = ties / total_comparisons

    return {
        'wins': wins,
        'losses': losses,
        'ties': ties,
        'win_rate': win_rate,
        'loss_rate': loss_rate,
        'tie_rate': tie_rate,
        'net_advantage': win_rate - loss_rate
    }


def read_data_from_csv(file_path, column_name):
    """
    从CSV文件中读取指定列的数据

    参数:
    file_path: CSV文件路径
    column_name: 要读取的列名

    返回:
    list: 指定列的值列表
    """
    try:
        df = pd.read_csv(file_path, comment='#')
        if column_name in df.columns:
            data = df[column_name].dropna().tolist()  # 移除缺失值
            return data
        else:
            print(f"错误: {file_path} 中没有找到'{column_name}'列")
            return None
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return None


def get_comparison_methods(classifier):
    """
    获取所有对比方法的文件路径

    参数:
    classifier: 分类器名称 (knn, rf, lr, nb, svm, tree)

    返回:
    dict: 对比方法名称和文件路径的字典
    """
    comparison_methods = {
        # "ufidsf": f"ufidsf_results_metrics-final-multi-dataset\\{classifier}_ufidsf_results.csv",
        # "sb_gan": f"sb_gan_cv_results_metrics-final\\{classifier}_sb_gan_results.csv",
        # "cluster": f"cluster\\cluster_{classifier}.csv",
        # "rus": f"rus\\{classifier} rus 0 .csv",
        # "borderline_smote": f"borderline-smote\\{classifier} bor_smote 0 .csv",
        # "smote": f"smote\\{classifier} smote 0 .csv"

        "deep_mlp": f"AIIUS-4.0-deep_mlp/mean/{classifier}_mean_results.csv",
        "cnn": f"AIIUS-4.0-cnn/mean/{classifier}_mean_results.csv",
        "resnet": f"AIIUS-4.0-resnet/mean/{classifier}_mean_results.csv"

        # "no-cluster": f"AIIUS-4.0-noCluster/mean/{classifier}_mean_results.csv",
        # "a1-cluster": f"AIIUS-onlyCluster-1times/mean/{classifier}_mean_results.csv",
        # "a1.5-cluster": f"AIIUS-2.0/mean/{classifier}_mean_results.csv",
        # "a2-cluster": f"AIIUS-1.0/mean/{classifier}_mean_results.csv",
        # "a2.5-cluster": f"AIIUS-3.0/mean/{classifier}_mean_results.csv",
        # "a3.5-cluster": f"AIIUS-5.0/mean/{classifier}_mean_results.csv",
        # "a4-cluster": f"AIIUS-6.0/mean/{classifier}_mean_results.csv",

        # "a1-cluster": f"AIIUS-onlyCluster-1times/mean/{classifier}_mean_results.csv",
        # "no-cluster": f"AIIUS-4.0-noCluster/mean/{classifier}_mean_results.csv",
        # "only-cluster": f"AIIUS-onlyCluster-final/mean/{classifier}_mean_results.csv",
        #

    }

    # 对比方法显示名称
    comparison_names = {
        # "ufidsf": "UFIDSF",
        # "sb_gan": "SB-GAN",
        # "cluster": "Cluster",
        # "rus": "RUS",
        # "borderline_smote": "Borderline",
        # "smote": "SMOTE"

        "deep_mlp": "Deep-MLP",
        "cnn": "CNN",
        "resnet": "ResNet"

        # "no-cluster": "No-Cluster",
        # "a1-cluster": "1-Cluster",
        # "a1.5-cluster": "1.5-Cluster",
        # "a2-cluster": "2-Cluster",
        # "a2.5-cluster": "2.5-Cluster",
        # "a3.5-cluster": "3.5-Cluster",
        # "a4-cluster": "4-Cluster"

        # "a1-cluster": "1-Cluster",
        # "no-cluster": "No-Cluster",
        # "only-cluster": "Only-Cluster",

    }

    return comparison_methods, comparison_names


def format_p_value(p_value):
    """格式化p值输出"""
    if p_value < 0.05:
        return "< 0.05"
    else:
        return "> 0.05"


def format_cliffs_delta(delta):
    """格式化Cliff's delta输出"""
    return f"{delta:.3f}"


def format_performance(value):
    """格式化性能值输出"""
    return f"{value:.3f}"


def format_wtl_result(wins, losses, ties):
    """格式化胜负平结果输出"""
    total = wins + losses + ties
    win_rate = wins / total if total > 0 else 0
    return f"{wins}-{losses}-{ties} ({win_rate:.1%})"


def analyze_single_comparison(classifier, method_name, metric_column="pf", metric_type="lower_better"):
    """
    分析单个比较并返回结果

    返回:
    tuple: (performance_mean, p_value, cliffs_delta, wins, losses, ties)
    """
    comparison_methods, comparison_names = get_comparison_methods(classifier)

    file_A_path = f"AIIUS-4.0\\mean\\{classifier}_mean_results.csv"
    file_B_path = comparison_methods[method_name]

    # 读取数据
    data_A = read_data_from_csv(file_A_path, metric_column)
    data_B = read_data_from_csv(file_B_path, metric_column)

    if data_A is None or data_B is None:
        return None, None, None, None, None, None

    # 计算性能均值
    performance_mean = np.mean(data_B)

    # Wilcoxon检验
    wilcoxon_result = wilcoxon_signed_rank_test(data_A, data_B)
    p_value = wilcoxon_result['p_value']

    # Cliff's delta
    cliffs_result = cliffs_delta(data_A, data_B, lower_better=(metric_type == "lower_better"), paired=True)
    delta_value = cliffs_result['cliffs_delta']

    # 胜负平分析（新增）
    wtl_result = win_tie_loss_analysis(data_A, data_B, paired=True, lower_better=(metric_type == "lower_better"))
    wins = wtl_result['wins']
    losses = wtl_result['losses']
    ties = wtl_result['ties']

    return performance_mean, p_value, delta_value, wins, losses, ties


def generate_comparison_table(classifiers=None, metric_column="pf", metric_type="lower_better"):
    """
    生成与图片格式相同的比较表格，现在包含胜负平分析

    参数:
    classifiers: 分类器列表
    metric_column: 指标列名
    metric_type: 指标类型
    """
    if classifiers is None:
        classifiers = ["knn", "rf", "lr", "nb", "tree", "svm"]
        # classifiers = ["tree"]

    # methods_order = ["no-cluster", "a1-cluster", "a1.5-cluster", "a2-cluster", "a2.5-cluster", "a3.5-cluster", "a4-cluster"]
    # methods_order = ["a1-cluster", "no-cluster", "only-cluster"]
    # methods_order = ["ufidsf", "sb_gan", "cluster", "rus", "borderline_smote", "smote"]
    methods_order = [ "deep_mlp", "cnn", "resnet"]
    method_display_names = {
        # "no-cluster": "No-Cluster",
        # "a1-cluster": "1-Cluster",
        # "a1.5-cluster": "1.5-Cluster",
        # "a2-cluster": "2-Cluster",
        # "a2.5-cluster": "2.5-Cluster",
        # "a3.5-cluster": "3.5-Cluster",
        # "a4-cluster": "4-Cluster"

        # "a1-cluster": "1-Cluster",
        # "no-cluster": "No-Cluster",
        # "only-cluster": "Only-Cluster",

        # "ufidsf": "UFIDSF",
        # "sb_gan": "SB-GAN",
        # "cluster": "Cluster",
        # "rus": "RUS",
        # "borderline_smote": "Borderline",
        # "smote": "SMOTE"

    }
    method_display_names = {
         "deep_mlp": "Deep-MLP", "cnn": "CNN", "resnet": "ResNet"
    }

    # 存储所有结果
    results = {}

    print("\n" + "=" * 140)
    print(f"Table X The performance of Our Method and baselines in terms of {metric_column.upper()}.")
    print("=" * 140)

    # 表头 - 现在每列包含4行信息：性能、p值、Cliff's δ、胜负平
    header = "|    | 3-Cluster(AIIUS) | " + " | ".join([method_display_names[m] for m in methods_order]) + " |"
    separator = "|" + "|---" * (len(methods_order) + 2) + "|"

    print(header)
    print(separator)

    # 对每个分类器进行分析
    for classifier in classifiers:
        classifier_results = {}

        # 获取我们的方法的性能
        our_method_path = f"AIIUS-4.0\\mean\\{classifier}_mean_results.csv"
        our_data = read_data_from_csv(our_method_path, metric_column)
        our_performance = np.mean(our_data) if our_data else 0.0

        # 表行：性能值
        performance_row = f"| {classifier.upper()} | {format_performance(our_performance)} | "

        # 表行：p值
        pvalue_row = f"| p-value    | "

        # 表行：Cliff's delta
        delta_row = f"| Cliff's δ  | "

        # 表行：胜负平（新增）
        wtl_row = f"| W-T-L      | "

        for method in methods_order:
            perf, p_val, delta, wins, losses, ties = analyze_single_comparison(classifier, method, metric_column,
                                                                               metric_type)

            if perf is not None:
                performance_row += f"{format_performance(perf)} | "
                pvalue_row += f"{format_p_value(p_val)} | "
                delta_row += f"{format_cliffs_delta(delta)} | "
                wtl_row += f"{format_wtl_result(wins, losses, ties)} | "

                classifier_results[method] = {
                    'performance': perf,
                    'p_value': p_val,
                    'cliffs_delta': delta,
                    'wins': wins,
                    'losses': losses,
                    'ties': ties
                }
            else:
                performance_row += "N/A | "
                pvalue_row += "N/A | "
                delta_row += "N/A | "
                wtl_row += "N/A | "

        # 输出结果
        print(performance_row)
        print(pvalue_row)
        print(delta_row)
        print(wtl_row)

        # 在每个分类器结果之间添加分隔线（除了最后一个）
        if classifier != classifiers[-1]:
            print("|" + "|---" * (len(methods_order) + 2) + "|")

        results[classifier] = {
            'our_method': our_performance,
            'comparisons': classifier_results
        }

    print("=" * 140)
    print("Note: W-T-L format: Wins-Losses-Ties (Win Rate) from Our Method's perspective")
    return results


def batch_compare_all_classifiers_and_methods(classifiers=None,  metric_column = "pf", metric_type="lower_better"):
    """
    批量比较所有分类器和所有对比方法，按照图片格式输出

    参数:
    classifiers: 分类器列表，如果为None则使用默认列表
    metric_column: 指标列名
    metric_type: 指标类型
    """
    # 默认分类器列表
    if classifiers is None:
        classifiers = ["knn", "rf", "lr", "nb", "tree"]
        # classifiers = ["tree"]

    print("开始批量比较所有分类器和所有对比方法...")
    print(f"分类器列表: {[c.upper() for c in classifiers]}")
    print(f"指标: {metric_column.upper()} ({'值越小越好' if metric_type == 'lower_better' else '值越大越好'})")

    # 生成比较表格
    all_results = generate_comparison_table(
        classifiers=classifiers,
        metric_column=metric_column,
        metric_type=metric_type
    )

    print("\n" + "=" * 100)
    print("所有分类器和所有对比方法比较完成!")
    print("=" * 100)

    return all_results


# 使用示例
if __name__ == "__main__":
    # 设置要分析的所有分类器
    classifiers_to_analyze = ["knn", "lr", "nb", "rf", "tree"]
    # classifiers_to_analyze = ["tree"]

    # 指标类型（PF是越小越好，MCC是越大越好）
    metric_column = "mcc"  # 或 "mcc"
    metric_type = "lower_better" if metric_column == "pf" else "higher_better"

    # 执行批量比较
    all_results = batch_compare_all_classifiers_and_methods(
        classifiers=classifiers_to_analyze,
        metric_column=metric_column,
        metric_type=metric_type
    )