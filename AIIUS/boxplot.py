# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.patches import Polygon
# import pandas as pd
# import seaborn as sns
#
# # all_data = [np.random.normal(0, std, 100) for std in range(1, 20)]
# plt.rc('font', family='Times New Roman')
#
# tips = pd.read_csv("C:\\Users\\shuof\\论文数据\\ltrus to reliability\\ltrus ltrus-ratio ltrus-thres mcc boxplot.csv")
# ax = sns.boxplot(x="Classifier", y="mcc", hue="Technique",
#                  data=tips, palette="Set3", showmeans=True, whis=100, meanprops={"markerfacecolor": "black", "markeredgecolor": "black"}, medianprops={"color": "black"}) # 平均数是三角，中位数是横线
#
# # ax.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
# ax.set_ylabel('')
# ax.set_xlabel('')
# plt.xticks(fontsize=30)
# plt.yticks(fontsize=30)
# # plt.legend()z
# plt.legend(fontsize=20, title_fontsize=30, borderaxespad=0., ncol=7) , # bbox_to_anchor=(0.95, -0.06)) ncol是用来控制legend是竖着放还是横着放
# # smote_knn = smote_knn.drop(columns="inputfile")
# # knn = knn.T
#
#
# # svm = pd.read_csv("D:\\radius class overlapping\\svm auc.csv")
# # # smote_knn = smote_knn.drop(columns="inputfile")
# # svm = svm.T
#
# # rf = pd.read_csv("D:\\radius class overlapping\\rf auc.csv")
# # # smote_knn = smote_knn.drop(columns="inputfile")
# # rf = rf.T
# # # rf = pd.Series(rf)
# #
# # lr = pd.read_csv("D:\\radius class overlapping\\lr auc.csv")
# # # smote_knn = smote_knn.drop(columns="inputfile")
# # lr = lr.T
# #
# # nb = pd.read_csv("D:\\radius class overlapping\\nb auc.csv")
# # # smote_knn = smote_knn.drop(columns="inputfile")
# # nb = nb.T
#
# # smote_svm = pd.read_csv("D:\\distance metric\\smote\\combined smote svm auc.csv")
# # # smote_svm = smote_svm.drop(columns="inputfile")
# # smote_svm = smote_svm.T
# #
# # smote_rf = pd.read_csv("D:\\distance metric\\smote\\combined smote rf auc.csv")
# # # smote_rf = smote_rf.drop(columns="inputfile")
# # smote_rf = smote_rf.T
# #
# # smote_nb = pd.read_csv("D:\\distance metric\\smote\\combined smote nb auc.csv")
# # # smote_nb = smote_nb.drop(columns="inputfile")
# # smote_nb = smote_nb.T
# #
# # smote_log = pd.read_csv("D:\\distance metric\\smote\\combined smote log auc.csv")
# # # smote_log = smote_log.drop(columns="inputfile")
# # smote_log = smote_log.T
# # all_data=[np.random.normal(0,std,100) for std in range(1,4)]
# # meanlineprops = dict(color='yellow')
# # classifiers = data["classifier"]
# # auc = data["auc"]
# # technique = data["technique"]
# # sns.boxplot(x=data["classifier"], hue=technique, data=auc, linewidth=1)
# #                     # order=['KNN', 'RF', 'LR', 'NB'],
# #                     # hue_order=['ROCT', 'NCL', 'IKMCCA', 'ENN', 'Tomek'])
# plt.show()
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 设置字体
plt.rc('font', family='Times New Roman')


def prepare_boxplot_data():
    """
    读取所有方法与分类器的数据，整理为 boxplot 所需的 DataFrame
    """
    classifiers = ["knn", "lr", "nb", "rf", "tree"]

    method_display_names = {
        "ufidsf": "UFIDSF",
        "sb_gan": "SB-GAN",
        "cluster": "Cluster",
        "rus": "RUS",
        "borderline_smote": "Borderline",
        "smote": "SMOTE",
        "AIIUS": "AIIUS"
    }

    metric_column = "pf"
    all_data = []

    for classifier in classifiers:

        # ===== 我们的方法 AIIUS =====
        our_file_path = f"AIIUS-4.0\\mean\\{classifier}_mean_results.csv"
        try:
            our_df = pd.read_csv(our_file_path, comment='#')
            if metric_column in our_df.columns:
                for val in our_df[metric_column].dropna():
                    all_data.append({
                        "Classifier": classifier.upper(),
                        "Technique": "AIIUS",
                        "Value": val
                    })
        except Exception as e:
            print(f"读取失败: {our_file_path} | {e}")

        # ===== 对比方法 =====
        comparison_methods = {
            "ufidsf": f"ufidsf_results_metrics-final-multi-dataset\\{classifier}_ufidsf_results.csv",
            "sb_gan": f"sb_gan_cv_results_metrics-final\\{classifier}_sb_gan_results.csv",
            "cluster": f"cluster\\cluster_{classifier}.csv",
            "rus": f"rus\\{classifier} rus 0 .csv",
            "borderline_smote": f"borderline-smote\\{classifier} bor_smote 0 .csv",
            "smote": f"smote\\{classifier} smote 0 .csv"
        }

        for method, path in comparison_methods.items():
            try:
                df = pd.read_csv(path, comment='#')
                if metric_column in df.columns:
                    for val in df[metric_column].dropna():
                        all_data.append({
                            "Classifier": classifier.upper(),
                            "Technique": method_display_names[method],
                            "Value": val
                        })
            except Exception as e:
                print(f"读取失败: {path} | {e}")

    return pd.DataFrame(all_data)


def create_boxplot_with_internal_legend():
    """
    创建盒图（图例位于图内上方），并正确保存图片
    """
    tips = prepare_boxplot_data()
    if tips.empty:
        print("无可绘制数据")
        return None

    technique_order = [
        "AIIUS", "UFIDSF", "SB-GAN",
        "Cluster", "RUS", "Borderline", "SMOTE"
    ]

    tips = tips[tips["Technique"].isin(technique_order)]
    tips["Technique"] = pd.Categorical(
        tips["Technique"],
        categories=technique_order,
        ordered=True
    )

    # ===== 创建 Figure / Axes（关键）=====
    fig, ax = plt.subplots(figsize=(12, 8))

    sns.boxplot(
        x="Classifier",
        y="Value",
        hue="Technique",
        data=tips,
        palette="Set3",
        showmeans=True,
        whis=100,
        meanprops={
            "marker": "^",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markersize": 6
        },
        medianprops={
            "color": "black",
            "linewidth": 1.5
        },
        linewidth=1,
        ax=ax
    )

    # 坐标轴
    ax.set_xlabel("Classifier", fontsize=14)
    ax.set_ylabel("PF", fontsize=14)
    # ax.set_ylim(-0.05, 1.05)

    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    # 网格
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    # 图例（内部上方，一行）
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=7,
        fontsize=10,
        frameon=True,
        fancybox=True,
        edgecolor="black",
        framealpha=0.9
    )

    # ===== 关键顺序：先 tight_layout，再 savefig，最后 show =====
    plt.tight_layout()
    fig.savefig("boxplot_internal_legend.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("图像已成功保存为 boxplot_internal_legend.png")
    return tips


# ================= 主程序 =================
if __name__ == "__main__":
    print("开始绘制盒图...")
    data_df = create_boxplot_with_internal_legend()

    if data_df is not None:
        print("\n数据统计：")
        print("分类器:", sorted(data_df["Classifier"].unique()))
        print("方法:", sorted(data_df["Technique"].unique()))
        print("\n每组样本数：")
        print(
            data_df.groupby(["Classifier", "Technique"])
                   .size()
                   .unstack()
                   .fillna(0)
                   .astype(int)
        )
