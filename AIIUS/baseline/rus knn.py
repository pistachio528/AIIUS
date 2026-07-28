import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import neighbors
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score,matthews_corrcoef
import time
import csv
import warnings
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.naive_bayes import GaussianNB
np.set_printoptions(suppress=True)
warnings.filterwarnings('ignore')
classifier_for_selection = {"knn": neighbors.KNeighborsClassifier(n_neighbors=5), "svm": svm.SVC(), "rf": RandomForestClassifier(random_state=0), "tree": tree.DecisionTreeClassifier(random_state=0), "lr": LogisticRegression(random_state=0), "nb": GaussianNB()}
classifier = "knn"

skf = StratifiedKFold(n_splits=5)

#设置输出目录
output_dir = 'rus'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for iteration in range(1):
    single = open(os.path.join(output_dir,f'{classifier} rus {iteration} .csv'), 'a',
                  newline='')
    single_writer = csv.writer(single)
    single_writer.writerow(["inputfile", "auc", "balance", "pd", "pf","mcc", "defect ratio"])
    for inputfile in os.listdir("datasets\\relink dataset_csv"):
        # if inputfile == "jedit-4.3.csv" or inputfile == "synapse-1.0.csv" or inputfile == "synapse-1.1.csv":
        #     continue
        print("inputfile: ", inputfile)
        start_time = time.time()
        dataset = pd.read_csv("datasets\\relink dataset_csv\\"+ inputfile)
        # dataset = dataset.drop(columns="name")
        # dataset = dataset.drop(columns="version")
        # dataset = dataset.drop(columns="name.1")
        total_number = len(dataset)
        #defect_ratio = len(dataset[dataset["bug"] > 0]) / total_number
        # if defect_ratio > 0.5:  # 注意这点导致的有几个文件没有选上
        #     print(inputfile, " defect ratio larger than 0.45")
        #     continue

        for z in range(total_number):
            if dataset.loc[z, "isDefective"] == "buggy" :
                dataset.loc[z, "isDefective"] = 1
            else:
                dataset.loc[z, "isDefective"] = 0
        dataset['isDefective'] = dataset['isDefective'].astype(int)
        cols = list(dataset.columns)
        for col in cols:
            if col == "isDefective":
                continue
            dataset[col] = np.log(dataset[col] + 1)  # (dataset[col] - column_min) / (column_max - column_min)

        dataset = np.array(dataset)
        x = dataset
        y = dataset[:, -1]

        total_auc = 0
        total_balance = 0
        total_recall = 0
        total_pf = 0
        total_ratio = 0
        total_mcc = 0

        for i in range(10):
            for train, test in skf.split(x, y):
                test_x = x[test]
                test_x = test_x[:, 0:-1]
                test_y = y[test]

                train_dataset = x[train]
                train_x = x[train]
                train_x = train_x[:, 0:-1]
                train_y = y[train]

                rus = RandomUnderSampler()
                rus_train_x, rus_train_y = rus.fit_resample(train_x, train_y)

                total_ratio = total_ratio + (len(rus_train_y[rus_train_y == 1]) / len(rus_train_y))
                clf = classifier_for_selection[classifier]
                clf.fit(rus_train_x, rus_train_y)
                predict_result = clf.predict(test_x)
                true_negative, false_positive, false_negative, true_positive = confusion_matrix(test_y,
                                                                                                predict_result).ravel()

                auc = roc_auc_score(test_y, predict_result)
                total_auc = total_auc + auc

                recall = recall_score(test_y, predict_result)
                total_recall = total_recall + recall

                pf = false_positive / (true_negative + false_positive)
                total_pf = total_pf + pf

                balance = 1 - (((0 - pf) ** 2 + (1 - recall) ** 2) / 2) ** 0.5
                total_balance = total_balance + balance

                # 计算MCC
                mcc = matthews_corrcoef(test_y, predict_result)
                total_mcc = total_mcc + mcc

        average_auc = total_auc / 50
        average_balance = total_balance / 50
        average_recall = total_recall / 50
        average_pf = total_pf / 50
        average_ratio = total_ratio / 50
        average_mcc = total_mcc / 50
        single_writer.writerow([inputfile, average_auc, average_balance, average_recall, average_pf, average_mcc,average_ratio])
        print("final auc: ", average_auc)
        print("final mcc: ", average_mcc)
        print("--------------------------------------")







