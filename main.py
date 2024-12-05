import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# 定义数据文件的路径
file_path = r"/Users/linlu/Documents/Project/DataAnalyze/dataset/data.csv"
# 使用pandas读取CSV文件并将数据加载到df DataFrame中
df = pd.read_csv(file_path)

# 查看数据的描述性统计信息
print(df.describe())

# 查看数据的前5行，快速了解数据的基本结构和内容
df.head()
# 查看每列的缺失值数量，帮助了解数据中缺失情况
df.isnull().sum()

# 删除存在缺失值的几列
df.drop(["OnlineSecurity"], axis=1, inplace=True)
df.drop(["OnlineBackup"], axis=1, inplace=True)
df.drop(["DeviceProtection"], axis=1, inplace=True)
df.drop(["TechSupport"], axis=1, inplace=True)
df.drop(["StreamingTV"], axis=1, inplace=True)
df.drop(["StreamingMovies"], axis=1, inplace=True)

# 删除包含任何缺失值的行，how='any'表示只要一行中有缺失值就删除，inplace=True表示直接修改df
df.dropna(how='any', inplace=True)

# 删除不必要的列
df.drop(['MultipleLines', 'InternetService', 'Contract', 'PaperlessBilling', 'PaymentMethod'], axis=1, inplace=True)

# 检查数据类型及各列情况，找出可能存在问题的列（将所有列的数据类型打印出来查看）
print(df.info())

# 找出所有数据类型为对象（通常是分类变量）的列，排除已经处理过的Churn列
categorical_cols = df.select_dtypes(include='object').columns.difference(['Churn'])

# 对这些分类变量进行独热编码
dumm = pd.get_dummies(df[categorical_cols], drop_first=True)

# 将独热编码后的列合并回原数据集
df = pd.concat([df, dumm], axis=1)

# 提取Churn列作为目标变量y，然后再删除原始分类变量列
y = df['Churn']
df.drop(categorical_cols.tolist() + ['Churn'], axis=1, inplace=True)

df.head()

# 再次检查数据类型及各列情况，重点关注可能出现问题的列（将所有列的数据类型打印出来查看）
print(df.info())










# 处理空格字符串
for column in df.columns:
    unique_values = df[column].unique()
    if '' in unique_values or ' ' in unique_values:
        # 将空格或仅含空格的字符串替换为NaN
        df[column] = df[column].replace(r'^\s*$', np.nan, regex=True)
    # 这里不仅删除包含NaN的行，还处理了其他可能的数据类型转换问题
    try:
        df[column] = df[column].astype(float)
    except ValueError:
        df[column] = df[column].astype(str)
        df[column] = df[column].str.strip()  # 去除字符串两端空白字符
        df[column] = pd.to_numeric(df[column], errors='coerce')  # 将无法转为数字的部分设为NaN
    df.dropna(subset=[column], inplace=True)  # 删除NaN所在行

# 划分特征（X）和目标变量（y）
X = df.drop(columns=['目标列名'])  # 假设目标列名为'目标列名'，请替换为实际的列名
y = df['目标列名']

# 数据集分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义要尝试的不同参数组合来优化决策树模型
depth_list = [3, 5, 7]
min_samples_leaf_list = [3, 5, 7]

best_accuracy = 0
best_params = None

for max_depth in depth_list:
    for min_samples_leaf in min_samples_leaf_list:
        # 创建决策树分类器对象
        clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100,
                                          max_depth=max_depth, min_samples_leaf=min_samples_leaf)

        # 模型训练
        clf_gini.fit(X_train, y_train)

        # 用测试集进行预测
        y_pre = clf_gini.predict(X_test)

        # 计算准确率
        accuracy = accuracy_score(y_test, y_pre)

        # 更新最佳参数和准确率
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf}

print(f"最佳决策树参数：{best_params}，最佳准确率：{best_accuracy}")

# 使用最佳参数重新训练决策树模型
clf_gini_best = DecisionTreeClassifier(criterion="gini", random_state=100, **best_params)
clf_gini_best.fit(X_train, y_train)

# 用测试集预测
y_pre_best = clf_gini_best.predict(X_test)

# 混淆矩阵
confsMat_clf_gini_best = confusion_matrix(y_test, y_pre_best)
font = {'family': 'FangSong', 'size': 15}
matplotlib.rc('font', **font)

# 绘图函数
def painting(name, model, confsMat, x_test, y_test, file):
    # 绘制混淆矩阵
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.heatmap(confsMat, annot=True, fmt="d", cmap='Blues')
    plt.title(f"{name}混淆矩阵")

    # 设置坐标轴标签字体大小和颜色
    plt.xlabel("预测值", fontsize=12, color='black')
    plt.ylabel("实际值", fontsize=12, color='black')

    # 绘制ROC曲线
    plt.subplot(1, 2, 2)
    y_pre_proba = model.predict_proba(x_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pre_proba[:, 1])
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"ROC curve(area={auc:.2f})")
    plt.plot([0, 1], [0, 1], "--", color="r")
    plt.legend(loc='lower right')
    plt.title(f"{name}：Receiver Operating Characteristic")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(file, bbox_inches='tight', pad_inches=0)
    plt.show()

print(classification_report(y_test, y_pre_best))

painting("决策树", clf_gini_best, confsMat_clf_gini_best, X_test, y_test, "1.png")

# 随机森林优化
parameters = {
    "n_estimators": [100],  # 增加树的数量
    "max_depth": [5, 10, 20]
}
forest_clf = GridSearchCV(RandomForestClassifier(oob_score=True), param_grid=parameters, scoring="accuracy")
forest_clf.fit(X_train, y_train)

# 使用训练好的随机森林模型（通过GridSearchCV选择最优参数后的）对测试集进行预测
random_tree_pred = forest_clf.predict(X_test)

# 计算随机森林模型的混淆矩阵
confsMat_RandomForest = confusion_matrix(y_test, random_tree_pred)

# 打印随机森林模型的分类报告
print(metrics.classification_report(y_test, random_tree_pred))

# 调用函数绘制随机森林模型的混淆矩阵和ROC曲线并保存图形
painting("随机森林", forest_clf, confsMat_RandomForest, X_test, y_test, "2.png")
