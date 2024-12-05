import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, f1_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import LabelEncoder
from matplotlib import font_manager

# 定义数据集路径
file_path = r"/Users/linlu/Documents/Project/DataAnalyze/dataset/data.csv"

# 读取数据集
df = pd.read_csv(file_path)
df.head()

# 查看数据的描述性统计信息及部分数据信息
print(df.describe())
print(df.head())

# 进行缺失值处理
df.isnull().sum()
# 查看缺失值
print(df.isnull().sum())

# 存在缺失值则删除
df.dropna(how='any', inplace=True)

# 处理空格字符串
for column in df.columns:
    unique_values = df[column].unique()
    if '' in unique_values or ' ' in unique_values:
        # 将空格或仅含空格的字符串替换为NaN
        df[column] = df[column].replace(r'^\s*$', np.nan, regex=True)

    # 针对数值型数据进行处理
    if df[column].dtype in ['float64', 'int64']:
        # 填充数值型列的缺失值为均值
        df[column] = df[column].fillna(df[column].mean())
    else:
        # 对于非数值型数据，进行模式填充
        df[column] = df[column].fillna(df[column].mode()[0])

# 删除含有任何缺失值的行
df.dropna(inplace=True)

# 对类别数据进行编码
categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                       'PaymentMethod', 'Churn']

label_encoder = LabelEncoder()
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

# 输出数据预处理后的信息
print(df.info())

# 确保数据不为空
if df.empty:
    print("数据集为空。")
else:
    print("数据集已处理完成。")

# 检查删除缺失值后的数据集大小
print(f"数据集大小：{df.shape}")

# 划分特征和目标变量
X = df.drop(columns=['customerID', 'Churn'])
y = df['Churn']

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
matplotlib.rc('font', family= ['SimHei', 'Songti SC'])

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
    plt.title(f"{name}")
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
