# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('wine_quality_combined.csv')
print("数据形状:", df.shape)
print("\n前5行:")
print(df.head())

print("\n数据信息:")
print(df.info())

print("\n描述性统计:")
print(df.describe())

print("\n缺失值:")
print(df.isnull().sum())

print("\n质量分布:")
print(df['quality'].value_counts().sort_index())

# 处理异常值
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'quality' in numeric_cols:
    numeric_cols.remove('quality')

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower, upper)

# type编码
df['type_encoded'] = df['type'].map({'red': 0, 'white': 1})
if df['type_encoded'].isna().any():
    df['type_encoded'] = (df['type'] == 'white').astype(int)

# 质量等级（报告里用）
df['quality_level'] = pd.cut(df['quality'], bins=[0, 5, 7, 10], labels=['低', '中', '高'])

# 二分类目标
df['is_good'] = (df['quality'] >= 7).astype(int)

# 特征列
feature_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                'pH', 'sulphates', 'alcohol', 'type_encoded']

# 标准化
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[feature_cols])
scaled_df = pd.DataFrame(scaled_data, columns=[f'{col}_scaled' for col in feature_cols])
df = pd.concat([df, scaled_df], axis=1)

# 按质量等级分组
print("\n各特征按质量等级均值:")
print(df.groupby('quality_level')[feature_cols].mean().round(3))

# 相关性
print("\n各特征与质量的相关性:")
corr = df[feature_cols + ['quality']].corr()['quality'].sort_values(ascending=False)
print(corr)

# 可视化
import os
if not os.path.exists('figures'):
    os.makedirs('figures')

# 图1:质量分布
plt.figure(figsize=(10, 6))
df['quality'].value_counts().sort_index().plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('葡萄酒质量评分分布', fontsize=14)
plt.xlabel('质量评分')
plt.ylabel('频数')
for i, v in enumerate(df['quality'].value_counts().sort_index().values):
    plt.text(i, v + 10, str(v), ha='center')
plt.tight_layout()
plt.savefig('figures/quality_distribution.png', dpi=300)
plt.show()

# 图2:相关性热力图
plt.figure(figsize=(12, 10))
corr_matrix = df[feature_cols + ['quality']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True)
plt.title('特征相关性热力图', fontsize=14)
plt.tight_layout()
plt.savefig('figures/correlation_heatmap.png', dpi=300)
plt.show()

# 图3:箱线图
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
key_features = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid', 'density', 'residual sugar']
for idx, feature in enumerate(key_features):
    row, col = idx // 3, idx % 3
    data = [df[df['quality_level'] == '低'][feature].dropna(),
            df[df['quality_level'] == '中'][feature].dropna(),
            df[df['quality_level'] == '高'][feature].dropna()]
    axes[row, col].boxplot(data, labels=['低', '中', '高'])
    axes[row, col].set_title(feature, fontsize=12)
    axes[row, col].grid(True, alpha=0.3)
plt.suptitle('关键特征在不同质量等级下的分布', fontsize=14)
plt.tight_layout()
plt.savefig('figures/boxplots.png', dpi=300)
plt.show()

# 随机森林特征重要性
X_cols = [col for col in df.columns if col.endswith('_scaled')]
X = df[X_cols]
y = df['is_good']

rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
rf_temp.fit(X, y)

plt.figure(figsize=(10, 6))
imp_df = pd.DataFrame({'特征': X.columns, '重要性': rf_temp.feature_importances_}).sort_values('重要性')
plt.barh(imp_df['特征'], imp_df['重要性'], color='coral')
plt.title('特征重要性排序（预测优质酒）', fontsize=14)
plt.xlabel('重要性得分')
plt.tight_layout()
plt.savefig('figures/feature_importance.png', dpi=300)
plt.show()

# 建模
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n训练集:{X_train.shape[0]}个, 测试集:{X_test.shape[0]}个")

# 逻辑回归
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("\n逻辑回归:")
print("准确率:", round(accuracy_score(y_test, y_pred_lr), 4))
print("F1分数:", round(f1_score(y_test, y_pred_lr), 4))
print(classification_report(y_test, y_pred_lr, target_names=['非优质', '优质']))

# 随机森林
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("\n随机森林:")
print("准确率:", round(accuracy_score(y_test, y_pred_rf), 4))
print("F1分数:", round(f1_score(y_test, y_pred_rf), 4))
print(classification_report(y_test, y_pred_rf, target_names=['非优质', '优质']))

# 交叉验证
cv_lr = cross_val_score(lr, X, y, cv=5, scoring='f1')
cv_rf = cross_val_score(rf, X, y, cv=5, scoring='f1')
print("\n交叉验证F1均值:")
print("逻辑回归:", round(cv_lr.mean(), 4), "(±", round(cv_lr.std(), 4), ")")
print("随机森林:", round(cv_rf.mean(), 4), "(±", round(cv_rf.std(), 4), ")")

# 混淆矩阵
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['非优质', '优质'], yticklabels=['非优质', '优质'])
plt.title('随机森林混淆矩阵', fontsize=14)
plt.xlabel('预测值')
plt.ylabel('真实值')
plt.tight_layout()
plt.savefig('figures/confusion_matrix.png', dpi=300)
plt.show()

# 保存
df.to_csv('wine_data_processed.csv', index=False)
print("\n处理后的数据已保存: wine_data_processed.csv")
print("图表已保存至 figures/ 目录")