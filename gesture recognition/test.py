# 测试集评估
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 加载保存的最佳模型
import joblib
best_model = joblib.load('gesture_model.pkl')

# 测试集预测
y_pred = best_model.predict(X_test)
y_true = y_test

# 生成评估报告
print("## 模型性能报告 ##")
print(classification_report(y_true, y_pred))

# 绘制混淆矩阵
plt.figure(figsize=(10,8))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=gesture_labels,
            yticklabels=gesture_labels)
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('手势识别混淆矩阵')
plt.savefig('confusion_matrix.png')