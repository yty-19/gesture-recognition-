import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from adaptive_learning import AdaptiveGestureLearner
import joblib

# 加载数据集
features = np.load('gesture_dataset/features.npy', allow_pickle=True)
labels = np.load('gesture_dataset/labels.npy')

# 转换为2D矩阵 (样本数, 帧数*关键点数*3)
X = np.array([x.flatten() for x in features])
y = np.array(labels)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练基础模型
base_model = RandomForestClassifier(n_estimators=100, random_state=42)
base_model.fit(X_train_scaled, y_train)

# 保存基础模型和标准化器
joblib.dump(base_model, 'models/base_model.pkl')
joblib.dump(scaler, 'models/standard_scaler.pkl')

# 初始化自适应学习系统并保存
adaptive_system = AdaptiveGestureLearner('models/base_model.pkl')
adaptive_system.scaler = scaler
adaptive_system.save_scaler('models/adaptive_scaler.pkl')

# 评估
y_pred = base_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Base Model Accuracy: {accuracy:.4f}")