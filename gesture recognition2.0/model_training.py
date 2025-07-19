import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# --- 第一步：定义文件路径和目录 ---
# 将所有重要的路径和文件名定义为变量，方便管理
DATASET_DIR = 'gesture_dataset'
FEATURES_FILE = os.path.join(DATASET_DIR, 'features.npy')
LABELS_FILE = os.path.join(DATASET_DIR, 'labels.npy')
OUTPUT_DIR = 'models'
MODEL_FILE = os.path.join(OUTPUT_DIR, 'base_model.pkl')
SCALER_FILE = os.path.join(OUTPUT_DIR, 'standard_scaler.pkl')


def train_model():
    """
    加载特征数据，训练模型，并将其保存到文件。
    """
    # --- 第二步：检查并加载数据 ---
    # 检查必需的特征文件是否存在，如果不存在则给出清晰的错误提示
    if not os.path.exists(FEATURES_FILE) or not os.path.exists(LABELS_FILE):
        print(f"错误：找不到特征文件 '{FEATURES_FILE}' 或 '{LABELS_FILE}'。")
        print("请先运行 feature_extraction_image.py 或 feature_extraction_video.py 来生成它们。")
        return  # 提前退出函数

    print("正在加载特征和标签...")
    features = np.load(FEATURES_FILE)
    labels = np.load(LABELS_FILE)
    print(f"加载完成！共找到 {len(features)} 个样本。")

    # --- 第三步：数据预处理 ---
    # 将 (n_samples, 21, 3) 的数据“压平”成 (n_samples, 63)
    n_samples = features.shape[0]
    features_reshaped = features.reshape(n_samples, -1)

    # 将数据集分割为训练集和测试集 (80% 训练, 20% 测试)
    X_train, X_test, y_train, y_test = train_test_split(
        features_reshaped, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print("数据集分割完成。")

    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("特征标准化完成。")

    # --- 第四步：模型训练 ---
    # 初始化随机森林分类器
    # n_estimators=100 表示由100棵决策树组成
    # random_state=42 确保每次训练结果都一样，方便调试
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)

    print("开始训练基础模型...")
    base_model.fit(X_train_scaled, y_train)
    print("模型训练完成！")

    # --- 第五步：保存模型和标准化工具 (核心修改) ---
    # 在保存之前，检查输出目录是否存在，如果不存在，则创建它
    print(f"正在准备保存模型到 '{OUTPUT_DIR}' 文件夹...")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"文件夹 '{OUTPUT_DIR}' 不存在，已自动创建。")

    # 使用 joblib 保存训练好的模型和 scaler
    joblib.dump(base_model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"模型已成功保存到: {MODEL_FILE}")
    print(f"标准化工具已成功保存到: {SCALER_FILE}")

    # --- 第六步：评估模型性能 ---
    # 在测试集上进行预测并计算准确率
    y_pred = base_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print("-" * 30)
    print(f"模型评估结果：")
    print(f"在测试集上的准确率: {accuracy:.4f} ({accuracy:.2%})")
    print("-" * 30)


# 当这个脚本被直接运行时，才执行 train_model 函数
if __name__ == '__main__':
    train_model()