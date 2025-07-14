import numpy as np
from sklearn.base import clone
from sklearn.linear_model import SGDClassifier
from joblib import dump, load
import os
from collections import defaultdict


class AdaptiveGestureLearner:
    def __init__(self, base_model_path):
        self.base_model = load(base_model_path)
        self.user_models = {}  # 用户ID -> 个性化模型
        self.user_data = defaultdict(lambda: {'X': [], 'y': []})  # 用户数据缓冲区
        self.scaler = None

    def initialize_user_model(self, user_id):
        """为指定用户创建个性化模型副本"""
        # 使用SGDClassifier进行增量学习
        user_model = SGDClassifier(
            loss='log_loss',  # 逻辑回归，适用于概率预测
            warm_start=True,  # 允许增量学习
            class_weight='balanced',
            eta0=0.1,  # 初始学习率
            learning_rate='adaptive',
            early_stopping=True,
            n_iter_no_change=5,
            max_iter=1000,
            random_state=42
        )

        # 初始化模型参数
        user_model.classes_ = np.arange(11)  # 0-10个手势
        user_model.coef_ = self.base_model.coef_.copy() if hasattr(self.base_model, 'coef_') else None
        user_model.intercept_ = self.base_model.intercept_.copy() if hasattr(self.base_model, 'intercept_') else None

        self.user_models[user_id] = user_model
        return user_model

    def add_sample(self, user_id, features, label):
        """添加新样本到用户缓冲区"""
        if user_id not in self.user_models:
            self.initialize_user_model(user_id)

        # 添加到缓冲区
        self.user_data[user_id]['X'].append(features)
        self.user_data[user_id]['y'].append(label)

        # 检查缓冲区是否满
        buffer_size = len(self.user_data[user_id]['y'])

        # 显示缓冲区状态
        print(f"用户 {user_id} 缓冲区: {buffer_size}/20 样本")

        # 缓冲区满时触发训练
        if buffer_size >= 20:
            self.update_user_model(user_id)

    def update_user_model(self, user_id):
        """用缓冲区数据更新用户模型"""
        print(f"更新用户 {user_id} 的模型...")

        X = np.array(self.user_data[user_id]['X'])
        y = np.array(self.user_data[user_id]['y'])

        # 标准化特征
        X_scaled = self.scaler.transform(X)

        # 增量学习
        self.user_models[user_id].partial_fit(X_scaled, y, classes=np.arange(11))

        # 清空缓冲区
        self.user_data[user_id] = {'X': [], 'y': []}

        # 保存更新后的模型
        self.save_user_model(user_id)

        print(f"用户 {user_id} 模型更新完成!")

    def save_user_model(self, user_id):
        """保存用户模型"""
        if user_id in self.user_models:
            os.makedirs("models/user_models", exist_ok=True)
            dump(self.user_models[user_id], f"models/user_models/user_{user_id}.pkl")

    def predict(self, user_id, features):
        """使用个性化模型预测"""
        # 标准化特征
        features_scaled = self.scaler.transform([features])[0]

        if user_id in self.user_models and len(self.user_data[user_id]['y']) > 5:
            # 使用个性化模型
            return self.user_models[user_id].predict([features_scaled])[0]
        else:
            # 使用基础模型
            return self.base_model.predict([features_scaled])[0]

    def load_scaler(self, path):
        """加载标准化器"""
        self.scaler = load(path)