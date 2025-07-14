import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import os


class AdaptiveGestureLearner:
    def __init__(self, base_model_path):
        self.base_model = load(base_model_path)
        self.user_models = {}  # 用户ID -> 个性化模型
        self.user_data = {}  # 用户ID -> 数据缓冲区
        self.scaler = StandardScaler()

    def initialize_user_model(self, user_id):
        """为指定用户创建个性化模型副本"""
        user_model = SGDClassifier(
            loss='log',  # 逻辑回归，适用于增量学习
            warm_start=True,
            class_weight='balanced'
        )
        # 初始化模型参数
        user_model.coef_ = self.base_model.coef_.copy()
        user_model.intercept_ = self.base_model.intercept_.copy()
        user_model.classes_ = np.arange(11)  # 0-10个手势

        self.user_models[user_id] = user_model
        self.user_data[user_id] = {'X': [], 'y': []}
        return user_model

    def add_sample(self, user_id, features, label):
        """添加新样本到用户缓冲区"""
        if user_id not in self.user_data:
            self.initialize_user_model(user_id)

        # 标准化特征
        features = self.scaler.transform([features])[0]
        self.user_data[user_id]['X'].append(features)
        self.user_data[user_id]['y'].append(label)

        # 缓冲区满时触发训练
        if len(self.user_data[user_id]['y']) >= 20:
            self.update_user_model(user_id)

    def update_user_model(self, user_id):
        """用缓冲区数据更新用户模型"""
        X = np.array(self.user_data[user_id]['X'])
        y = np.array(self.user_data[user_id]['y'])

        # 增量学习
        self.user_models[user_id].partial_fit(X, y)

        # 清空缓冲区
        self.user_data[user_id] = {'X': [], 'y': []}

        # 保存更新后的模型
        dump(self.user_models[user_id], f"models/user_models/user_{user_id}.pkl")

    def predict(self, user_id, features):
        """使用个性化模型预测"""
        if user_id not in self.user_models:
            return self.base_model.predict([features])[0]

        features = self.scaler.transform([features])[0]
        return self.user_models[user_id].predict([features])[0]

    def save_scaler(self, path):
        """保存标准化器"""
        dump(self.scaler, path)

    def load_scaler(self, path):
        """加载标准化器"""
        self.scaler = load(path)