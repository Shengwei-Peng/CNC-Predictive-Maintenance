import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple, List
from types import SimpleNamespace
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from imblearn.over_sampling import SMOTE

class CNC():
    def __init__(self, args: SimpleNamespace):
        self.args = args
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
    
    def pre_process(self):
        data = pd.read_csv(self.args.file_path)
        data['time'] = pd.to_datetime(data['time'])
        data['time'] = data['time'].astype(int) // 10**9
        self.data = self.oversample_data(data)
        x, y = self.create_rolling_features()
        y_list: List[np.ndarray] = [y[:, i] for i in range(self.args.future_steps)]
        x_train, x_test, *y_splits = train_test_split(x, *y_list, test_size=self.args.test_size, random_state=self.args.seed)
        y_train = [y_splits[i] for i in range(0, len(y_splits), 2)]
        y_test = [y_splits[i] for i in range(1, len(y_splits), 2)]

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        print(f"X_train shape: {self.x_train.shape}")
        print(f"X_test shape: {self.x_test.shape}")
        for i in range(self.args.future_steps):
            print(f"y_train[{i}] shape: {self.y_train[i].shape}")
            print(f"y_test[{i}] shape: {self.y_test[i].shape}")
    
    def train(self):
        self.models: List[RandomForestClassifier] = []
        for i in  tqdm(range(self.args.future_steps), desc="Training models"):
            model = RandomForestClassifier(random_state=self.args.seed)
            model.fit(self.x_train, self.y_train[i])
            self.models.append(model)
    
    def evaluate(self):
        self.y_preds = []
        self.y_probas = []
        for i in range(self.args.future_steps):
            y_pred = self.models[i].predict(self.x_test)
            y_proba = self.models[i].predict_proba(self.x_test)[:, 1]
            self.y_preds.append(y_pred)
            self.y_probas.append(y_proba)
            report = classification_report(self.y_test[i], y_pred)
            print(f"Report - Future step {i+1}:")
            print(report)
    
    def visualize(self):
        for i in range(self.args.future_steps):
            cm = confusion_matrix(self.y_test[i], self.y_preds[i])
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix - Future step {i+1}')
            
            fpr, tpr, _ = roc_curve(self.y_test[i], self.y_probas[i])
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Receiver Operating Characteristic - Future step {i+1}')
            plt.legend(loc="lower right")

            precision, recall, _ = precision_recall_curve(self.y_test[i], self.y_probas[i])
            plt.figure()
            plt.plot(recall, precision, color='blue', lw=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - Future step {i+1}')
            
        plt.show()

    def oversample_data(self, data: pd.DataFrame) -> pd.DataFrame:
        smote = SMOTE(random_state=self.args.seed)
        features = data.drop(columns=['Anomaly'])
        labels = data['Anomaly']
        features_resampled, labels_resampled = smote.fit_resample(features, labels)
        data_resampled = pd.concat([pd.DataFrame(features_resampled), pd.DataFrame(labels_resampled, columns=['Anomaly'])], axis=1)
        return data_resampled

    def create_rolling_features(self) -> Tuple[np.ndarray, np.ndarray]:
        x, y = [], []
        for i in tqdm(range(len(self.data) - self.args.window_size - self.args.future_steps + 1), desc="Creating rolling window features"):
            features = self.data.iloc[i:i + self.args.window_size].drop(columns=['Anomaly']).values.flatten()
            labels = self.data.iloc[i + self.args.window_size:i + self.args.window_size + self.args.future_steps]['Anomaly'].values
            x.append(features)
            y.append(labels)
        return np.array(x), np.array(y)
