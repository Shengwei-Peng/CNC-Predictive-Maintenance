import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple, List
from types import SimpleNamespace
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from imblearn.over_sampling import SMOTE

class CNC():
    def __init__(self, args: SimpleNamespace):
        self.args = args
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        self.model_map = {
            "NB": GaussianNB(),
            "KNN": KNeighborsClassifier(),
            "DT": DecisionTreeClassifier(random_state=self.args.seed),
            "RF": RandomForestClassifier(random_state=self.args.seed),
            "SVM": SVC(probability=True, random_state=self.args.seed),
            "MLP": MLPClassifier(random_state=self.args.seed),
        }

    def pre_process(self):
        self.data = pd.read_csv(self.args.file_path)
        self.data['time'] = pd.to_datetime(self.data['time'])
        self.data['time'] = self.data['time'].astype(int) // 10**9
        x, y = self._create_rolling_features()
        self._split_data(x, y)

    def train(self):
        self.models = []
        for i in tqdm(range(self.args.future_steps), desc="Training models"):
            model = self.model_map.get(self.args.model)
            x, y = self.x_train, self.y_train[i]
            if model is None:
                raise ValueError(f"Unknown model type: {self.args.model}")
            if self.args.over_sampling:
                x, y = self._over_sampling(x, y)
            model.fit(x, y)
            self.models.append(model)

    def evaluate(self):
        self.y_preds = []
        self.y_probas = []
        for i in range(self.args.future_steps):
            y_proba = self.models[i].predict_proba(self.x_test)[:, 1]
            y_pred = (y_proba >= self.args.threshold).astype(int)
            self.y_preds.append(y_pred)
            self.y_probas.append(y_proba)
            report = classification_report(self.y_test[i], y_pred)
            print(f"Report - Future step {i+1}:")
            print(report)

    def visualize(self):
        for i in range(self.args.future_steps):
            self._plot_confusion_matrix(i)
            self._plot_roc_curve(i)
            self._plot_precision_recall_curve(i)
            self._plot_recall_vs_threshold(i)
        plt.show()

    def _create_rolling_features(self) -> Tuple[np.ndarray, np.ndarray]:
        x, y = [], []
        for i in tqdm(range(len(self.data) - self.args.window_size - self.args.future_steps + 1), desc="Creating rolling window features"):
            features = self.data.iloc[i:i + self.args.window_size].drop(columns=['Anomaly']).values.flatten()
            labels = self.data.iloc[i + self.args.window_size:i + self.args.window_size + self.args.future_steps]['Anomaly'].values
            x.append(features)
            y.append(labels)
        return np.array(x), np.array(y)

    def _over_sampling(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        smote = SMOTE(random_state=self.args.seed)
        x_resampled, y_resampled = smote.fit_resample(x, y)
        return x_resampled, y_resampled

    def _split_data(self, x: np.ndarray, y: np.ndarray):
        y_list: List[np.ndarray] = [y[:, i] for i in range(self.args.future_steps)]
        self.x_train, self.x_test, *y_splits = train_test_split(x, *y_list, test_size=self.args.test_size, random_state=self.args.seed)
        self.y_train = [y_splits[i] for i in range(0, len(y_splits), 2)]
        self.y_test = [y_splits[i] for i in range(1, len(y_splits), 2)]

    def _plot_confusion_matrix(self, i: int):
        cm = confusion_matrix(self.y_test[i], self.y_preds[i])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, xticklabels=['Non-Anomaly', 'Anomaly'], yticklabels=['Non-Anomaly', 'Anomaly'])
        plt.xlabel('Predicted', fontsize=14)
        plt.ylabel('Actual', fontsize=14)
        plt.title(f'Confusion Matrix - Future step {i+1}', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()

    def _plot_roc_curve(self, i: int):
        fpr, tpr, _ = roc_curve(self.y_test[i], self.y_probas[i])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title(f'Receiver Operating Characteristic - Future step {i+1}', fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.tight_layout()

    def _plot_precision_recall_curve(self, i: int):
        precision, recall, _ = precision_recall_curve(self.y_test[i], self.y_probas[i])
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title(f'Precision-Recall curve - Future step {i+1}', fontsize=16)
        plt.legend(loc="lower left", fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.tight_layout()

    def _plot_recall_vs_threshold(self, i: int):
        _, recall, thresholds = precision_recall_curve(self.y_test[i], self.y_probas[i])
        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, recall[:-1], "b--", label="Recall")
        plt.xlabel('Threshold', fontsize=14)
        plt.ylabel('Recall', fontsize=14)
        plt.title(f'Recall vs Threshold - Future step {i+1}', fontsize=16)
        plt.legend(loc="lower left", fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
