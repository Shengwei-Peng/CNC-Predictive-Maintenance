import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple, List
from types import SimpleNamespace
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    make_scorer,
    recall_score,
    precision_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)


class CNC():
    def __init__(self, args: SimpleNamespace):
        self.args = args
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        self.model_map = {
            "NB": {
                "name": "Gaussian Naive Bayes",
                "model": GaussianNB()
            },
            "KNN": {
                "name": "K-Nearest Neighbors",
                "model": KNeighborsClassifier()
            },
            "DT": {
                "name": "Decision Tree",
                "model": DecisionTreeClassifier(random_state=self.args.seed)
            },
            "RF": {
                "name": "Random Forest",
                "model": RandomForestClassifier(random_state=self.args.seed)
            },
            "SVM": {
                "name": "Support Vector Machine",
                "model": SVC(probability=True, random_state=self.args.seed)
            },
            "MLP": {
                "name": "Multi-Layer Perceptron",
                "model": MLPClassifier(random_state=self.args.seed)
            }
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
            model = self.model_map[self.args.model]["model"]
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
            self._plot_roc_pr_curves(i)
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
        _, ax = plt.subplots(1, 2, figsize=(12, 6))
        labels, counts = np.unique(y, return_counts=True)
        label_names = ['Non-Anomaly', 'Anomaly']
        colors = ['#ff9999','#66b3ff']
        ax[0].pie(counts, labels=[f'{label_names[label]} ({count})' for label, count in zip(labels, counts)], autopct='%1.1f%%', startangle=90, colors=colors)
        ax[0].set_title(f'Before')
        smote = SMOTE(random_state=self.args.seed)
        x_resampled, y_resampled = smote.fit_resample(x, y)
        labels_resampled, counts_resampled = np.unique(y_resampled, return_counts=True)
        ax[1].pie(counts_resampled, labels=[f'{label_names[label]} ({count})' for label, count in zip(labels_resampled, counts_resampled)], autopct='%1.1f%%', startangle=90, colors=colors)
        ax[1].set_title(f'After')
        plt.suptitle('Class Distribution in Training Dataset', fontsize=16)
        plt.tight_layout()
        plt.show()
        return x_resampled, y_resampled

    def _split_data(self, x: np.ndarray, y: np.ndarray):
        y_list: List[np.ndarray] = [y[:, i] for i in range(self.args.future_steps)]
        self.x_train, self.x_test, *y_splits = train_test_split(x, *y_list, test_size=self.args.test_size, random_state=self.args.seed)
        self.y_train = [y_splits[i] for i in range(0, len(y_splits), 2)]
        self.y_test = [y_splits[i] for i in range(1, len(y_splits), 2)]

    def _plot_confusion_matrix(self, i: int):
        model_name = self.model_map[self.args.model]["name"]
        cm = confusion_matrix(self.y_test[i], self.y_preds[i])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, xticklabels=['Non-Anomaly', 'Anomaly'], yticklabels=['Non-Anomaly', 'Anomaly'])
        plt.xlabel('Predicted', fontsize=14)
        plt.ylabel('Actual', fontsize=14)
        plt.title(f'Confusion Matrix of the {model_name} - Future step {i+1}', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        
    def _plot_roc_pr_curves(self, i: int):
        pos_label, neg_label = True, False
        model_name = self.model_map[self.args.model]["name"]
        def fpr_score(y, y_pred, neg_label, pos_label):
            cm = confusion_matrix(y, y_pred, labels=[neg_label, pos_label])
            tn, fp, _, _ = cm.ravel()
            tnr = tn / (tn + fp)
            return 1 - tnr
        tpr_score = recall_score
        scoring = {
            "precision": make_scorer(precision_score, pos_label=pos_label),
            "recall": make_scorer(recall_score, pos_label=pos_label),
            "fpr": make_scorer(fpr_score, neg_label=neg_label, pos_label=pos_label),
            "tpr": make_scorer(tpr_score, pos_label=pos_label),
        }
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
        pr_display = PrecisionRecallDisplay.from_estimator(
            self.models[i], self.x_test, self.y_test[i], pos_label=pos_label, ax=axs[0], name=model_name
        )
        axs[0].plot(
            scoring["recall"](self.models[i], self.x_test, self.y_test[i]),
            scoring["precision"](self.models[i], self.x_test, self.y_test[i]),
            marker="o",
            markersize=10,
            color="tab:blue",
            label="Default cut-off point at a probability of 0.5",
        )
        axs[0].fill_between(pr_display.recall, pr_display.precision, step='post', alpha=0.2, color="b", label=f"AP area")
        axs[0].set_title("Precision-Recall curve")
        axs[0].legend()
        axs[0].grid(True)
        roc_display = RocCurveDisplay.from_estimator(
            self.models[i], 
            self.x_test, 
            self.y_test[i],
            pos_label=pos_label,
            ax=axs[1],
            name=model_name,
            plot_chance_level=True,
        )
        axs[1].plot(
            scoring["fpr"](self.models[i], self.x_test, self.y_test[i]),
            scoring["tpr"](self.models[i], self.x_test, self.y_test[i]),
            marker="o",
            markersize=10,
            color="tab:blue",
            label="Default cut-off point at a probability of 0.5",
        )
        axs[1].fill_between(roc_display.fpr, roc_display.tpr, alpha=0.2, color="b", label=f"AUC area")
        axs[1].set_title("ROC curve")
        axs[1].legend()
        axs[1].grid(True)
        fig.suptitle(f"Evaluation of the {model_name} - Future step {i+1}")
