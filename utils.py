import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple, List
from types import SimpleNamespace
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.metrics import (
    make_scorer,
    recall_score,
    precision_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    get_scorer,
)
import warnings
warnings.filterwarnings('ignore')


class CNC():
    def __init__(self, args: SimpleNamespace):
        self.args = args
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        self.model_map = {
            "LR": {
                "name": "Logistic Regression",
                "model": LogisticRegression()
            },
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
            "XGB": {
                "name": "eXtreme Gradient Boosting",
                "model": XGBClassifier(random_state=self.args.seed)
            },
        }
        self.sampler_map = {
            "SMOTE": SMOTE(random_state=self.args.seed),
            "ADASYN": ADASYN(random_state=self.args.seed),
            "BorderlineSMOTE": BorderlineSMOTE(random_state=self.args.seed),
            "RandomOverSampler": RandomOverSampler(random_state=self.args.seed),
            "RandomUnderSampler": RandomUnderSampler(random_state=self.args.seed),
        }

    def pre_process(self):
        self.data = pd.read_csv(self.args.file_path)
        self.data['time'] = pd.to_datetime(self.data['time'])
        self.data['time'] = self.data['time'].astype(int) // 10**9
        x, y = self._create_rolling_features()
        self._split_data(x, y)

    def train(self):
        self.models = []
        self.tuned_models = []
        for i in tqdm(range(self.args.future_steps), desc="Training models"):
            model = self.model_map[self.args.model]["model"]
            x, y = self.x_train, self.y_train[i]
            if model is None:
                raise ValueError(f"Unknown model type: {self.args.model}")
            if self.args.sampler is not None:
                x, y = self._sampling(x, y)
            model.fit(x, y)
            self.models.append(model)
            tuned_model = TunedThresholdClassifierCV(
                estimator=model,
                scoring="balanced_accuracy",
                store_cv_results=True,
                random_state=self.args.seed,
            )
            tuned_model.fit(x, y)
            self.tuned_models.append(tuned_model)

    def evaluate(self):
        self.y_preds = []
        balanced_accuracy_scorer = get_scorer("balanced_accuracy")
        for i in range(self.args.future_steps):
            
            vanilla_pred = self.models[i].predict(self.x_test)
            vanilla_report = classification_report(self.y_test[i], vanilla_pred, output_dict=True, target_names=['Non-Anomaly', 'Anomaly'])
            vanilla_acc = balanced_accuracy_scorer(self.models[i], self.x_test, self.y_test[i])
            
            print(f"Vanilla model Report - Future step {i+1}:")
            print(classification_report(self.y_test[i], vanilla_pred, target_names=['Non-Anomaly', 'Anomaly']))
            print(f"Balanced accuracy of Vanilla model: {vanilla_acc:.2%}")

            tuned_pred = self.tuned_models[i].predict(self.x_test)
            tuned_report = classification_report(self.y_test[i], tuned_pred, output_dict=True, target_names=['Non-Anomaly', 'Anomaly'])
            tuned_acc = balanced_accuracy_scorer(self.tuned_models[i], self.x_test, self.y_test[i])
            
            print(f"\nTuned model Report - Future step {i+1}:")
            print(classification_report(self.y_test[i], tuned_pred, target_names=['Non-Anomaly', 'Anomaly']))
            print(f"Balanced accuracy of Tuned model: {tuned_acc:.2%}")
            self._visualize_classification_report(i + 1, vanilla_report, tuned_report)
            
            model = self.tuned_models[i] if tuned_acc > vanilla_acc else self.models[i]
            y_pred = model.predict(self.x_test)
            self.y_preds.append(y_pred)
    
    def visualize(self):
        for i in range(self.args.future_steps):
            self._plot_confusion_matrix(i)
            self._plot_roc_pr_curves(i)
        plt.show()

    def _visualize_classification_report(self, step, vanilla_report, tuned_report):
        metrics = ['precision', 'recall', 'f1-score']
        classes = list(vanilla_report.keys())[:-3]
        n_classes = len(classes)
        n_metrics = len(metrics)
        
        _, ax = plt.subplots(figsize=(16, 9))
        width = 0.35
        x = np.arange(n_classes * n_metrics)

        vanilla_scores = []
        tuned_scores = []
        labels = []
        
        for metric in metrics:
            for cls in classes:
                vanilla_scores.append(vanilla_report[cls][metric])
                tuned_scores.append(tuned_report[cls][metric])
                labels.append(f'{cls} - {metric}')
        
        ax.bar(x - width/2, vanilla_scores, width, label='Vanilla', color='skyblue', edgecolor='black')
        ax.bar(x + width/2, tuned_scores, width, label='Tuned', color='orange', edgecolor='black')
        
        ax.set_ylabel('Scores')
        ax.set_title(f"Model Comparison Report - Step {step}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend(loc='best')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()

    def _create_rolling_features(self) -> Tuple[np.ndarray, np.ndarray]:
        x, y = [], []
        for i in tqdm(range(len(self.data) - self.args.window_size - self.args.future_steps + 1), desc="Creating rolling window features"):
            features = self.data.iloc[i:i + self.args.window_size].drop(columns=['Anomaly']).values.flatten()
            labels = self.data.iloc[i + self.args.window_size:i + self.args.window_size + self.args.future_steps]['Anomaly'].values
            x.append(features)
            y.append(labels)
        return np.array(x), np.array(y)

    def _sampling(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        _, ax = plt.subplots(1, 2, figsize=(12, 6))
        labels, counts = np.unique(y, return_counts=True)
        label_names = ['Non-Anomaly', 'Anomaly']
        colors = ['#ff9999','#66b3ff']
        ax[0].pie(counts, labels=[f'{label_names[label]} ({count})' for label, count in zip(labels, counts)], autopct='%1.1f%%', startangle=90, colors=colors)
        ax[0].set_title('Before')
        sampler = self.sampler_map.get(self.args.sampler)
        if sampler is None:
            raise ValueError(f"Unknown sampling method: {self.args.sampler}")
        x_resampled, y_resampled = sampler.fit_resample(x, y)
        labels_resampled, counts_resampled = np.unique(y_resampled, return_counts=True)
        ax[1].pie(counts_resampled, labels=[f'{label_names[label]} ({count})' for label, count in zip(labels_resampled, counts_resampled)], autopct='%1.1f%%', startangle=90, colors=colors)
        ax[1].set_title('After')
        plt.suptitle(self.args.sampler, fontsize=16)
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
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(21, 6))
        linestyles = ("dashed", "dotted")
        markerstyles = ("o", ">")
        colors = ("tab:blue", "tab:orange")
        names = (f"Vanilla {model_name}", f"Tuned {model_name}")
        for idx, (est, linestyle, marker, color, name) in enumerate(
        zip((self.models[i], self.tuned_models[i]), linestyles, markerstyles, colors, names)
        ):
            decision_threshold = getattr(est, "best_threshold_", 0.5)
            pr_display = PrecisionRecallDisplay.from_estimator(
                est,
                self.x_test,
                self.y_test[i],
                pos_label=pos_label,
                linestyle=linestyle,
                color=color,
                ax=axs[0],
                name=name,
            )
            axs[0].plot(
                scoring["recall"](est, self.x_test, self.y_test[i]),
                scoring["precision"](est, self.x_test, self.y_test[i]),
                marker,
                markersize=10,
                color=color,
                label=f"Cut-off point at probability of {decision_threshold:.2f}",
            )
            roc_display = RocCurveDisplay.from_estimator(
                est,
                self.x_test,
                self.y_test[i],
                pos_label=pos_label,
                linestyle=linestyle,
                color=color,
                ax=axs[1],
                name=name,
                plot_chance_level=idx == 1,
            )
            axs[1].plot(
                scoring["fpr"](est, self.x_test, self.y_test[i]),
                scoring["tpr"](est, self.x_test, self.y_test[i]),
                marker,
                markersize=10,
                color=color,
                label=f"Cut-off point at probability of {decision_threshold:.2f}",
            )
        axs[0].fill_between(pr_display.recall, pr_display.precision, step='post', alpha=0.2, color="b", label=f"AP area")
        axs[0].set_title("Precision-Recall curve")
        axs[0].legend()
        axs[0].grid(True)
        axs[1].fill_between(roc_display.fpr, roc_display.tpr, alpha=0.2, color="b", label=f"AUC area")
        axs[1].set_title("ROC curve")
        axs[1].legend()
        axs[1].grid(True)
        
        axs[2].plot(
        self.tuned_models[i].cv_results_["thresholds"],
        self.tuned_models[i].cv_results_["scores"],
        color="tab:orange",
        )
        axs[2].plot(
            self.tuned_models[i].best_threshold_,
            self.tuned_models[i].best_score_,
            "o",
            markersize=10,
            color="tab:orange",
            label=f"Optimal cut-off point for the business metric ({self.tuned_models[i].best_threshold_:.2f})",
        )
        axs[2].legend()
        axs[2].set_xlabel("Decision threshold (probability)")
        axs[2].set_ylabel("Objective score (using balanced accuracy)")
        axs[2].set_title("Objective score as a function of the decision threshold")
        axs[2].grid(True)
        
        fig.suptitle(f"Evaluation of the {model_name} - Future step {i+1}")
        
