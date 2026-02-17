import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize


class MetricsCalculator:
    def __init__(self, labels, text_widget=None):
        self.labels = labels
        self.text = text_widget  # Tkinter Text widget handle

        self.precision = []
        self.recall = []
        self.fscore = []
        self.accuracy = []

        self.metrics_df = pd.DataFrame(columns=['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
        self.class_report_df = pd.DataFrame()
        self.class_performance_dfs = {}

        if not os.path.exists('results'):
            os.makedirs('results')

    def calculate_metrics(self, algorithm, predict, y_test, y_score=None):
        categories = self.labels

        # Overall metrics
        a = accuracy_score(y_test, predict) * 100
        p = precision_score(y_test, predict, average='macro') * 100
        r = recall_score(y_test, predict, average='macro') * 100
        f = f1_score(y_test, predict, average='macro') * 100

        self.accuracy.append(a)
        self.precision.append(p)
        self.recall.append(r)
        self.fscore.append(f)

        metrics_entry = pd.DataFrame({
            'Algorithm': [algorithm],
            'Accuracy': [a],
            'Precision': [p],
            'Recall': [r],
            'F1-Score': [f]
        })
        self.metrics_df = pd.concat([self.metrics_df, metrics_entry], ignore_index=True)

        # Print to Text widget if available
        if self.text:
            self.text.insert('end', f"{algorithm} Accuracy  : {a:.4f}\n")
            self.text.insert('end', f"{algorithm} Precision : {p:.4f}\n")
            self.text.insert('end', f"{algorithm} Recall    : {r:.4f}\n")
            self.text.insert('end', f"{algorithm} FScore    : {f:.4f}\n\n")

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, predict)
        CR = classification_report(y_test, predict, target_names=[str(c) for c in categories], output_dict=True)

        if self.text:
            self.text.insert('end', f"{algorithm} Classification Report\n")
            self.text.insert('end', classification_report(y_test, predict, target_names=[str(c) for c in categories]) + "\n")

        cr_df = pd.DataFrame(CR).transpose()
        cr_df['Algorithm'] = algorithm
        self.class_report_df = pd.concat([self.class_report_df, cr_df], ignore_index=False)

        # Per-class performance
        for category in categories:
            class_entry = pd.DataFrame({
                'Algorithm': [algorithm],
                'Precision': [CR[str(category)]['precision'] * 100],
                'Recall': [CR[str(category)]['recall'] * 100],
                'F1-Score': [CR[str(category)]['f1-score'] * 100],
                'Support': [CR[str(category)]['support']]
            })

            if str(category) not in self.class_performance_dfs:
                self.class_performance_dfs[str(category)] = pd.DataFrame(columns=['Algorithm', 'Precision', 'Recall', 'F1-Score', 'Support'])

            self.class_performance_dfs[str(category)] = pd.concat([self.class_performance_dfs[str(category)], class_entry], ignore_index=True)

        # Confusion Matrix Plot
        plt.figure(figsize=(8, 8))
        ax = sns.heatmap(conf_matrix, xticklabels=categories, yticklabels=categories, annot=True, cmap="viridis", fmt="g")
        ax.set_ylim([0, len(categories)])
        plt.title(f"{algorithm} Confusion Matrix")
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.savefig(f"results/{algorithm.replace(' ', '_')}_confusion_matrix.png")
        plt.show()

       

    def plot_classification_graphs(self):
        melted_df = pd.melt(self.metrics_df, id_vars=['Algorithm'],
                            value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                            var_name='Parameters', value_name='Value')

        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='Parameters', y='Value', hue='Algorithm', data=melted_df)
        plt.title('Classifier Performance Comparison', fontsize=14, pad=10)
        plt.ylabel('Score (%)', fontsize=12)
        plt.xlabel('Metrics', fontsize=12)
        plt.xticks(rotation=0)
        plt.legend(title='Algorithms', bbox_to_anchor=(1.05, 1), loc='upper left')

        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', padding=3)

        os.makedirs('results', exist_ok=True)
        plt.tight_layout()
        plt.savefig('results/classifier_performance.png')
        plt.show()

        # Class-specific bar plots
        for class_name, class_df in self.class_performance_dfs.items():
            melted_class_df = pd.melt(class_df, id_vars=['Algorithm'],
                                      value_vars=['Precision', 'Recall', 'F1-Score'],
                                      var_name='Parameters', value_name='Value')

            plt.figure(figsize=(12, 6))
            ax = sns.barplot(x='Parameters', y='Value', hue='Algorithm', data=melted_class_df)
            plt.title(f'Class {class_name} Performance Comparison', fontsize=14, pad=10)
            plt.ylabel('Score (%)', fontsize=12)
            plt.xlabel('Metrics', fontsize=12)
            plt.xticks(rotation=0)
            plt.legend(title='Algorithms', bbox_to_anchor=(1.05, 1), loc='upper left')

            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f', padding=3)

            plt.tight_layout()
            plt.savefig(f'results/class_{class_name}_performance.png')
            plt.show()

        # Return formatted metrics table
        melted_df_new = self.metrics_df[['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].copy()
        melted_df_new = melted_df_new.round(3)
        return melted_df_new
