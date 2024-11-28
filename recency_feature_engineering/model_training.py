from sklearn.linear_model import LogisticRegression
from sklearn.metrics import root_mean_squared_error, accuracy_score, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler

class RecencyModelTraining:
    def __init__(self):
        pass
    
    def train_with_iterations(self, X_train, y_train, X_test, y_test, iterations):
        metrics = {
            'train_error': [],
            'test_error': [],
            'train_accuracy': [],
            'test_accuracy': [],
            'train_rmse': [],
            'test_rmse': [],
            'train_precision': [],
            'test_precision': [],
            'train_recall': [],
            'test_recall': [],
            'train_f1': [],
            'test_f1': []
        }

        for max_iter in iterations:
            self.model = LogisticRegression(
                multi_class='multinomial',
                solver='lbfgs',
                max_iter=max_iter,
                # class_weight='balanced',
                # penalty='elasticnet',
                # l1_ratio=0.0,
                random_state=31
            )
            self.model.fit(X_train, y_train)

            # Predictions
            train_predictions = self.model.predict(X_train)
            test_predictions = self.model.predict(X_test)

            # Accuracy
            train_accuracy = accuracy_score(y_train, train_predictions)
            test_accuracy = accuracy_score(y_test, test_predictions)

            # Errors
            metrics['train_error'].append(1 - train_accuracy)
            metrics['test_error'].append(1 - test_accuracy)
            metrics['train_accuracy'].append(train_accuracy)
            metrics['test_accuracy'].append(test_accuracy)

            # RMSE
            metrics['train_rmse'].append(np.sqrt(np.mean((y_train - train_predictions) ** 2)))
            metrics['test_rmse'].append(np.sqrt(np.mean((y_test - test_predictions) ** 2)))

            # Precision, Recall, F1
            train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
                y_train, train_predictions, average='weighted'
            )
            test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
                y_test, test_predictions, average='weighted'
            )

            metrics['train_precision'].append(train_precision)
            metrics['test_precision'].append(test_precision)
            metrics['train_recall'].append(train_recall)
            metrics['test_recall'].append(test_recall)
            metrics['train_f1'].append(train_f1)
            metrics['test_f1'].append(test_f1)

        return metrics

    def train_and_evaluate_knn(self, X_train, y_train, X_test, y_test, neighbors_range):
        metrics = {
            'train_accuracy': [],
            'test_accuracy': [],
            'train_rmse': [],
            'test_rmse': [],
            'train_precision': [],
            'test_precision': [],
            'train_recall': [],
            'test_recall': [],
            'train_f1': [],
            'test_f1': []
        }

        for n_neighbors in neighbors_range:
            # Train KNN model
            knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            knn.fit(X_train, y_train)

            # Predictions
            train_predictions = knn.predict(X_train)
            test_predictions = knn.predict(X_test)

            # Accuracy
            train_accuracy = accuracy_score(y_train, train_predictions)
            test_accuracy = accuracy_score(y_test, test_predictions)
            metrics['train_accuracy'].append(train_accuracy)
            metrics['test_accuracy'].append(test_accuracy)

            metrics['train_rmse'].append(np.sqrt(np.mean((y_train - train_predictions) ** 2)))
            metrics['test_rmse'].append(np.sqrt(np.mean((y_test - test_predictions) ** 2)))

            # Precision, Recall, F1
            train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
                y_train, train_predictions, average='weighted'
            )
            test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
                y_test, test_predictions, average='weighted'
            )

            metrics['train_precision'].append(train_precision)
            metrics['test_precision'].append(test_precision)
            metrics['train_recall'].append(train_recall)
            metrics['test_recall'].append(test_recall)
            metrics['train_f1'].append(train_f1)
            metrics['test_f1'].append(test_f1)

        return metrics
    
    def plot_knn_metrics(self, metrics, neighbors_range):
        metric_keys = list(metrics.keys())
        n_metrics = len(metric_keys)

        plt.figure(figsize=(15, 10))
        for i, key in enumerate(metric_keys, 1):

            plt.subplot((n_metrics + 1) // 2, 2, i)
            plt.plot(neighbors_range, metrics[key], label=key, marker='o')

            # Add formatted labels for each point
            for x, y in zip(neighbors_range, metrics[key]):
                plt.text(
                    x, y, f'{y:.2f}', fontsize=8, ha='center', va='bottom'
                )

            plt.xlabel('Number of Neighbors')
            plt.ylabel(f'{key}')
            plt.title(f'{key} over Number of Neighbors')
            plt.grid()

        plt.tight_layout()
        plt.show()

    def plot_logistic_metrics(self, metrics, iterations):
        metric_keys = list(metrics.keys())
        n_metrics = len(metric_keys)

        plt.figure(figsize=(15, 10))
        for i, key in enumerate(metric_keys, 1):
            plt.subplot((n_metrics + 1) // 2, 2, i)  
            plt.plot(iterations, metrics[key], label=key, marker='o')

            for x, y in zip(iterations, metrics[key]):
                plt.text(
                    x, y, f'{y:.3f}', fontsize=7, ha='center', va='bottom'
                )

            plt.xlabel('Iterations')
            plt.ylabel(f'{key}')
            plt.title(f'{key} over Iterations')
            plt.grid()

        plt.tight_layout()
        plt.show()



    def save_model(self, filename):
        joblib.dump(self.model, filename)

    def load_model(self, filename):
        self.model = joblib.load(filename)
