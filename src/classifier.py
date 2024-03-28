import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import  GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import time

class TextClassifier:
    def __init__(self, model_type='logistic', random_state=42):
        """
        Initialize the TextClassifier.
        
        Args:
            model_type (str): Type of the classifier model. Options: 'logistic', 'random_forest', 'svm'.
            random_state (int): Random seed for reproducibility.
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.best_params = None

    def train(self, X, y, cv=5):
        """
        Train the classifier model using grid search and cross-validation.
        
        Args:
            X (array-like): Input features.
            y (array-like): Target labels.
            cv (int): Number of cross-validation splits.
        """
        # Define the parameter grid based on the model type
        if self.model_type == 'logistic':
            param_grid = {'C': [0.1, 1, 5, 10], 'penalty': ['l2']}
            model = LogisticRegression(random_state=self.random_state)
        elif self.model_type == 'random_forest':
            param_grid = {'n_estimators': [20, 25, 50, 100], 'max_depth': [5, 10, 20], 'min_samples_split': [2, 5, 10]}
            model = RandomForestClassifier(random_state=self.random_state)
        elif self.model_type == 'svm':
            param_grid = {'C': [0.1, 1, 5, 10]}
            model = LinearSVC(random_state=self.random_state)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Perform grid search with cross-validation
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(X, y)

        # Store the best model and parameters
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_

        print(f"Best parameters: {self.best_params}")
        print(f"Best F1-score: {grid_search.best_score_:.4f}")

    def predict(self, X):
        """
        Make predictions using the trained classifier model.
        
        Args:
            X (array-like): Input features.
        
        Returns:
            array: Predicted labels.
        """
        return self.model.predict(X)

    def evaluate(self, X, y):
        """
        Evaluate the classifier model on the given data.
        
        Args:
            X (array-like): Input features.
            y (array-like): True labels.
        """
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted')

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-score: {f1:.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y, y_pred))
        print("\nClassification Report:")
        print(classification_report(y, y_pred))

    def save_model(self, model_path):
        """
        Save the trained classifier model to a file.
        
        Args:
            model_path (str): Path to save the model.
        """
        with open(model_path, 'wb') as file:
            pickle.dump(self.model, file)

    def load_model(self, model_path):
        """
        Load a trained classifier model from a file.
        
        Args:
            model_path (str): Path to load the model from.
        """
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)

    def get_model_size(self):
        """
        Get the size of the trained classifier model in bytes.
        
        Returns:
            int: Size of the model in bytes.
        """
        model_path = 'temp_model.pkl'
        self.save_model(model_path)
        model_size = os.path.getsize(model_path)
        os.remove(model_path)
        return model_size

    def measure_inference_time(self, X, n_iterations=100):
        """
        Measure the average inference time of the trained classifier model.
        
        Args:
            X (array-like): Input features.
            n_iterations (int): Number of iterations to average the inference time.
        
        Returns:
            float: Average inference time in seconds.
        """
        total_time = 0
        for _ in range(n_iterations):
            start_time = time.time()
            _ = self.predict(X)
            end_time = time.time()
            total_time += end_time - start_time
        avg_time = total_time / n_iterations
        return avg_time
    
    def plot_learning_curve(self, X, y, train_sizes=None):
        if train_sizes is None:
            train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        train_sizes, train_scores, test_scores = learning_curve(
            self.model, X, y, train_sizes=train_sizes, cv=5, scoring='f1_weighted', n_jobs=-1
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(8, 6))
        plt.xlabel("Training examples")
        plt.ylabel("F1-score")
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        plt.legend(loc="best")
        plt.show()