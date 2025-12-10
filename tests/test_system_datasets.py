import unittest
import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.datasets import make_moons
import sys
import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt

# -------------------------------
# ROOT DIRECTORY CONFIGURATION
# -------------------------------

# tests/ -> project_root/
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/"))
PLOTS_DIR = os.path.join(ROOT_DIR, "plots")
RESULT_DIR= os.path.join(ROOT_DIR, "logs")
RESULTS_FILE = os.path.join(RESULT_DIR, "system_test_results.csv")

os.makedirs(PLOTS_DIR, exist_ok=True)

# Adjust path to allow imports from src
sys.path.append(ROOT_DIR)

from src.vqc_fnn.models.hybrid_model import HybridVQCFNN
from src.utils.input_encoder import InputPreprocessing


class TestSystemDatasets(unittest.TestCase):

    # -------------------------------
    # CSV LOGGING
    # -------------------------------
    @staticmethod
    def log_results(dataset_name, init_loss, final_loss, acc):
        file_exists = os.path.isfile(RESULTS_FILE)

        with open(RESULTS_FILE, mode="a", newline="") as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "dataset",
                    "initial_loss",
                    "final_loss",
                    "accuracy"
                ])

            writer.writerow([
                datetime.now().isoformat(),
                dataset_name,
                f"{init_loss:.6f}",
                f"{final_loss:.6f}",
                f"{acc:.6f}"
            ])

    # -------------------------------
    # PLOT GENERATION
    # -------------------------------
    @staticmethod
    def generate_plots():
        datasets_list = []
        initial_losses = []
        final_losses = []
        accuracies = []

        with open(RESULTS_FILE, mode="r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                datasets_list.append(row["dataset"])
                initial_losses.append(float(row["initial_loss"]))
                final_losses.append(float(row["final_loss"]))
                accuracies.append(float(row["accuracy"]))

        # ----- Loss Comparison -----
        plt.figure()
        plt.bar(datasets_list, initial_losses)
        plt.bar(datasets_list, final_losses)
        plt.title("Initial vs Final Loss per Dataset")
        plt.xlabel("Dataset")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(PLOTS_DIR, "loss_comparison.png"))
        plt.close()

        # ----- Accuracy Comparison -----
        plt.figure()
        plt.bar(datasets_list, accuracies)
        plt.title("Accuracy per Dataset")
        plt.xlabel("Dataset")
        plt.ylabel("Accuracy")
        plt.savefig(os.path.join(PLOTS_DIR, "accuracy_comparison.png"))
        plt.close()

        # ----- Accuracy Trend -----
        plt.figure()
        plt.plot(accuracies)
        plt.title("Accuracy Trend Over Test Runs")
        plt.xlabel("Test Run Index")
        plt.ylabel("Accuracy")
        plt.savefig(os.path.join(PLOTS_DIR, "accuracy_trend.png"))
        plt.close()

    # -------------------------------
    # MODEL TRAINING
    # -------------------------------
    def train_model(self, X, y, input_dim, output_dim, epochs=2):
        preprocessor = InputPreprocessing(X, y, test_size=0.3, random_state=42)
        X_train, X_test, y_train, y_test = preprocessor.preprocess()

        model = HybridVQCFNN(
            num_qubits=4,
            n_layers=2,
            input_dim=input_dim,
            output_dim=output_dim
        )

        model.classical_layer.fit_pca(X_train)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        initial_loss = None
        final_loss = None

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            if epoch == 0:
                initial_loss = loss.item()
            final_loss = loss.item()

        model.eval()
        with torch.no_grad():
            preds = torch.argmax(model(X_test), dim=1)
            acc = (preds == y_test).float().mean().item()

        return initial_loss, final_loss, acc

    # -------------------------------
    # TEST CASES
    # -------------------------------
    def test_iris_training(self):
        iris = datasets.load_iris()
        X, y = iris.data, iris.target

        init_loss, final_loss, acc = self.train_model(X, y, 4, 3)
        self.log_results("Iris", init_loss, final_loss, acc)

        self.assertLess(final_loss, init_loss)
        self.assertGreater(acc, 0.2)

    def test_breast_cancer_training(self):
        cancer = datasets.load_breast_cancer()
        X, y = cancer.data, cancer.target

        init_loss, final_loss, acc = self.train_model(X, y, 30, 2)
        self.log_results("Breast Cancer", init_loss, final_loss, acc)

        self.assertLess(final_loss, init_loss)
        self.assertGreater(acc, 0.1)

    def test_make_moons_training(self):
        X, y = make_moons(n_samples=100, noise=0.1, random_state=42)

        init_loss, final_loss, acc = self.train_model(X, y, 2, 2)
        self.log_results("Make Moons", init_loss, final_loss, acc)

        self.assertLess(final_loss, init_loss)
        self.assertGreater(acc, 0.31)

    # -------------------------------
    # GLOBAL TEARDOWN
    # -------------------------------
    @classmethod
    def tearDownClass(cls):
        if os.path.exists(RESULTS_FILE):
            cls.generate_plots()


if __name__ == '__main__':
    unittest.main()
