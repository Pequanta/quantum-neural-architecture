# test_model_comparison.py
import unittest
import os
import csv
from datetime import datetime
import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np

from src.vqc_fnn.models.hybrid_model import HybridVQCFNN
from src.vqc_fnn.models.benchmark_networks import ParamMatchedMLP, RFFSmallMLP
from src.utils.input_encoder import InputPreprocessing
import logging



RESULTS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "comparison_log.csv"))
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)


class TestModelComparison(unittest.TestCase):
    n_q = 6  # number of qubits / target input dimension for classical baselines

    @staticmethod
    def log_results(model_name, dataset_name, init_loss, final_loss, acc, n_params):
        file_exists = os.path.isfile(RESULTS_FILE)
        with open(RESULTS_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "model", "dataset", "init_loss", "final_loss", "accuracy", "params"])
            writer.writerow([datetime.now().isoformat(), model_name, dataset_name,
                             f"{init_loss:.6f}", f"{final_loss:.6f}", f"{acc:.6f}", n_params])

    @staticmethod
    def train_model(model, X_train, y_train, X_test, y_test, epochs=10, lr=0.01):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        initial_loss = None
        final_loss = None

        # Convert data to torch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)

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

        # Evaluation
        model.eval()
        with torch.no_grad():
            preds = torch.argmax(model(X_test), dim=1)
            acc = (preds == y_test).float().mean().item()

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return initial_loss, final_loss, acc, n_params

    @staticmethod
    def load_dataset(name):
        if name == "iris":
            data = datasets.load_iris()
            X, y = data.data, data.target
        elif name == "breast_cancer":
            data = datasets.load_breast_cancer()
            X, y = data.data, data.target
        else:
            raise ValueError("Unsupported dataset")

        X = StandardScaler().fit_transform(X)
        return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    @classmethod
    def adjust_features(cls, X_train, X_test):
        n_samples_train, n_features = X_train.shape
        target_dim = cls.n_q

        if n_features > target_dim:
            # reduce to n_q dimensions via PCA
            pca = PCA(n_components=target_dim)
            X_train_reduced = pca.fit_transform(X_train)
            X_test_reduced = pca.transform(X_test)
        elif n_features < target_dim:
            # pad with zeros
            pad_width = target_dim - n_features
            X_train_reduced = np.hstack([X_train, np.zeros((X_train.shape[0], pad_width))])
            X_test_reduced = np.hstack([X_test, np.zeros((X_test.shape[0], pad_width))])
        else:
            X_train_reduced = X_train
            X_test_reduced = X_test

        return X_train_reduced, X_test_reduced

    def test_comparison(self):
        datasets_to_run = ["iris", "breast_cancer"]
        torch.manual_seed(42)
        np.random.seed(42)

        for ds_name in datasets_to_run:
            X_train, X_test, y_train, y_test = self.load_dataset(ds_name)
            n_features = X_train.shape[1]
            n_classes = len(np.unique(y_train))

            print(f"\n--- Dataset: {ds_name} ---")

            # ----------------- VQC -----------------
            vqc_model = HybridVQCFNN(num_qubits=self.n_q, n_layers=3, input_dim=n_features, output_dim=n_classes)

            vqc_params = sum(p.numel() for p in vqc_model.q_layer.parameters() if p.requires_grad)
            logging.info("VQC params:", vqc_params)
            logging.info("dimensions: ", X_train.shape,",",y_train.shape)

            hidden_dim = int((vqc_params - X_train.shape[1] - 3) / (X_train.shape[1] + 3 + 1))

            init_loss, final_loss, acc, n_params = self.train_model(vqc_model, X_train, y_train, X_test, y_test)
            self.log_results("VQC", ds_name, init_loss, final_loss, acc, n_params)

            
            X_train_reduced, X_test_reduced = self.adjust_features(X_train, X_test)

            # ----------------- Param-Matched MLP -----------------
            pm_mlp = ParamMatchedMLP(input_dim=6, hidden_dim=10,output_dim=y_test.shape[0])
            init_loss, final_loss, acc, n_params = self.train_model(pm_mlp, X_train_reduced, y_train, X_test_reduced, y_test)
            self.log_results("ParamMatchedMLP", ds_name, init_loss, final_loss, acc, n_params)

            rff_mlp = RFFSmallMLP(n_q=self.n_q, m=6, hidden=15, train_w=False)
            init_loss, final_loss, acc, n_params = self.train_model(rff_mlp, X_train_reduced, y_train, X_test_reduced, y_test)
            self.log_results("RFFSmallMLP", ds_name, init_loss, final_loss, acc, n_params)

            # Sanity check
            self.assertLess(final_loss, init_loss, f"{ds_name} final_loss not less than init_loss")


if __name__ == "__main__":
    unittest.main()
