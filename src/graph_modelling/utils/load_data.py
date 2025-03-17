import torch
import os
import pandas as pd
from pathlib import Path
import numpy as np


def load_train_val_data(cities, is_train=True, ALL_DIR=Path(".")):
    """
    Load training or validation data for each city.

    Args:
        cities (list): List of cities (folder names).
        is_train (bool): Flag to specify whether loading train data (True) or val data (False).

    Returns:
        torch.Tensor: Node feature tensor (x).
        torch.Tensor: Node label tensor (y).
    """
    # Initialize lists for features and labels
    node_features = []
    node_labels = []

    for city in cities:
        city_path = ALL_DIR / city

        # Select the appropriate file pattern (train or validation)
        data_type = "train" if is_train else "val"
        feature_pattern = f"{data_type}_u.csv"
        label_pattern = f"{data_type}_y.csv"

        # Get all feature and label files for the city
        feature_files = sorted(
            [
                f
                for f in os.listdir(city_path)
                if f.startswith(data_type) and f.endswith("_u.csv")
            ]
        )
        label_files = sorted(
            [
                f
                for f in os.listdir(city_path)
                if f.startswith(data_type) and f.endswith("_y.csv")
            ]
        )

        # Read and aggregate features across all years (or months, days, etc.)
        city_features = []
        city_labels = []

        for feat_file, label_file in zip(feature_files, label_files):
            feat_df = pd.read_csv(
                os.path.join(city_path, feat_file), delimiter=";"
            ).drop(columns=["DateTime"])
            label_df = pd.read_csv(
                os.path.join(city_path, label_file), delimiter=";"
            ).drop(columns=["DateTime"])

            city_features.append(
                feat_df.mean(axis=0)
            )  # Aggregate features (e.g., over time)
            city_labels.append(
                label_df.mean(axis=0)
            )  # Aggregate labels (e.g., over time)

        # Compute overall mean feature vector for the city
        node_features.append(pd.concat(city_features, axis=1).mean(axis=1).values)
        node_labels.append(pd.concat(city_labels, axis=1).mean(axis=1).values)

    # Convert lists to PyTorch tensors
    x = torch.tensor(np.array(node_features), dtype=torch.float)
    y = torch.tensor(np.array(node_labels), dtype=torch.float)

    print(f"Node feature shape ({'train' if is_train else 'val'}):", x.shape)
    print(f"Node label shape ({'train' if is_train else 'val'}):", y.shape)

    return x, y


def load_test_data(cities, ALL_DIR=Path(".")):
    test_features = []
    y_true = []

    for city in cities:
        city_path = ALL_DIR / city

        test_feature_files = sorted(
            [
                f
                for f in os.listdir(city_path)
                if f.startswith("test_") and f.endswith("_u.csv")
            ]
        )
        test_label_files = sorted(
            [
                f
                for f in os.listdir(city_path)
                if f.startswith("test_") and f.endswith("_y.csv")
            ]
        )

        city_test_features = []
        city_test_labels = []

        for feat_file, label_file in zip(test_feature_files, test_label_files):
            feat_df = pd.read_csv(
                os.path.join(city_path, feat_file), delimiter=";"
            ).drop(columns=["DateTime"])
            label_df = pd.read_csv(
                os.path.join(city_path, label_file), delimiter=";"
            ).drop(columns=["DateTime"])

            city_test_features.append(feat_df.mean(axis=0))  # Aggregate over time
            city_test_labels.append(label_df.mean(axis=0))  # Aggregate over time

        test_features.append(pd.concat(city_test_features, axis=1).mean(axis=1).values)
        y_true.append(pd.concat(city_test_labels, axis=1).mean(axis=1).values)

    # Convert to PyTorch tensors
    x_test = torch.tensor(np.array(test_features), dtype=torch.float)
    y_true = torch.tensor(np.array(y_true), dtype=torch.float)
    return x_test, y_true
