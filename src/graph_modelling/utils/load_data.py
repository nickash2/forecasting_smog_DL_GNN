import torch
import os
import pandas as pd
from pathlib import Path
import numpy as np


def get_data_files(city_path, data_type):
    """
    Get all feature and label files for a city for a specific data type.

    Args:
        city_path (Path): Path to the city directory.
        data_type (str): Type of data (train, val, or test).

    Returns:
        tuple: Lists of feature and label files.
    """
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

    return feature_files, label_files


def read_csv_files(
    city_path, feature_files, label_files, drop_datetime=True, city_name=None
):
    """
    Read feature and label CSV files.

    Args:
        city_path (Path): Path to the city directory.
        feature_files (list): List of feature file names.
        label_files (list): List of label file names.
        drop_datetime (bool, optional): Whether to drop DateTime column. Defaults to True.

    Returns:
        tuple: Lists of feature and label DataFrames.
    """
    feature_dfs = []
    label_dfs = []

    for feat_file, label_file in zip(feature_files, label_files):
        feat_df = pd.read_csv(os.path.join(city_path, feat_file), delimiter=";")
        label_df = pd.read_csv(os.path.join(city_path, label_file), delimiter=";")
        if city_name is not None:
            feat_df["city_name"] = city_name
            label_df["city_name"] = city_name
        if drop_datetime:
            feat_df = feat_df.drop(columns=["DateTime"])
            label_df = label_df.drop(columns=["DateTime"])

        feature_dfs.append(feat_df)
        label_dfs.append(label_df)

    return feature_dfs, label_dfs


def aggregate_time_series(dataframes, method="mean"):
    """
    Aggregate time series data using the specified method.

    Args:
        dataframes (list): List of DataFrames to aggregate.
        method (str): Aggregation method ('mean', 'max', 'min', 'sum', None).
            If None, returns the original DataFrames.

    Returns:
        list: Aggregated DataFrames.
    """
    if method is None:
        return dataframes

    aggregated = []
    for df in dataframes:
        if method == "mean":
            aggregated.append(df.mean(axis=0))
        elif method == "max":
            aggregated.append(df.max(axis=0))
        elif method == "min":
            aggregated.append(df.min(axis=0))
        elif method == "sum":
            aggregated.append(df.sum(axis=0))
        else:
            raise ValueError(f"Unsupported aggregation method: {method}")

    return aggregated


def read_and_aggregate_files(
    city_path, feature_files, label_files, agg_method="mean", drop_datetime=True
):
    """
    Read and aggregate features and labels from files.

    Args:
        city_path (Path): Path to the city directory.
        feature_files (list): List of feature file names.
        label_files (list): List of label file names.
        agg_method (str, optional): Aggregation method ('mean', 'max', 'min', 'sum', None).
            If None, no aggregation is performed. Defaults to 'mean'.
        drop_datetime (bool, optional): Whether to drop DateTime column. Defaults to True.

    Returns:
        tuple: Lists of aggregated features and labels.
    """
    # Read CSV files
    feature_dfs, label_dfs = read_csv_files(
        city_path, feature_files, label_files, drop_datetime
    )

    # Aggregate the data if requested
    city_features = aggregate_time_series(feature_dfs, agg_method)
    city_labels = aggregate_time_series(label_dfs, agg_method)

    return city_features, city_labels


def aggregate_city_data(city_features, city_labels):
    """
    Compute overall mean feature and label vectors for a city.

    Args:
        city_features (list): List of feature DataFrames or Series.
        city_labels (list): List of label DataFrames or Series.

    Returns:
        tuple: Mean feature and label vectors.
    """
    # Convert to DataFrames if they are Series
    features_data = []
    for feat in city_features:
        if isinstance(feat, pd.Series):
            features_data.append(feat.values)
        else:
            features_data.append(feat.values.flatten())

    labels_data = []
    for label in city_labels:
        if isinstance(label, pd.Series):
            labels_data.append(label.values)
        else:
            labels_data.append(label.values.flatten())

    # Convert to numpy arrays and compute mean
    features_array = np.array(features_data)
    labels_array = np.array(labels_data)

    node_features = np.mean(features_array, axis=0)
    node_labels = np.mean(labels_array, axis=0)

    return node_features, node_labels


def convert_to_tensors(features, labels, data_type):
    """
    Convert lists to PyTorch tensors and print shapes.

    Args:
        features (list): List of feature vectors.
        labels (list): List of label vectors.
        data_type (str): Type of data (train, val, or test).

    Returns:
        tuple: Feature and label tensors.
    """
    x = torch.tensor(np.array(features), dtype=torch.float)
    y = torch.tensor(np.array(labels), dtype=torch.float)

    print(f"Node feature shape ({data_type}):", x.shape)
    print(f"Node label shape ({data_type}):", y.shape)

    return x, y


def load_train_val_data(
    cities, is_train=True, ALL_DIR=Path("."), agg_method="mean", drop_datetime=True
):
    """
    Load training or validation data for each city.

    Args:
        cities (list): List of cities (folder names).
        is_train (bool): Flag to specify whether loading train data (True) or val data (False).
        ALL_DIR (Path): Base directory containing city folders.
        agg_method (str, optional): Aggregation method ('mean', 'max', 'min', 'sum', None).
            If None, no aggregation is performed. Defaults to 'mean'.
        drop_datetime (bool, optional): Whether to drop DateTime column. Defaults to True.

    Returns:
        tuple: Node feature tensor (x) and node label tensor (y).
    """
    node_features = []
    node_labels = []
    data_type = "train" if is_train else "val"

    for city in cities:
        city_path = ALL_DIR / city

        # Get data files
        feature_files, label_files = get_data_files(city_path, data_type)
        print(feature_files)
        # Read raw data
        city_features, city_labels = read_and_aggregate_files(
            city_path, feature_files, label_files, agg_method, drop_datetime
        )

        # Apply aggregation if needed
        if agg_method is not None:
            city_features, city_labels = aggregate_city_data(city_features, city_labels)
        else:
            raise NotImplementedError(
                "Non-aggregated data processing not implemented yet"
            )

        node_features.append(city_features)
        node_labels.append(city_labels)
    # Convert to tensors
    node_features, node_labels = convert_to_tensors(
        node_features, node_labels, data_type
    )

    return node_features, node_labels


def load_test_data(cities, ALL_DIR=Path("."), agg_method="mean", drop_datetime=True):
    """
    Load test data for each city.

    Args:
        cities (list): List of cities (folder names).
        ALL_DIR (Path): Base directory containing city folders.
        agg_method (str, optional): Aggregation method ('mean', 'max', 'min', 'sum', None).
            If None, no aggregation is performed. Defaults to 'mean'.
        drop_datetime (bool, optional): Whether to drop DateTime column. Defaults to True.

    Returns:
        tuple: Test feature tensor (x_test) and ground truth tensor (y_true).
    """
    node_features = []
    node_labels = []
    data_type = "test"

    for city in cities:
        city_path = ALL_DIR / city

        # Get data files
        feature_files, label_files = get_data_files(city_path, data_type)

        # Read and aggregate data
        city_features, city_labels = read_and_aggregate_files(
            city_path, feature_files, label_files, agg_method, drop_datetime
        )

        # Compute city-level aggregates if aggregation was performed
        if agg_method is not None:
            city_node_features, city_node_labels = aggregate_city_data(
                city_features, city_labels
            )
        else:
            raise NotImplementedError(
                "Non-aggregated data processing not implemented yet"
            )

        node_features.append(city_node_features)
        node_labels.append(city_node_labels)

    # Convert to tensors
    return convert_to_tensors(node_features, node_labels, data_type)
