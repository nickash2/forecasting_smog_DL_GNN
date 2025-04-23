import torch
import pandas as pd
import numpy as np  # Add numpy import
from torch_geometric.data import Data
from pathlib import Path
import os
import pickle  # Import the pickle module
from GNNTimeSeriesDataset import GNNTimeSeriesDataset


def feature_wise_normalize(arr, arr_min, arr_max):
    """Normalize each feature independently."""
    return (arr - arr_min) / (arr_max - arr_min + 1e-8)


def get_feature_min_max(dataset, attr_name):
    """Calculates min and max values for each feature independently."""
    all_data = torch.cat([getattr(data, attr_name) for data in dataset], dim=0)
    arr = all_data.numpy()

    # For feature-wise normalization, we need min/max per feature
    # Keep dimensions for broadcasting during normalization
    arr_min = arr.min(axis=0, keepdims=True)  # Min per feature
    arr_max = arr.max(axis=0, keepdims=True)  # Max per feature

    return arr_min, arr_max


if __name__ == "__main__":
    # Define project structure variables
    HABROK = bool(0)
    BASE_DIR = Path.cwd().parent
    MODEL_PATH = BASE_DIR / "results" / "models"
    DATA_DIR = BASE_DIR / "data" / "data_combined"
    ALL_DIR = DATA_DIR / "all"

    print("BASE_DIR: ", BASE_DIR)
    print("MODEL_PATH: ", MODEL_PATH)
    print("ALL_DIR: ", ALL_DIR)

    torch.manual_seed(34)

    N_HOURS_U = 72  # number of hours to use for input
    N_HOURS_Y = 24  # number of hours to predict
    N_HOURS_STEP = 24  # "sampling rate" in hours
    CONTAMINANTS = ["NO2", "O3"]  # 'PM10', 'PM25']
    cities = [
        "amsterdam",
        "rotterdam",
        "utrecht",
    ]  # Defined here so its available in the script
    SAVE_DIR = ALL_DIR / "geometric_pkl"  # Added saving directory

    # Ensure directory exists
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    def get_data_files(city_path, data_type):
        """Get all feature and label files for a city for a specific data type."""
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

    def add_time_encoding(df):
        """Add time encoding features (hour of day, day of week) using sine and cosine transformation."""
        # Convert DateTime to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df["DateTime"]):
            df["DateTime"] = pd.to_datetime(df["DateTime"])

        # Hour of day encoding (24-hour cycle)
        hours = df["DateTime"].dt.hour
        df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)

        return df

    def read_csv_files(
        city_path, feature_files, label_files, drop_datetime=True, city_name=None
    ):
        """Read feature and label CSV files."""
        feature_dfs = []
        label_dfs = []

        for feat_file, label_file in zip(feature_files, label_files):
            feat_df = pd.read_csv(os.path.join(city_path, feat_file), delimiter=";")
            label_df = pd.read_csv(os.path.join(city_path, label_file), delimiter=";")
            if city_name is not None:
                feat_df.insert(
                    feat_df.columns.get_loc("DateTime") + 1, "city_name", city_name
                )
                label_df.insert(
                    label_df.columns.get_loc("DateTime") + 1, "city_name", city_name
                )
            if drop_datetime:
                feat_df = feat_df.drop(columns=["DateTime"])
                label_df = label_df.drop(columns=["DateTime"])

            feature_dfs.append(feat_df.drop(columns=["O3"]))
            label_dfs.append(label_df.drop(columns=["O3"]))

        return feature_dfs, label_dfs

    def minmax_normalize_arr(arr, arr_min, arr_max):
        # Normalize with provided min and max
        return (arr - arr_min) / (arr_max - arr_min + 1e-8)

    def load_gnn_data(split_type="train", drop_datetime=True):
        f = pd.DataFrame()
        l = pd.DataFrame()
        for idx, city in enumerate(cities):
            x, y = get_data_files(ALL_DIR / city, split_type)
            x, y = read_csv_files(
                ALL_DIR / city, x, y, drop_datetime=False, city_name=idx
            )
            for element in x:
                # Add time encoding features before dropping DateTime
                element = add_time_encoding(element)
                if drop_datetime:
                    element = element.drop(columns=["DateTime"])
                f = pd.concat([f, element], axis=0)
            for element in y:
                # Also add time encodings to target data
                element = add_time_encoding(element)
                if drop_datetime:
                    element = element.drop(columns=["DateTime"])
                l = pd.concat([l, element], axis=0)

        return f, l

    def process_dataset(X, y, is_train=False):
        """Process a dataset without doing further train/val/test splitting"""
        # Ensure data is sorted by time before reshaping
        X_sorted = X.sort_values(by=["DateTime", "city_name"])
        num_timesteps = len(X_sorted) // 3
        num_features = X_sorted.shape[1] - 2
        x = X_sorted.iloc[:, 2:].values.reshape(num_timesteps, 3, num_features)
        x = torch.tensor(x, dtype=torch.float)

        # Extract only the NO2 column for y data
        y_sorted = y.sort_values(by=["DateTime", "city_name"])
        no2_column_idx = y_sorted.columns.get_loc("NO2")
        y_no2 = y_sorted.iloc[:, [no2_column_idx]]
        y = y_no2.values.reshape(num_timesteps, 3, 1)
        y = torch.tensor(y, dtype=torch.float)

        # Distance matrix between cities in kilometers
        # Order: [Amsterdam, Rotterdam, Utrecht]
        distances = {
            (0, 1): 58.6,  # Amsterdam to Rotterdam
            (0, 2): 35.3,  # Amsterdam to Utrecht
            (1, 0): 58.6,  # Rotterdam to Amsterdam
            (1, 2): 51.5,  # Rotterdam to Utrecht
            (2, 0): 35.3,  # Utrecht to Amsterdam
            (2, 1): 51.5,  # Utrecht to Rotterdam
        }

        # Create edge_attr tensor with distances
        edge_attr = torch.tensor(
            [
                distances[(0, 1)],  # Amsterdam to Rotterdam
                distances[(0, 2)],  # Amsterdam to Utrecht
                distances[(1, 0)],  # Rotterdam to Amsterdam
                distances[(1, 2)],  # Rotterdam to Utrecht
                distances[(2, 0)],  # Utrecht to Amsterdam
                distances[(2, 1)],  # Utrecht to Rotterdam
            ],
            dtype=torch.float,
        ).reshape(-1, 1)  # Shape: [num_edges, 1]

        # Keep your existing edge_index
        edge_index = torch.tensor(
            [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], dtype=torch.long
        )

        # Create dataset without splitting
        dataset = GNNTimeSeriesDataset(
            x=x,
            y=y,
            window_size=N_HOURS_U,
            forecast_horizon=N_HOURS_Y,
            step=N_HOURS_STEP,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        return dataset

    # Don't concatenate pre-split data
    X_train, y_train = load_gnn_data("train", drop_datetime=False)
    X_val, y_val = load_gnn_data("val", drop_datetime=False)
    X_test, y_test = load_gnn_data("test", drop_datetime=False)

    # Process each without further splitting
    train_dataset = process_dataset(X_train, y_train)
    val_dataset = process_dataset(X_val, y_val)
    test_dataset = process_dataset(X_test, y_test)

    # Step 3: Normalize using only training data statistics
    train_dataset, (x_min, x_max), (y_min, y_max) = train_dataset.normalize()
    val_dataset, _, _ = val_dataset.normalize(x_min, x_max, y_min, y_max)
    test_dataset, _, _ = test_dataset.normalize(x_min, x_max, y_min, y_max)

    # After normalizing
    print(f"Training data normalized range: [{x_min.min():.4f}, {x_max.max():.4f}]")
    print(f"Validation data normalized using same stats")

    # Step 4: Save datasets
    GNNTimeSeriesDataset.save_datasets(
        train_dataset, val_dataset, test_dataset, SAVE_DIR, y_min, y_max
    )

    print("Preprocessed datasets saved to:", SAVE_DIR)
