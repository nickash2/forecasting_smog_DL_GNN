# from src.run_forecast import main
# from graph_modelling.utils.rescale import rescale
# %%
import torch
import pandas as pd
from torch_geometric.data import Data
from pathlib import Path
import os
import pickle  # Import the pickle module

if __name__ == "__main__":
    # sensors = [
    #     ["NL01485", "NL01494"],  # Rotterdam
    #     ["NL10636", "NL10641"],  # Utrecht
    #     ["NL49003", "NL49012"],  # Amsterdam
    # ]

    # years = [2017, 2018, 2020, 2021, 2022, 2023]
    # cities = ["rotterdam", "utrecht", "amsterdam"]

    # rescale(sensors, years, cities)

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
                ALL_DIR / city, x, y, drop_datetime=drop_datetime, city_name=idx
            )
            for element in x:
                f = pd.concat([f, element], axis=0)
            for element in y:
                l = pd.concat([l, element], axis=0)

        return f, l

    # %%
    X_train, y_train = load_gnn_data("train", drop_datetime=False)
    X_val, y_val = load_gnn_data("val", drop_datetime=False)
    X_test, y_test = load_gnn_data("test", drop_datetime=False)
    X = pd.concat([X_train, X_val, X_test], axis=0)
    y = pd.concat([y_train, y_val, y_test], axis=0)

    print(y)
    print(X.shape, y.shape)

    # Ensure data is sorted by time before reshaping
    X_sorted = X.sort_values(by=["DateTime", "city_name"])
    num_timesteps = len(X_sorted) // 3  # Since we have 3 nodes per timestep
    num_features = X_sorted.shape[1] - 2  # Exclude DateTime and city_name
    x = X_sorted.iloc[:, 2:].values.reshape(num_timesteps, 3, num_features)

    # %%

    x = torch.tensor(x, dtype=torch.float)

    y_sorted = y.sort_values(by=["DateTime", "city_name"])
    y = y_sorted.iloc[:, 2:].values.reshape(
        num_timesteps, 3, -1
    )  # (num_timesteps, 3, target_features)
    # %%
    y = torch.tensor(y, dtype=torch.float)

    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], dtype=torch.long
    )

    def create_sliding_windows(X, Y, window_size, forecast_horizon):
        """Create sliding windows of data."""
        X_windows, Y_windows = [], []
        for i in range(len(X) - window_size - forecast_horizon + 1):
            X_windows.append(X[i : i + window_size])
            Y_windows.append(Y[i + window_size : i + window_size + forecast_horizon])

        return torch.stack(X_windows), torch.stack(Y_windows)

    # %%
    # Parameters:
    N_HOURS_U = 72  # Number of past hours to use
    N_HOURS_Y = 24  # Number of future hours to predict

    # Generate sliding window data
    X_windows, Y_windows = create_sliding_windows(x, y, N_HOURS_U, N_HOURS_Y)

    num_samples, window_size, num_nodes, num_features = X_windows.shape
    _, forecast_horizon, _, target_features = Y_windows.shape

    # Flatten the input window per node:
    X_windows_flat = X_windows.reshape(
        num_samples, num_nodes, window_size * num_features
    )
    # Flatten the forecast horizon window into one target vector per node:
    Y_windows_flat = Y_windows.reshape(
        num_samples, num_nodes, forecast_horizon * target_features
    )

    dataset = []
    for i in range(num_samples):
        # Permute to have shape: (num_nodes, window_size, num_features)
        x_seq = X_windows[i].permute(1, 0, 2).contiguous()
        data = Data(
            x_seq=x_seq,  # storing the time series separately
            edge_index=edge_index,
            y=Y_windows[i],
        )
        dataset.append(data)

    print(X_sorted.head())
    print(num_features)
    # %%
    # Split the dataset
    dataset_size = len(dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size : train_size + val_size]
    test_dataset = dataset[train_size + val_size :]

    # %%
    # Compute min and max using only the training set.

    def get_min_max(dataset, attr_name):
        """Calculates min and max values for a given attribute in the dataset."""
        all_data = torch.cat([getattr(data, attr_name) for data in dataset], dim=0)
        arr = all_data.numpy()
        arr_min = arr.min(axis=0, keepdims=True)
        arr_max = arr.max(axis=0, keepdims=True)
        return arr_min, arr_max

    x_min, x_max = get_min_max(train_dataset, "x_seq")
    y_min, y_max = get_min_max(train_dataset, "y")

    # Save the y_min and y_max
    with open(SAVE_DIR / "y_min_max.pkl", "wb") as f:
        pickle.dump((y_min, y_max), f)

    def normalize_dataset(dataset, x_min, x_max, y_min, y_max):
        """Normalizes x and y attributes of a dataset."""
        for data in dataset:
            x_arr = data.x_seq.numpy()
            x_norm = minmax_normalize_arr(x_arr, x_min, x_max)
            data.x = torch.tensor(x_norm, dtype=torch.float)
            y_arr = data.y.numpy()
            y_norm = minmax_normalize_arr(y_arr, y_min, y_max)
            data.y = torch.tensor(y_norm, dtype=torch.float)
        return dataset

    train_dataset = normalize_dataset(train_dataset, x_min, x_max, y_min, y_max)
    val_dataset = normalize_dataset(val_dataset, x_min, x_max, y_min, y_max)
    test_dataset = normalize_dataset(test_dataset, x_min, x_max, y_min, y_max)

    # Save the preprocessed datasets

    with open(SAVE_DIR / "train_dataset.pkl", "wb") as f:
        pickle.dump(train_dataset, f)

    with open(SAVE_DIR / "val_dataset.pkl", "wb") as f:
        pickle.dump(val_dataset, f)

    with open(SAVE_DIR / "test_dataset.pkl", "wb") as f:
        pickle.dump(test_dataset, f)

    print("Preprocessed datasets saved to:", SAVE_DIR)
