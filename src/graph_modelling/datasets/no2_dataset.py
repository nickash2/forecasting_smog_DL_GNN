import os
import numpy as np
import pandas as pd
import torch
import pickle
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from ..utils.index_dataset import IndexDataset
from torch_geometric_temporal.signal import StaticGraphTemporalSignalBatch


class NO2DatasetLoader(object):
    """A dataset for NO2 forecasting across three Dutch cities: Amsterdam, Rotterdam, and Utrecht.
    The underlying graph is static - vertices are cities and edges represent geographical connections.
    Edge weights are the distances between cities. Node features are lagged hourly NO2 measurements and weather variables.
    The target is the NO2 level for the upcoming time period.

    Args:
        index (bool, optional): If True, initializes the dataloader to use index-based batching.
            Defaults to False.
        data_dir (str, optional): Path to the data directory. Defaults to None.
        only_no2 (bool, optional): If True, only NO2 features are used. Defaults to False.
        cache_file (str, optional): Filename to save/load the processed dataset. If None, no caching is used.
            Defaults to "no2_dataset_cache.pkl".
        force_reload (bool, optional): If True, recompute the dataset even if cache exists. Defaults to False.
    """

    def __init__(
        self,
        index=False,
        data_dir=None,
        only_no2=False,
        cache_file="no2_dataset_cache.pkl",
        force_reload=False,
        smooth_data=False,
        smooth_window=5,
    ):
        self.cities = ["amsterdam", "utrecht", "rotterdam"]
        self.index = index
        self.data_dir = data_dir
        self.only_no2 = only_no2  # New parameter to control feature selection
        self.scalers = {}  # Dictionary to store MinMaxScalers for each feature
        self.cache_file = cache_file
        self.force_reload = force_reload
        self.smooth_data = smooth_data
        self.smooth_window = smooth_window

        # Set default data directory if not provided
        if self.data_dir is None:
            import pathlib

            base_dir = pathlib.Path(__file__).parent.parent.parent.parent
            self.data_dir = base_dir / "data" / "data_gnn"

        self._read_data()

        if index:
            self.IndexDataset = IndexDataset

    def _apply_smoothing(self, variables=None):
        """Apply moving average smoothing to specified variables."""
        if variables is None:
            # Only smooth NO2 by default
            variables = ["NO2"]

        print(f"Applying smoothing with window size {self.smooth_window}...")

        # Apply smoothing to each city's data separately
        for var in variables:
            for city in self.cities:
                col_name = f"{city}_{var}"
                if col_name in self._data.columns:
                    # Apply moving average smoothing
                    self._data[col_name] = (
                        self._data[col_name]
                        .rolling(window=self.smooth_window, center=True)
                        .mean()
                        .fillna(method="bfill")
                        .fillna(method="ffill")  # Handle edges
                    )
                    print(f"Applied smoothing to {col_name}")

    def _read_data(self):
        """Read the data from CSV files and combine them for lagged feature approach."""
        # Check if cached dataset exists
        if self.cache_file and not self.force_reload:
            cache_path = os.path.join(self.data_dir, self.cache_file)
            if os.path.exists(cache_path):
                print(f"Loading cached dataset from {cache_path}")
                if self._load_cached_dataset(cache_path):
                    print("Successfully loaded cached dataset")
                    return
                else:
                    print("Failed to load cached dataset. Processing from scratch.")

        print("Processing dataset from source files...")
        x_path = os.path.join(self.data_dir, "X.csv")

        self._data = pd.read_csv(x_path, sep=",")

        # Ensure data is sorted by datetime
        self._data = self._data.sort_values(by=["DateTime"])

        # Save original data before normalization
        self._data_original = self._data.copy()

        # Apply normalization for each feature across all cities
        self._normalize_data()

        if self.smooth_data:
            self._apply_smoothing()
        # Save scalers for later denormalization
        self._save_scalers()

        # Save processed dataset to cache if enabled
        if self.cache_file:
            cache_path = os.path.join(self.data_dir, self.cache_file)
            self._save_cached_dataset(cache_path)

    def _save_cached_dataset(self, cache_path):
        """Save the processed dataset to disk."""
        try:
            cache_dir = os.path.dirname(cache_path)
            os.makedirs(cache_dir, exist_ok=True)

            cache_data = {
                "data": self._data,
                "data_original": self._data_original,
                "scalers": self.scalers,
            }

            with open(cache_path, "wb") as f:
                pickle.dump(cache_data, f)

            print(f"Dataset cached to {cache_path}")
            return True
        except Exception as e:
            print(f"Failed to cache dataset: {e}")
            return False

    def _load_cached_dataset(self, cache_path):
        """Load the processed dataset from disk."""
        try:
            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)

            self._data = cache_data["data"]
            self._data_original = cache_data["data_original"]
            self.scalers = cache_data["scalers"]

            print("Cached dataset loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading cached dataset: {e}")
            return False

    def _normalize_data(self):
        """Normalize the data using MinMaxScaler for all features."""
        # List of weather variables to normalize
        variables = ["NO2", "P", "SQ", "WD", "Wvh", "dewP", "temp"]

        for var in variables:
            # Create a scaler for each variable type
            scaler = MinMaxScaler()

            # Check if we have the city prefix or just the variable name
            if any(f"{city}_{var}" in self._data.columns for city in self.cities):
                # Collect columns for this variable across all cities with city prefix
                cols = []
                for city in self.cities:
                    col_name = f"{city}_{var}"
                    if col_name in self._data.columns:
                        cols.append(col_name)

                if not cols:
                    continue  # Skip if no columns found for this variable

                # Fit the scaler on all data for this variable
                all_data = self._data[cols].values.flatten().reshape(-1, 1)
                scaler.fit(all_data)

                # Store the scaler for later use
                self.scalers[var] = scaler

                # Apply normalization to data
                for col in cols:
                    self._data[col] = scaler.transform(
                        self._data[col].values.reshape(-1, 1)
                    ).flatten()

            elif var in self._data.columns:
                # If column is present without city prefix (e.g., just 'NO2')
                all_data = self._data[var].values.reshape(-1, 1)
                scaler.fit(all_data)
                self.scalers[var] = scaler
                self._data[var] = scaler.transform(all_data).flatten()

    def _save_scalers(self):
        """Save the MinMaxScaler parameters for later denormalization."""
        import pickle
        import os

        # Create a directory for scalers if it doesn't exist
        scalers_dir = os.path.join(self.data_dir, "scalers")
        os.makedirs(scalers_dir, exist_ok=True)

        # Save each scaler
        for var, scaler in self.scalers.items():
            scaler_path = os.path.join(scalers_dir, f"{var}_scaler.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)

        print(f"Saved scalers to {scalers_dir}")

    def denormalize_no2(self, normalized_values):
        """Denormalize NO2 predictions using the saved scaler."""
        if "NO2" in self.scalers:
            return (
                self.scalers["NO2"]
                .inverse_transform(normalized_values.reshape(-1, 1))
                .flatten()
            )
        else:
            print("Warning: NO2 scaler not found. Returning original values.")
            return normalized_values

    def _get_edges(self):
        """Define the edges between the cities."""
        # Define connections between the cities (fully connected graph)
        city_pairs = [
            (0, 1),  # amsterdam -> rotterdam
            (1, 0),  # rotterdam -> amsterdam
            (0, 2),  # amsterdam -> utrecht
            (2, 0),  # utrecht -> amsterdam
            (1, 2),  # rotterdam -> utrecht
            (2, 1),  # utrecht -> rotterdam
        ]
        self._edges = np.array(city_pairs).T

    def _get_edge_weights(self):
        """Define the edge weights as distances between cities (in km)."""
        # Distances between cities in kilometers
        distances = {
            (0, 1): 57,  # amsterdam -> rotterdam
            (1, 0): 57,  # rotterdam -> amsterdam
            (0, 2): 35,  # amsterdam -> utrecht
            (2, 0): 35,  # utrecht -> amsterdam
            (1, 2): 47,  # rotterdam -> utrecht
            (2, 1): 47,  # utrecht -> rotterdam
        }

        raw_distances = np.array(
            [distances[edge] for edge in zip(self._edges[0], self._edges[1])]
        )
        # Normalize edge weights
        # Inverse distance with small epsilon to avoid division by zero
        inv_distances = 1.0 / (raw_distances + 1e-6)

        # Normalize to [0, 1]
        self._edge_weights = inv_distances / inv_distances.max()

    def _get_targets_and_features(self):
        """
        Extract lagged features including all variables and targets from the data.
        Features are structured to preserve both temporal and variable relationships.
        """
        # List of variables to include as features
        if self.only_no2:
            variables = ["NO2"]
        else:
            variables = ["NO2", "P", "SQ", "WD", "Wvh", "dewP", "temp"]

        # Process data organized by city_name column (your current data format)
        stacked_data_by_variable = {var: [] for var in variables}

        # Get unique city_ids from the data
        city_ids = sorted(self._data["city_name"].unique())

        # For each time step, create a row with values for each city
        unique_datetimes = sorted(self._data["DateTime"].unique())

        # Sample datetimes early to reduce memory usage
        if hasattr(self, "sample_size") and self.sample_size is not None:
            if isinstance(self.sample_size, float) and 0 < self.sample_size < 1:
                # Use a fraction of the datetimes
                n_samples = int(len(unique_datetimes) * self.sample_size)
                unique_datetimes = unique_datetimes[:n_samples]
                print(f"Using {n_samples} timestamps ({self.sample_size:.2%} of data)")
            elif isinstance(self.sample_size, int) and self.sample_size > 0:
                # Use specified number of timestamps
                n_samples = min(self.sample_size, len(unique_datetimes))
                unique_datetimes = unique_datetimes[:n_samples]
                print(f"Using {n_samples} timestamps (out of {len(unique_datetimes)})")

            # Continue processing with reduced set of datetimes
            for dt in unique_datetimes:
                rows_by_variable = {var: [] for var in variables}
                valid_data = True

                for city_id in city_ids:
                    city_data = self._data[
                        (self._data["DateTime"] == dt)
                        & (self._data["city_name"] == city_id)
                    ]

                    if len(city_data) == 0:
                        valid_data = False
                        break

                    # Get each variable value
                    for var in variables:
                        if var in city_data.columns:
                            rows_by_variable[var].append(city_data[var].values[0])
                        else:
                            valid_data = False
                            break

                if valid_data:
                    # Add data for this timestamp
                    for var in variables:
                        stacked_data_by_variable[var].append(rows_by_variable[var])

            # Convert lists to numpy arrays
            for var in variables:
                stacked_data_by_variable[var] = np.array(stacked_data_by_variable[var])

        # Create features (lagged values for all variables)
        self.features = []
        for i in range(len(stacked_data_by_variable["NO2"]) - self.lags):
            # For each city, create a feature array with all variables and lags
            city_features = []

            for city_idx in range(
                len(city_ids if "city_ids" in locals() else self.cities)
            ):
                # Extract data for this city across all variables and lags
                city_var_lags = []

                for var in variables:
                    # Get lagged data for this variable
                    var_values = stacked_data_by_variable[var][
                        i : i + self.lags, city_idx
                    ]
                    city_var_lags.extend(var_values)

                # Add features for this city (flattened variables × lags)
                city_features.append(city_var_lags)

            # Shape: [num_cities, num_variables * lags]
            self.features.append(np.array(city_features))

        # Create targets (next NO2 value after the lags)
        self.targets = []
        for i in range(len(stacked_data_by_variable["NO2"]) - self.lags):
            # Get NO2 values for the next timestep for all cities
            target = stacked_data_by_variable["NO2"][i + self.lags, :]
            self.targets.append(target)

        print(f"Created {len(self.features)} temporal snapshots")
        print(
            f"Feature shape: {self.features[0].shape} (num_cities, features_per_city)"
        )
        print(f"Target shape: {len(self.targets[0])} (num_cities)")

        # Describe the feature composition
        num_vars = len(variables)
        print(
            f"Each city node has features from {num_vars} variables across {self.lags} time lags"
        )
        print(f"Variables used: {', '.join(variables)}")

    def get_dataset(
        self, lags=24, only_no2=None, sample_size=None, cache=True, cache_suffix=None
    ) -> StaticGraphTemporalSignal:
        """Return the NO2 forecasting dataset with lagged features.

        Args:
            lags (int, optional): The number of time lags. Defaults to 24.
            only_no2 (bool, optional): If provided, overrides the instance setting for using only NO2 as features.
            sample_size (int or float, optional): If int, use only that many samples.
                                                If float between 0-1, use that fraction of data.
                                                If None, use all data. Defaults to None.
            cache (bool, optional): Whether to cache the processed dataset. Defaults to True.
            cache_suffix (str, optional): Suffix to add to the cache filename to differentiate different parameter sets.
                                          Defaults to None.
        """
        # Create a specific cache file for this parameter set if caching is enabled
        dataset_cache_file = None
        if cache and self.cache_file:
            # Include sample_size in the cache filename
            sample_tag = ""
            if sample_size is not None:
                if isinstance(sample_size, float):
                    sample_tag = f"_s{int(sample_size * 100)}"
                else:
                    sample_tag = f"_s{sample_size}"

            suffix = f"_l{lags}{sample_tag}" + (
                f"_{cache_suffix}" if cache_suffix else ""
            )
            dataset_cache_file = f"dataset{suffix}.pkl"
            dataset_cache_path = os.path.join(self.data_dir, dataset_cache_file)

            # Try to load cached dataset
            if os.path.exists(dataset_cache_path):
                try:
                    with open(dataset_cache_path, "rb") as f:
                        cached_result = pickle.load(f)
                    print(f"Loaded cached dataset from {dataset_cache_path}")
                    return cached_result
                except Exception as e:
                    print(f"Failed to load cached dataset: {e}")

        self.lags = lags
        if only_no2 is not None:
            self.only_no2 = only_no2
        self._get_edges()
        self._get_edge_weights()

        # Store sample_size for use in _get_targets_and_features
        self.sample_size = sample_size

        # Now _get_targets_and_features will use the sample_size
        self._get_targets_and_features()

        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )

        # Cache the dataset if requested
        if cache and dataset_cache_file:
            try:
                with open(dataset_cache_path, "wb") as f:
                    pickle.dump(dataset, f)
                print(f"Dataset cached to {dataset_cache_path}")
            except Exception as e:
                print(f"Failed to cache dataset: {e}")

        return dataset

    def get_batched_dataset(
        self,
        lags=24,
        batch_size=32,
        only_no2=None,
        sample_size=None,
        cache=True,
        cache_suffix=None,
    ) -> StaticGraphTemporalSignalBatch:
        """Return batched NO2 forecasting dataset using StaticGraphTemporalSignalBatch.

        Args:
            lags (int, optional): The number of time lags. Defaults to 24.
            batch_size (int, optional): Size of each batch. Defaults to 32.
            only_no2 (bool, optional): If provided, overrides the instance setting for using only NO2 as features.
            sample_size (int or float, optional): If int, use only that many samples.
                                                If float between 0-1, use that fraction of data.
                                                If None, use all data. Defaults to None.
            cache (bool, optional): Whether to cache the processed dataset. Defaults to True.
            cache_suffix (str, optional): Suffix to add to the cache filename. Defaults to None.

        Returns:
            StaticGraphTemporalSignalBatch: Batched temporal graph dataset
        """
        # Create a specific cache file for this parameter set if caching is enabled
        batch_cache_file = None
        if cache and self.cache_file:
            # Include parameters in the cache filename
            suffix = f"_l{lags}_b{batch_size}"
            if sample_size is not None:
                if isinstance(sample_size, float):
                    suffix += f"_s{int(sample_size * 100)}"
                else:
                    suffix += f"_s{sample_size}"
            if only_no2:
                suffix += "_no2only"
            if cache_suffix:
                suffix += f"_{cache_suffix}"

            batch_cache_file = f"batched_dataset{suffix}.pkl"
            batch_cache_path = os.path.join(self.data_dir, batch_cache_file)

            # Try to load cached dataset
            if os.path.exists(batch_cache_path):
                try:
                    with open(batch_cache_path, "rb") as f:
                        cached_result = pickle.load(f)
                    print(f"Loaded cached batched dataset from {batch_cache_path}")
                    return cached_result
                except Exception as e:
                    print(f"Failed to load cached batched dataset: {e}")

        # Processing parameters
        self.lags = lags
        if only_no2 is not None:
            self.only_no2 = only_no2
        self._get_edges()
        self._get_edge_weights()

        # Store sample_size for use in _get_targets_and_features
        self.sample_size = sample_size

        # Get features and targets
        self._get_targets_and_features()

        # Get number of cities
        num_cities = len(self.cities)

        # Shape: [num_cities]
        batches = np.zeros(num_cities, dtype=np.int64)

        # Create the batched dataset with proper sequencing
        # We're passing a list of features and targets, where each element represents a timestep
        batched_dataset = StaticGraphTemporalSignalBatch(
            edge_index=self._edges,
            edge_weight=self._edge_weights,
            features=self.features,  # Already in right format: list of [num_cities, feature_dim] arrays
            targets=self.targets,  # Already in right format: list of [num_cities] arrays
            batches=batches,  # Each node (city) belongs to the same batch
        )

        # Cache the dataset if requested
        if cache and batch_cache_file:
            try:
                with open(batch_cache_path, "wb") as f:
                    pickle.dump(batched_dataset, f)
                print(f"Batched dataset cached to {batch_cache_path}")
            except Exception as e:
                print(f"Failed to cache batched dataset: {e}")

        return batched_dataset

    def get_index_dataset(
        self,
        lags=24,
        batch_size=4,
        shuffle=False,
        allGPU=-1,
        ratio=(0.7, 0.1, 0.2),
        dask_batching=False,
        only_no2=None,
        sample_size=None,
        horizon=None,
        cache=True,
        cache_suffix=None,
        step_size=24,
    ):
        """
        Returns torch dataloaders using index batching for NO2 forecasting dataset.

        Args:
            lags (int, optional): The number of time lags. Defaults to 24.
            batch_size (int, optional): Batch size. Defaults to 4.
            shuffle (bool, optional): If the data should be shuffled. Defaults to False.
            allGPU (int, optional): GPU device ID for preprocessing. If -1, uses CPU. Defaults to -1.
            ratio (tuple of float, optional): Train, validation, test split ratios. Defaults to (0.7, 0.1, 0.2).
            dask_batching (bool, optional): Whether to use dask for lazy loading. Defaults to False.
            only_no2 (bool, optional): If provided, overrides the instance setting for features.
            sample_size (int or float, optional): If int, use that many samples.
                                                 If float between 0-1, use that fraction.
                                                 If None, use all data. Defaults to None.
            horizon (int, optional): Prediction horizon. Defaults to None.
            cache (bool, optional): Whether to cache the processed dataset. Defaults to True.
            cache_suffix (str, optional): Suffix for cache filename. Defaults to None.

        Returns:
            Tuple: (train_dataloader, val_dataloader, test_dataloader, edges, edge_weights)
        """
        # Create a specific cache file for this parameter set if caching is enabled
        self.lags = lags
        index_dataset_cache_file = None
        if cache and self.cache_file:
            suffix = f"_l{lags}_b{batch_size}_r{ratio[0]}-{ratio[1]}-{ratio[2]}"
            if sample_size:
                suffix += f"_s{sample_size}"
            if horizon:
                suffix += f"_h{horizon}"
            if cache_suffix:
                suffix += f"_{cache_suffix}"

            suffix += "_no2only" if only_no2 else "_allvars"

            if self.smooth_data and cache_suffix:
                cache_suffix = f"{cache_suffix}_smooth{self.smooth_window}"
            elif self.smooth_data:
                suffix = f"smooth{self.smooth_window}"

            index_dataset_cache_file = f"index_dataset{suffix}.pkl"
            index_dataset_cache_path = os.path.join(
                self.data_dir, index_dataset_cache_file
            )

            # Try to load cached index dataset
            if os.path.exists(index_dataset_cache_path):
                try:
                    with open(index_dataset_cache_path, "rb") as f:
                        train_dl, val_dl, test_dl, edges, edge_weights = pickle.load(f)
                    print(
                        f"Loaded cached index dataset from {index_dataset_cache_path}"
                    )
                    return train_dl, val_dl, test_dl, edges, edge_weights
                except Exception as e:
                    print(f"Failed to load cached index dataset: {e}")

        if not self.index:
            raise ValueError(
                "get_index_dataset requires 'index=True' in the constructor."
            )

        # Define which variables to include
        if only_no2 is not None:
            self.only_no2 = only_no2

        if self.only_no2:
            variables = ["NO2"]
        else:
            variables = ["NO2", "P", "SQ", "WD", "Wvh", "dewP", "temp"]

        self._get_edges()
        self._get_edge_weights()

        # Check if we have city prefixes in the data
        has_city_prefix = any(
            f"{city}_NO2" in self._data.columns for city in self.cities
        )

        if has_city_prefix:
            # Get data for each city and each variable
            variable_data = []
            for var in variables:
                var_data = np.stack(
                    [self._data[f"{city}_{var}"].values for city in self.cities], axis=1
                )
                variable_data.append(var_data)

            # Stack all variable data together
            if len(variable_data) > 1:
                data = np.concatenate(variable_data, axis=1)
            else:
                data = variable_data[0]
        else:
            # Process data organized by city_name column
            stacked_data = []

            # Get unique city_ids from the data
            city_ids = sorted(self._data["city_name"].unique())

            # For each time step, create a row with values for each city
            unique_datetimes = sorted(self._data["DateTime"].unique())

            for dt in unique_datetimes:
                row = []
                valid_data = True

                for city_id in city_ids:
                    city_data = self._data[
                        (self._data["DateTime"] == dt)
                        & (self._data["city_name"] == city_id)
                    ]

                    if len(city_data) == 0:
                        valid_data = False
                        break

                    # Get values for each variable
                    city_values = []
                    for var in variables:
                        if var in city_data.columns:
                            city_values.append(city_data[var].values[0])
                        else:
                            valid_data = False
                            break

                    if valid_data:
                        row.extend(city_values)
                    else:
                        break

                if valid_data:
                    stacked_data.append(row)

            # Convert to numpy array
            data = np.array(stacked_data)
        num_samples = data.shape[0]

        # Apply sample_size if specified
        if sample_size is not None:
            if isinstance(sample_size, float) and 0 < sample_size <= 1:
                num_samples_to_use = max(round(num_samples * sample_size), self.lags)
                data = data[:num_samples_to_use]
                print(f"Using {num_samples_to_use} samples ({sample_size:.2%} of data)")

            elif isinstance(sample_size, int) and sample_size > 0:
                # Use specified number of samples
                num_samples_to_use = min(sample_size, num_samples)
                data = data[:num_samples_to_use]
                print(f"Using {num_samples_to_use} samples (out of {num_samples})")
            else:
                raise ValueError(
                    "sample_size must be a positive int or float between 0-1"
                )

            # Adjust num_samples after sampling
            num_samples = data.shape[0]

        # Create tensor versions of edges and weights
        edges = torch.tensor(self._edges, dtype=torch.int64)
        edge_weights = torch.tensor(self._edge_weights, dtype=torch.float)

        # Now create indices for train/val/test split
        # Note: x_i should account for lags and horizon
        print("x_i lags", lags)
        x_i = np.arange(0, num_samples - lags - horizon, step=step_size)

        # Recalculate number of samples after adjustments
        num_samples = x_i.shape[0]

        # Recalculate split sizes
        num_train = round(num_samples * ratio[0])
        num_val = round(num_samples * ratio[1])
        num_test = num_samples - num_train - num_val

        # Split indices
        x_train = x_i[:num_train]
        x_val = x_i[num_train : num_train + num_val]
        x_test = x_i[-num_test:]

        # Create datasets
        train_dataset = self.IndexDataset(
            x_train,
            data,
            horizon,
            gpu=(allGPU != -1),
            lazy=dask_batching,
            lags=self.lags,
        )
        val_dataset = self.IndexDataset(
            x_val, data, horizon, gpu=(allGPU != -1), lazy=dask_batching, lags=self.lags
        )
        test_dataset = self.IndexDataset(
            x_test,
            data,
            horizon,
            gpu=(allGPU != -1),
            lazy=dask_batching,
            lags=self.lags,
        )

        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle
        )
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # At the end of the method, cache the result
        result = (
            train_dataloader,
            val_dataloader,
            test_dataloader,
            edges,
            edge_weights,
        )
        if cache and index_dataset_cache_file:
            try:
                with open(index_dataset_cache_path, "wb") as f:
                    pickle.dump(result, f)
                print(f"Index dataset cached to {index_dataset_cache_path}")
            except Exception as e:
                print(f"Failed to cache index dataset: {e}")

        return result
