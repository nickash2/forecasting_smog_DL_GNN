import os
import numpy as np
import pandas as pd
import torch
import pickle
import hashlib
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from ..utils.index_dataset import IndexDataset
from torch_geometric_temporal.signal import StaticGraphTemporalSignalBatch


class NO2DatasetLoader:
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

    # Set a version for cache compatibility checks
    CACHE_VERSION = 72.2449

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

        # Set up cache directory
        self.cache_dir = os.path.join(self.data_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self._read_data()

        if index:
            self.IndexDataset = IndexDataset

    def _get_cache_path(self, cache_name, params=None):
        """Generate a standardized cache file path based on parameters.

        Args:
            cache_name (str): Base name for the cache file
            params (dict, optional): Dictionary of parameters to include in the cache key

        Returns:
            str: Full path to the cache file
        """
        if not self.cache_file:
            return None

        # Start with base name
        filename = cache_name

        # Add parameter hash if provided
        if params:
            # Convert params to a string and hash it
            param_str = str(sorted(params.items()))
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
            filename = f"{filename}_{param_hash}"

        # Add version to prevent using incompatible caches
        filename = f"{filename}_v{self.CACHE_VERSION}.pkl"

        return os.path.join(self.cache_dir, filename)

    def _load_cache(self, cache_path):
        """Load data from cache file.

        Args:
            cache_path (str): Path to the cache file

        Returns:
            object or None: Cached data if successful, None otherwise
        """
        if not cache_path or not os.path.exists(cache_path) or self.force_reload:
            return None

        try:
            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)
            print(f"Loaded from cache: {cache_path}")
            return cache_data
        except Exception as e:
            print(f"Failed to load cache ({cache_path}): {e}")
            return None

    def _save_cache(self, cache_path, data):
        """Save data to cache file.

        Args:
            cache_path (str): Path to the cache file
            data: Data to be cached

        Returns:
            bool: True if successful, False otherwise
        """
        if not cache_path:
            return False

        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)

            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
            print(f"Saved to cache: {cache_path}")
            return True
        except Exception as e:
            print(f"Failed to save cache ({cache_path}): {e}")
            return False

    def _read_data(self):
        """Read the data from CSV files and combine them for lagged feature approach."""
        # Get cache path for raw data
        cache_path = self._get_cache_path("raw_data")

        # Try to load from cache
        cache_data = self._load_cache(cache_path)
        if cache_data is not None:
            self._data = cache_data["data"]
            self._data_original = cache_data["data_original"]
            self.scalers = cache_data["scalers"]
            return

        print("Processing dataset from source files...")
        x_path = os.path.join(self.data_dir, "X.csv")

        self._data = pd.read_csv(x_path, sep=",")

        # Ensure data is sorted by datetime
        self._data = self._data.sort_values(by=["DateTime"])

        # Save original data before normalization
        self._data_original = self._data.copy()

        # Apply normalization for each feature across all cities
        self._normalize_data()

        # Save scalers for later denormalization
        self._save_scalers()

        # Save processed dataset to cache
        cache_data = {
            "data": self._data,
            "data_original": self._data_original,
            "scalers": self.scalers,
        }
        self._save_cache(cache_path, cache_data)

    def _save_scalers(self):
        """Save the MinMaxScaler parameters for later denormalization."""
        # Create a directory for scalers if it doesn't exist
        scalers_dir = os.path.join(self.data_dir, "scalers")
        os.makedirs(scalers_dir, exist_ok=True)

        # Save each scaler
        for var, scaler in self.scalers.items():
            scaler_path = os.path.join(scalers_dir, f"{var}_scaler.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)

        print(f"Saved scalers to {scalers_dir}")

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
        """Return the NO2 forecasting dataset with lagged features."""
        # Create parameters dictionary for cache key
        params = {
            "lags": lags,
            "only_no2": self.only_no2 if only_no2 is None else only_no2,
            "sample_size": sample_size,
            "cache_suffix": cache_suffix,
        }

        # Get cache path
        cache_path = self._get_cache_path("dataset", params) if cache else None

        # Try to load from cache
        cached_result = self._load_cache(cache_path)
        if cached_result is not None:
            return cached_result

        # Process the dataset if not cached
        self.lags = lags
        if only_no2 is not None:
            self.only_no2 = only_no2
        self._get_edges()
        self._get_edge_weights()

        # Store sample_size for use in _get_targets_and_features
        self.sample_size = sample_size

        # Get features and targets
        self._get_targets_and_features()

        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )

        # Cache the result
        if cache_path:
            self._save_cache(cache_path, dataset)

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
        """Return batched NO2 forecasting dataset using StaticGraphTemporalSignalBatch."""
        # Create parameters dictionary for cache key
        params = {
            "lags": lags,
            "batch_size": batch_size,
            "only_no2": self.only_no2 if only_no2 is None else only_no2,
            "sample_size": sample_size,
            "cache_suffix": cache_suffix,
        }

        # Get cache path
        cache_path = self._get_cache_path("batched_dataset", params) if cache else None

        # Try to load from cache
        cached_result = self._load_cache(cache_path)
        if cached_result is not None:
            return cached_result

        # Process the dataset if not cached
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

        # Create the batched dataset
        batched_dataset = StaticGraphTemporalSignalBatch(
            edge_index=self._edges,
            edge_weight=self._edge_weights,
            features=self.features,  # Already in right format: list of [num_cities, feature_dim] arrays
            targets=self.targets,  # Already in right format: list of [num_cities] arrays
            batches=batches,  # Each node (city) belongs to the same batch
        )

        # Cache the result
        if cache_path:
            self._save_cache(cache_path, batched_dataset)

        return batched_dataset

    def _split_by_time(self, timestamps, original_indices, horizon):
        """Split data based on specific, potentially non-contiguous time periods.

        Args:
            timestamps: Array of timestamps corresponding to potential sample start dates.
            original_indices: The original indices (e.g., from x_i) corresponding to timestamps.
            horizon: Prediction horizon (used for validation).

        Returns:
            Tuple of (x_train, x_val, x_test) indices from original_indices.
        """
        if not isinstance(timestamps, pd.DatetimeIndex):
            timestamps = pd.to_datetime(timestamps)

        # Ensure we have a pandas Series for easier boolean indexing
        ts_series = pd.Series(timestamps)

        # --- Define Date Ranges ---
        # Training Ranges
        train_mask = (
            (
                (ts_series.dt.year == 2017)
                & (ts_series >= "2017-08-01")
                & (ts_series <= "2017-12-30")
            )
            | (
                (ts_series.dt.year == 2018)
                & (ts_series >= "2018-08-01")
                & (ts_series <= "2018-12-30")
            )
            | (
                (ts_series.dt.year == 2020)
                & (ts_series >= "2020-08-01")
                & (ts_series <= "2020-12-30")
            )
            | (
                (ts_series.dt.year == 2021)
                & (ts_series >= "2021-08-01")
                & (ts_series <= "2021-11-18")
            )
            | (
                (ts_series.dt.year == 2022)
                & (ts_series >= "2022-08-01")
                & (ts_series <= "2022-11-18")
            )
        )

        # Validation Ranges
        val_mask = (
            (
                (ts_series.dt.year == 2021)
                & (ts_series >= "2021-11-19")
                & (ts_series <= "2021-12-09")
            )
            | (
                (ts_series.dt.year == 2022)
                & (ts_series >= "2022-11-19")
                & (ts_series <= "2022-12-09")
            )
            | (
                (ts_series.dt.year == 2023)
                & (ts_series >= "2023-08-01")
                & (ts_series <= "2023-10-02")
            )
        )

        # Testing Ranges
        test_mask = (
            (
                (ts_series.dt.year == 2021)
                & (ts_series >= "2021-12-10")
                & (ts_series <= "2021-12-30")
            )  # Starts after validation ends
            | (
                (ts_series.dt.year == 2022)
                & (ts_series >= "2022-12-10")
                & (ts_series <= "2022-12-30")
            )  # Starts after validation ends
            | (
                (ts_series.dt.year == 2023)
                & (ts_series >= "2023-10-03")
                & (ts_series <= "2023-12-04")
            )
        )

        # --- Get Indices ---
        x_train = original_indices[train_mask.values]
        x_val = original_indices[val_mask.values]
        x_test = original_indices[test_mask.values]

        # --- Validation and Logging ---
        # Check for overlaps
        train_set, val_set, test_set = set(x_train), set(x_val), set(x_test)
        if (
            train_set.intersection(val_set)
            or train_set.intersection(test_set)
            or val_set.intersection(test_set)
        ):
            print(
                "Warning: Overlap detected between train/val/test sets after time splitting."
            )
            # You might want to raise an error or investigate further depending on the cause

        # Log the split periods found
        if len(x_train) > 0:
            print(
                f"Training period(s) found: {timestamps[train_mask].min()} to {timestamps[train_mask].max()}"
            )
        else:
            print("Warning: No training data found for the specified ranges.")
        if len(x_val) > 0:
            print(
                f"Validation period(s) found: {timestamps[val_mask].min()} to {timestamps[val_mask].max()}"
            )
        else:
            print("Warning: No validation data found for the specified ranges.")
        if len(x_test) > 0:
            print(
                f"Test period(s) found: {timestamps[test_mask].min()} to {timestamps[test_mask].max()}"
            )
        else:
            print("Warning: No testing data found for the specified ranges.")

        # Check that there's enough data in each split relative to horizon
        for split_name, split_indices in [
            ("Training", x_train),
            ("Validation", x_val),
            ("Test", x_test),
        ]:
            if len(split_indices) == 0:
                print(f"Warning: {split_name} set is empty.")
            # The IndexDataset itself handles the check if len < horizon internally,
            # so we don't strictly need the horizon check here anymore.

        # Calculate and print ratios
        total_samples = len(x_train) + len(x_val) + len(x_test)
        if total_samples > 0:
            train_ratio = len(x_train) / total_samples * 100
            val_ratio = len(x_val) / total_samples * 100
            test_ratio = len(x_test) / total_samples * 100
            print(
                f"Split Ratios: Train={train_ratio:.1f}% / Val={val_ratio:.1f}% / Test={test_ratio:.1f}%"
            )
        else:
            print("No samples found in any split.")

        return x_train, x_val, x_test

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
        split_dates=None,
        use_time_split=False,
        target_offset=72 - 24 + 1,
    ):
        """Returns torch dataloaders using index batching for NO2 forecasting dataset.

        Args:
            # ...existing arguments...
            split_dates: Tuple of (train_end, val_end) dates as strings in 'YYYY-MM-DD' format
            use_time_split: If True, split by time periods rather than percentages
        """
        # Create parameters dictionary for cache key
        params = {
            "lags": lags,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "ratio": ratio if not use_time_split else None,
            "only_no2": self.only_no2 if only_no2 is None else only_no2,
            "sample_size": sample_size,
            "horizon": horizon,
            "cache_suffix": cache_suffix,
            "step_size": step_size,
            "split_dates": split_dates,
            "use_time_split": use_time_split,
        }

        # Get cache path
        cache_path = self._get_cache_path("index_dataset", params) if cache else None

        # Try to load from cache
        cached_result = self._load_cache(cache_path)
        if cached_result is not None:
            return cached_result

        if not self.index:
            raise ValueError(
                "get_index_dataset requires 'index=True' in the constructor."
            )

        # Set parameters for data processing
        self.lags = lags
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

        # Get timestamps corresponding to the raw data points used
        if has_city_prefix:
            # Assuming DateTime index or column exists and matches the 'data' array
            all_timestamps = pd.to_datetime(
                self._data.index
                if self._data.index.name == "DateTime"
                else self._data["DateTime"]
            )
            # If sampling was applied, timestamps need to match the sampled 'data' array
            if sample_size is not None:
                all_timestamps = all_timestamps[: data.shape[0]]
        else:
            # Use the unique_datetimes collected during data stacking
            all_timestamps = pd.to_datetime(unique_datetimes)
            # If sampling was applied, timestamps need to match the sampled 'data' array
            if sample_size is not None:
                all_timestamps = all_timestamps[: data.shape[0]]

        # Now create indices for train/val/test split

        num_samples = data.shape[0]
        upper_bound_exclusive = num_samples - target_offset - horizon + 1

        if upper_bound_exclusive <= 0:
            raise ValueError(
                "Data is too short for the given lags, horizon, and target_offset."
            )

        x_i = np.arange(0, upper_bound_exclusive, step=step_size)

        # Get the actual timestamps for these potential start indices
        timestamps_for_x_i = all_timestamps[x_i]

        # Split based on time or percentages
        if use_time_split:
            print("Using custom time-based split...")
            # Pass the timestamps corresponding to x_i and x_i itself
            x_train, x_val, x_test = self._split_by_time(
                timestamps_for_x_i, x_i, horizon
            )
        else:
            # Use the original percentage-based split
            print(f"Using percentage-based split: {ratio}")
            num_samples_in_split = x_i.shape[0]  # Number of potential sequences

            # Recalculate split sizes based on the number of sequences
            num_train = round(num_samples_in_split * ratio[0])
            num_val = round(num_samples_in_split * ratio[1])
            num_test = num_samples_in_split - num_train - num_val

            # Split indices from x_i
            x_train = x_i[:num_train]
            x_val = x_i[num_train : num_train + num_val]
            x_test = x_i[-num_test:]  # Ensure all indices are used

            # Log the split periods found using percentage split
            if len(x_train) > 0:
                print(
                    f"Training period (percentage split): {timestamps_for_x_i[0]} to {timestamps_for_x_i[num_train - 1]}"
                )
            if len(x_val) > 0:
                print(
                    f"Validation period (percentage split): {timestamps_for_x_i[num_train]} to {timestamps_for_x_i[num_train + num_val - 1]}"
                )
            if len(x_test) > 0:
                print(
                    f"Test period (percentage split): {timestamps_for_x_i[num_train + num_val]} to {timestamps_for_x_i[-1]}"
                )

        # If smoothing is enabled, apply it only to training data
        if self.smooth_data:
            print("Applying smoothing to training data...")
            # Create a copy of the data for smoothing to avoid modifying the original
            train_data_smoothed = data.copy()  # Use the potentially sampled 'data'

            # Apply smoothing only to training indices
            # For each variable and city, apply smoothing
            if has_city_prefix:
                variables_to_smooth = (
                    ["NO2"]
                    if self.only_no2
                    else ["NO2", "P", "SQ", "WD", "Wvh", "dewP", "temp"]
                )
                num_cities = len(self.cities)
                num_vars = len(variables_to_smooth)

                # Determine the column indices in the 'data' array
                col_indices = {}
                current_idx = 0
                all_vars_in_data = variables  # Use the full list used to create 'data'
                for var in all_vars_in_data:
                    for city_idx in range(num_cities):
                        # Store index for (city, var) pair
                        col_indices[(city_idx, var)] = current_idx
                        current_idx += 1

                for var in variables_to_smooth:
                    for city_idx, city in enumerate(self.cities):
                        col_idx_in_data = col_indices.get((city_idx, var))

                        if col_idx_in_data is not None:
                            # Extract the relevant column data
                            column_data = train_data_smoothed[:, col_idx_in_data]

                            # Apply moving average using pandas Series for easy handling
                            smooth_values = (
                                pd.Series(column_data)  # Use the whole column
                                .rolling(
                                    window=self.smooth_window,
                                    min_periods=1,
                                    center=True,
                                )  # Use min_periods=1
                                .mean()
                                .fillna(method="bfill")
                                .fillna(method="ffill")
                            )

                            # Update the smoothed data array
                            train_data_smoothed[:, col_idx_in_data] = (
                                smooth_values.values
                            )

                            print(f"Applied smoothing to {city}_{var}")
            else:
                # Smoothing for non-prefixed data might need adjustment based on 'data' structure
                print(
                    "Warning: Smoothing for non-prefixed data structure not fully implemented."
                )

            # Use smoothed data only for training dataset creation
            train_dataset = self.IndexDataset(
                x_train,
                train_data_smoothed,  # Use the smoothed data
                horizon,
                gpu=(allGPU != -1),
                lazy=dask_batching,
                lags=self.lags,
                target_offset=target_offset,  # Pass target_offset
            )
        else:
            # Use original data for training dataset creation
            train_dataset = self.IndexDataset(
                x_train,
                data,  # Original data
                horizon,
                gpu=(allGPU != -1),
                lazy=dask_batching,
                lags=self.lags,
                target_offset=target_offset,  # Pass target_offset
            )

        # Use original data for validation and test
        val_dataset = self.IndexDataset(
            x_val,
            data,  # Original data
            horizon,
            gpu=(allGPU != -1),
            lazy=dask_batching,
            lags=self.lags,
            target_offset=target_offset,  # Pass target_offset
        )
        test_dataset = self.IndexDataset(
            x_test,
            data,  # Original data
            horizon,
            gpu=(allGPU != -1),
            lazy=dask_batching,
            lags=self.lags,
            target_offset=target_offset,  # Pass target_offset
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

        # Cache the result
        if cache_path:
            self._save_cache(cache_path, result)

        return result
