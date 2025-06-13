import os
import numpy as np
import pandas as pd
import torch
import pickle
import hashlib
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from ..utils.index_dataset import IndexDataset
from torch_geometric.transforms import LaplacianLambdaMax
from torch_geometric.data import Data


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

    CACHE_VERSION = 10

    def __init__(
        self,
        index=False,
        data_dir=None,
        only_no2=False,
        cache_file="no2_dataset_cache.pkl",
        force_reload=False,
        logger=None,
    ):
        self.cities = ["amsterdam", "utrecht", "rotterdam"]
        self.index = index
        self.data_dir = data_dir
        self._initial_only_no2 = only_no2  # Store initial setting
        self.only_no2 = only_no2  # Current setting, might be overridden
        self.scalers = {}  # Scalers will be fitted on train data
        self.cache_file = cache_file
        self.force_reload = force_reload
        self._data_original = None  # Stores raw data ONLY
        self.logger = logger
        self._lambda_max = None

        self.num_nodes = len(self.cities)

        if self.data_dir is None:
            import pathlib

            try:
                base_dir = pathlib.Path(__file__).resolve().parent.parent.parent.parent
            except NameError:
                base_dir = pathlib.Path(".").resolve()
            self.data_dir = base_dir / "data" / "data_gnn"

        self.cache_dir = os.path.join(self.data_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self._read_data()  # Reads raw data into self._data_original

        if index:
            self.IndexDataset = IndexDataset

    def _get_cache_path(self, cache_name, params=None):
        if not self.cache_file:
            return None
        filename = cache_name
        if params:
            sanitized_params = {
                k: str(v) if isinstance(v, (list, tuple)) else v
                for k, v in params.items()
            }
            param_str = str(sorted(sanitized_params.items()))
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
            filename = f"{filename}_{param_hash}"
        filename = f"{filename}_v{self.CACHE_VERSION}.pkl"
        return os.path.join(self.cache_dir, filename)

    def _load_cache(self, cache_path):
        if not cache_path or not os.path.exists(cache_path) or self.force_reload:
            return None
        try:
            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)
            self.logger.info(f"Loaded from cache: {cache_path}")
            return cache_data
        except Exception as e:
            self.logger.error(f"Failed to load cache ({cache_path}): {e}")
            return None

    def _save_cache(self, cache_path, data):
        if not cache_path:
            return False
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
            self.logger.info(f"Saved to cache: {cache_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save cache ({cache_path}): {e}")
            return False

    def _read_data(self):
        """Reads data from CSV and stores it unnormalized in self._data_original."""
        # Check if already loaded unless forcing reload
        if self._data_original is not None and not self.force_reload:
            self.logger.warning("Raw data already loaded in memory.")
            return

        # Use a specific cache name for just the raw data
        raw_cache_path = self._get_cache_path("raw_data_unnormalized_v2")
        cache_data = self._load_cache(raw_cache_path)
        if cache_data is not None:
            self._data_original = cache_data["data_original"]
            self.logger.info("Loaded raw unnormalized data from cache.")
            return

        self.logger.debug("Reading raw dataset from source files...")
        x_path = os.path.join(self.data_dir, "X.csv")
        if not os.path.exists(x_path):
            raise FileNotFoundError(f"Data file not found: {x_path}")

        current_data = pd.read_csv(x_path, sep=",")
        self.logger.debug(f"Data shape: {current_data.shape}")

        if "DateTime" not in current_data.columns:
            raise ValueError("'DateTime' column not found.")
        try:
            current_data["DateTime"] = pd.to_datetime(current_data["DateTime"])
        except Exception as e:
            raise ValueError(f"Could not parse 'DateTime': {e}")
        current_data = current_data.sort_values(by=["DateTime"]).reset_index(drop=True)
        # Convert Wvh from m/s to km/h
        for city in self.cities:
            wvh_col = f"{city}_Wvh"  # Adjust column name if needed
            if wvh_col in current_data.columns:
                current_data[wvh_col] = current_data[wvh_col] * 3.6

        self._data_original = current_data

        # Cache only the raw data
        self._save_cache(raw_cache_path, {"data_original": self._data_original})

    def _fit_scalers_on_train_data(self, train_data_df):
        """Fits MinMaxScaler objects using the provided training data subset."""
        self.logger.debug(f"Current self.only_no2: {self.only_no2}")

        self.scalers = {}  # Reset scalers

        variables = (
            ["NO2"]
            if self.only_no2
            else ["NO2", "P", "SQ", "WD", "Wvh", "dewP", "temp"]
        )
        self.logger.debug(f"Variables to process: {variables}")

        has_city_prefix = any(
            f"city{idx}_{variables[0]}" in train_data_df.columns
            for idx in range(len(self.cities))
        )

        self.logger.debug(
            f"Detected city prefix pattern ('city[idx]_var'): {has_city_prefix}"
        )

        processed_vars_count = 0
        for i, var in enumerate(variables):
            self.logger.debug(f"Processing variable {i + 1}/{len(variables)}: '{var}'")
            scaler = MinMaxScaler()
            cols_to_fit = []

            if has_city_prefix:
                # Look for columns named "city0_var", "city1_var", etc.
                for idx in range(len(self.cities)):  # Use index 0, 1, 2
                    col_name = f"city{idx}_{var}"
                    if col_name in train_data_df.columns:
                        cols_to_fit.append(col_name)
            elif var in train_data_df.columns:  # Fallback if no prefix pattern found
                cols_to_fit.append(var)

            self.logger.debug(f"cols_to_fit for '{var}': {cols_to_fit}")

            if not cols_to_fit:
                self.logger.debug(f"No columns found for '{var}', skipping.")
                continue

            try:
                fitting_values = (
                    train_data_df[cols_to_fit].values.flatten().reshape(-1, 1)
                )
                self.logger.debug(
                    f"Shape of fitting_values for '{var}': {fitting_values.shape}"
                )
                valid_fitting_values = fitting_values[~np.isnan(fitting_values)]
                self.logger.debug(
                    f"Shape of valid_fitting_values for '{var}': {valid_fitting_values.shape}"
                )

                if valid_fitting_values.shape[0] == 0:
                    self.logger.debug(
                        f"No non-NaN training data for {var}. Using default [0,1] scaler."
                    )
                else:
                    scaler.fit(valid_fitting_values.reshape(-1, 1))
                    self.logger.debug(
                        f"Scaler fitted for '{var}'. Min: {scaler.data_min_}, Max: {scaler.data_max_}"
                    )

                self.scalers[var] = scaler
                self.logger.debug(f"Stored scaler for '{var}' in self.scalers")
                processed_vars_count += 1

            except Exception as e:
                self.logger.debug(f"ERROR processing/fitting scaler for '{var}': {e}")

        self.logger.debug(f"== Exiting _fit_scalers_on_train_data ==")
        self.logger.debug(
            f"Total variables processed and stored: {processed_vars_count}"
        )
        self.logger.debug(f"Final self.scalers keys: {list(self.scalers.keys())}")

    def _apply_scalers(self, data_to_transform_df):
        """Applies the scalers stored in self.scalers to a DataFrame."""
        self.logger.debug(
            f"Attempting to apply scalers. Current self.scalers keys: {list(self.scalers.keys())}"
        )
        if not self.scalers:
            raise RuntimeError(
                "Scalers have not been fitted. Call _fit_scalers_on_train_data first."
            )

        normalized_df = data_to_transform_df.copy()

        variables = (
            ["NO2"]
            if self.only_no2
            else ["NO2", "P", "SQ", "WD", "Wvh", "dewP", "temp"]
        )

        has_city_prefix = any(
            f"city{idx}_{variables[0]}" in normalized_df.columns
            for idx in range(len(self.cities))
        )
        self.logger.debug(
            f"Applying scalers - Detected city prefix pattern ('city[idx]_var'): {has_city_prefix}"
        )

        for var in variables:
            if var not in self.scalers:
                continue

            scaler = self.scalers[var]
            cols_to_transform = []

            if has_city_prefix:
                # Look for columns named "city0_var", "city1_var", etc.
                for idx in range(len(self.cities)):  # Use index 0, 1, 2
                    col_name = f"city{idx}_{var}"
                    if col_name in normalized_df.columns:
                        cols_to_transform.append(col_name)
            elif var in normalized_df.columns:  # Fallback
                cols_to_transform.append(var)

            if not cols_to_transform:
                continue

            for col in cols_to_transform:
                col_data = normalized_df[col].values.reshape(-1, 1)
                valid_mask_transform = ~np.isnan(col_data).flatten()
                if np.any(valid_mask_transform):
                    try:
                        transformed_values = scaler.transform(
                            col_data[valid_mask_transform].reshape(-1, 1)
                        )
                        normalized_df.loc[valid_mask_transform, col] = (
                            transformed_values.flatten()
                        )
                    except Exception as e:
                        self.logger.debug(f"ERROR applying scaler for {col}: {e}")
                        normalized_df.loc[valid_mask_transform, col] = np.nan

        return normalized_df

    def _save_scalers(self):
        """Save the currently active MinMaxScaler parameters (assumed fitted on train)."""
        if not self.scalers:
            return
        scalers_dir = os.path.join(self.data_dir, "scalers")
        os.makedirs(scalers_dir, exist_ok=True)
        # Suffix distinguishes these train-fitted scalers
        scaler_suffix = "_train_fitted"
        for var, scaler in self.scalers.items():
            scaler_filename = f"{var}_scaler{scaler_suffix}.pkl"
            scaler_path = os.path.join(scalers_dir, scaler_filename)
            try:
                with open(scaler_path, "wb") as f:
                    pickle.dump(scaler, f)
            except Exception as e:
                self.logger.warning(f"Warning: Could not save scaler for {var}: {e}")

    def _load_scalers(self):
        """Load the train-fitted MinMaxScaler parameters."""
        scalers_dir = os.path.join(self.data_dir, "scalers")
        loaded_all = True
        new_scalers = {}
        scaler_suffix = "_train_fitted"
        # Determine variables based on current self.only_no2 state
        variables = (
            ["NO2"]
            if self.only_no2
            else ["NO2", "P", "SQ", "WD", "Wvh", "dewP", "temp"]
        )

        for var in variables:
            scaler_filename = f"{var}_scaler{scaler_suffix}.pkl"
            scaler_path = os.path.join(scalers_dir, scaler_filename)
            if os.path.exists(scaler_path):
                try:
                    with open(scaler_path, "rb") as f:
                        new_scalers[var] = pickle.load(f)
                except Exception as e:
                    self.logger.warning(
                        f"Warning: Failed to load scaler {scaler_path}: {e}"
                    )
                    loaded_all = False
                    break
            else:
                loaded_all = False
                break

        if loaded_all and new_scalers:
            self.scalers = new_scalers
            self.logger.info(f"Successfully loaded train-fitted scalers")
            return True
        else:
            self.scalers = {}  # Reset if loading failed
            return False

    def denormalize_no2(self, normalized_values):
        """Denormalize NO2 predictions using the train-fitted scaler."""
        if "NO2" in self.scalers:
            if isinstance(normalized_values, torch.Tensor):
                normalized_values = normalized_values.detach().cpu().numpy()
            np_values = np.asarray(normalized_values)
            output = np.full_like(np_values, np.nan, dtype=float)
            valid_mask = ~np.isnan(np_values)
            if np.any(valid_mask):
                output[valid_mask] = (
                    self.scalers["NO2"]
                    .inverse_transform(np_values[valid_mask].reshape(-1, 1))
                    .flatten()
                )
            return output.flatten()
        else:
            self.logger.warning(
                "Warning: NO2 scaler not found (train-fitted scaler expected). Returning original values."
            )
            # Ensure self.scalers is checked if loaded, otherwise this error might be misleading
            return normalized_values

    def _get_edges(self):
        city_pairs = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]
        self._edges = np.array(city_pairs).T

    def _get_edge_weights(self):
        distances = {
            (0, 1): 57,
            (1, 0): 57,
            (0, 2): 35,
            (2, 0): 35,
            (1, 2): 47,
            (2, 1): 47,
        }
        if not hasattr(self, "_edges") or self._edges is None:
            self._get_edges()
        raw_distances = np.array(
            [distances[edge] for edge in zip(self._edges[0], self._edges[1])]
        )
        inv_distances = 1.0 / (raw_distances + 1e-6)
        self._edge_weights = inv_distances / inv_distances.max()

    def _split_by_time(self, timestamps, original_indices):
        if not isinstance(timestamps, pd.DatetimeIndex):
            timestamps = pd.to_datetime(timestamps)
        if len(timestamps) != len(original_indices):
            raise ValueError("Length mismatch: timestamps vs indices.")
        ts_series = pd.Series(data=timestamps.values, index=original_indices)
        train_mask_series = (
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
        val_mask_series = (
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
        test_mask_series = (
            (
                (ts_series.dt.year == 2021)
                & (ts_series >= "2021-12-10")
                & (ts_series <= "2021-12-30")
            )
            | (
                (ts_series.dt.year == 2022)
                & (ts_series >= "2022-12-10")
                & (ts_series <= "2022-12-30")
            )
            | (
                (ts_series.dt.year == 2023)
                & (ts_series >= "2023-10-03")
                & (ts_series <= "2023-12-04")
            )
        )
        # Optional validation/logging can be added here
        return train_mask_series.values, val_mask_series.values, test_mask_series.values

    def get_index_dataset(
        self,
        lags=72,
        batch_size=16,
        shuffle=False,
        allGPU=-1,
        ratio=(0.7, 0.1, 0.2),  # Used only if use_time_split=False
        dask_batching=False,
        only_no2=None,
        sample_size=None,
        horizon=24,
        cache=True,
        cache_suffix=None,
        step_size=24,
        split_dates=None,
        use_time_split=False,
        target_offset=1,
    ):
        """ """

        if not self.index:
            raise ValueError("'index=True' required.")
        if self._data_original is None:
            raise RuntimeError("Raw data not loaded. Call _read_data.")

        current_only_no2 = self._initial_only_no2 if only_no2 is None else only_no2
        self.only_no2 = current_only_no2  # Update instance state
        self.logger.info(
            f"Processing dataset with features: {'NO2 only' if self.only_no2 else 'All variables'}"
        )

        params = {  # Define params used in this function call
            "lags": lags,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "ratio": ratio if not use_time_split else None,
            "only_no2": self.only_no2,
            "sample_size": sample_size,
            "horizon": horizon,
            "cache_suffix": cache_suffix,
            "step_size": step_size,
            "split_dates": split_dates,
            "use_time_split": use_time_split,
            "target_offset": target_offset,  # normalize_with_train_only is always True now
            "CACHE_VERSION": self.CACHE_VERSION,
            "DATA_FORMAT_PIVOTED": True,  # Indicate pivoting happened for cache key
        }
        cache_name = "index_dataset_norm_train_only"
        cache_path = self._get_cache_path(cache_name, params) if cache else None

        if not self.force_reload and cache_path and os.path.exists(cache_path):
            cached_result = self._load_cache(cache_path)
            if self._load_scalers():  # Load train-fitted scalers
                if cached_result is not None:
                    self.logger.info(
                        "Loaded final dataloaders, train-fitted scalers and lambda_max from cache."
                    )
                    edges_t = torch.tensor(cached_result[3], dtype=torch.long)
                    weights_t = torch.tensor(cached_result[4], dtype=torch.float)
                    return (
                        cached_result[0],
                        cached_result[1],
                        cached_result[2],
                        edges_t,
                        weights_t,
                        cached_result[5],  # lambda_max
                    )
            else:  # Scalers missing
                self.logger.info(
                    f"Cache {cache_path} found, but train-fitted scalers not loaded. Recomputing."
                )
                self.force_reload = True

        self.lags = lags

        working_data_df_raw = self._data_original.copy()  # Start with raw data
        if sample_size is not None:
            # ... (sampling logic on working_data_df_raw) ...
            num_raw = len(working_data_df_raw)  # (rest of sampling logic is unchanged)
            if isinstance(sample_size, float) and 0 < sample_size <= 1:
                n_samples = int(num_raw * sample_size)
            elif isinstance(sample_size, int) and sample_size > 0:
                n_samples = min(sample_size, num_raw)
            else:
                raise ValueError("Invalid sample_size.")
            min_len_needed = lags + max(0, target_offset - 1) + horizon
            if n_samples < min_len_needed:
                raise ValueError(f"Sample size {n_samples} too small.")
            working_data_df_raw = working_data_df_raw.iloc[:n_samples].copy()

        variables_to_pivot = (
            ["NO2"]
            if self.only_no2
            else ["NO2", "P", "SQ", "WD", "Wvh", "dewP", "temp"]
        )
        needs_pivot = "city_name" in working_data_df_raw.columns and not any(
            f"{city}_{variables_to_pivot[0]}" in working_data_df_raw.columns
            for city in self.cities
        )

        if needs_pivot:
            try:
                self.logger.debug(
                    f"Columns before pivot: {working_data_df_raw.columns.tolist()}"
                )
                existing_pivot_vars = [
                    v for v in variables_to_pivot if v in working_data_df_raw.columns
                ]
                if not existing_pivot_vars:
                    raise ValueError("Pivot variables missing.")
                self.logger.debug(
                    f"Existing variables used for pivot: {existing_pivot_vars}"
                )

                working_data_df_wide = pd.pivot_table(
                    working_data_df_raw,
                    index="DateTime",
                    columns="city_name",  # Assumes this contains 0, 1, 2 etc.
                    values=existing_pivot_vars,
                )

                self.logger.debug(
                    f"Columns after pivot_table (MultiIndex): {working_data_df_wide.columns.tolist()}"
                )

                # Assumes the second level of the MultiIndex is the city index (0, 1, 2)
                working_data_df_wide.columns = [
                    f"city{city_idx}_{var}"
                    for var, city_idx in working_data_df_wide.columns
                ]
                self.logger.debug(
                    f"Columns after flattening: {working_data_df_wide.columns.tolist()}"
                )

                # Create mapping from city name to its assumed index based on self.cities order
                city_name_to_idx = {name: idx for idx, name in enumerate(self.cities)}

                desired_order = []
                for city_name in self.cities:  # Iterate through defined city name order
                    city_idx = city_name_to_idx[
                        city_name
                    ]  # Get the corresponding index (0, 1, or 2)
                    for (
                        var
                    ) in variables_to_pivot:  # Iterate through defined variable order
                        # Construct the column name using the city *index*
                        col_name = f"city{city_idx}_{var}"
                        if (
                            col_name in working_data_df_wide.columns
                        ):  # Check if this name exists
                            desired_order.append(col_name)

                self.logger.debug(f"Columns selected in desired_order: {desired_order}")

                if not desired_order:
                    raise ValueError(
                        "Desired order list is empty after pivoting - check column names and mapping."
                    )

                working_data_df = working_data_df_wide[desired_order]

                working_data_df = (
                    working_data_df.reset_index()
                )  # Bring DateTime back as column
            except Exception as e:
                import traceback

                traceback.print_exc()
                raise RuntimeError(f"Failed to pivot data: {e}")
        else:
            working_data_df = working_data_df_raw
            self.logger.debug(
                f"Columns when pivoting not required: {working_data_df.columns.tolist()}"
            )

        self.logger.debug(
            f"Columns in working_data_df BEFORE split mask application: {working_data_df.columns.tolist()}"
        )

        all_timestamps = pd.to_datetime(working_data_df["DateTime"])
        # Use the index of the dataframe *after* potential pivoting
        all_indices = working_data_df.index.values

        #  Step 2: Determine Train/Val/Test split MASKS (on the potentially pivoted data)
        if use_time_split:
            # Pass timestamps and indices from the potentially pivoted df
            train_mask, val_mask, test_mask = self._split_by_time(
                all_timestamps, all_indices
            )
        else:
            num_samples = len(working_data_df)
            num_train = round(num_samples * ratio[0])
            num_val = round(num_samples * ratio[1])
            train_mask = np.zeros(num_samples, dtype=bool)
            val_mask = np.zeros(num_samples, dtype=bool)
            test_mask = np.zeros(num_samples, dtype=bool)
            train_mask[:num_train] = True
            val_mask[num_train : num_train + num_val] = True
            test_mask[num_train + num_val :] = True

        if train_mask.sum() == 0:
            raise ValueError("Training split is empty.")

        #  Step 3: Fit Scalers on Training Data Subset
        # Use the potentially pivoted dataframe and the corresponding mask
        train_data_subset_df = working_data_df.loc[train_mask]
        self._fit_scalers_on_train_data(train_data_subset_df)
        self._save_scalers()  # Save train-fitted scalers

        #  Step 4: Apply Scalers to the Entire Working Data
        normalized_data_df = self._apply_scalers(working_data_df)
        #  Step 5: Prepare data array for sequence generation
        # Feature selection now correctly assumes the wide format after pivoting
        variables = (
            ["NO2"]
            if self.only_no2
            else ["NO2", "P", "SQ", "WD", "Wvh", "dewP", "temp"]
        )

        # Check for the pattern 'city{idx}_var' in the *normalized* data
        has_city_prefix_pattern = any(
            f"city{idx}_{variables[0]}" in normalized_data_df.columns
            for idx in range(len(self.cities))
        )
        self.logger.debug(
            f"Checking for features - Detected city prefix pattern ('city[idx]_var'): {has_city_prefix_pattern}"
        )

        feature_columns = []
        if has_city_prefix_pattern:
            # Build list using the 'city{idx}_var' pattern
            for city_idx in range(len(self.cities)):  # Use index 0, 1, 2
                for var in variables:  # Use the defined variable order
                    col_name = f"city{city_idx}_{var}"
                    if col_name in normalized_data_df.columns:  # Check if it exists
                        feature_columns.append(col_name)
                    else:
                        # This warning is less critical now, maybe remove later
                        self.logger.debug(
                            f"Warning: Column {col_name} missing in normalized_data_df."
                        )
        else:
            self.logger.warning(
                "Warning: City prefix pattern not found. Selecting base variables."
            )
            for var in variables:
                if var in normalized_data_df.columns:
                    feature_columns.append(var)

        if not feature_columns:
            self.logger.error(
                f"ERROR: No features selected. Columns available in normalized_data_df: {normalized_data_df.columns.tolist()}"
            )
            raise ValueError("No feature columns selected.")

        data_for_sequences = normalized_data_df[feature_columns].fillna(0).values

        #  Step 6: Generate and Split Sequence Start Indices (Unchanged)
        num_samples_sequences = data_for_sequences.shape[0]
        max_target_lag = lags + max(0, target_offset - 1) + horizon - 1
        upper_bound_start_index = num_samples_sequences - 1 - max_target_lag
        if upper_bound_start_index < 0:
            raise ValueError("Data series length too short.")
        all_sequence_start_indices = np.arange(
            0, upper_bound_start_index + 1, step=step_size
        )
        train_seq_indices = all_sequence_start_indices[
            train_mask[all_sequence_start_indices]
        ]
        val_seq_indices = all_sequence_start_indices[
            val_mask[all_sequence_start_indices]
        ]
        test_seq_indices = all_sequence_start_indices[
            test_mask[all_sequence_start_indices]
        ]
        print(
            f"Number of sequences: Train={len(train_seq_indices)}, Val={len(val_seq_indices)}, Test={len(test_seq_indices)}"
        )

        data_train = data_for_sequences
        data_val_test = data_for_sequences

        # --- Calculate lambda_max using PyG Transform ---
        if self._lambda_max is None or self.force_reload:
            if self.logger:
                self.logger.info(
                    "Calculating lambda_max for ChebConv using PyG transform..."
                )
            self._get_edges()
            self._get_edge_weights()  # Ensure graph exists
            edges_tensor = torch.tensor(self._edges, dtype=torch.long)
            weights_tensor = (
                torch.tensor(self._edge_weights, dtype=torch.float)
                if self._edge_weights is not None
                else None
            )

            try:
                # Create a temporary Data object
                temp_data = Data(
                    edge_index=edges_tensor,
                    edge_attr=weights_tensor,  # Use edge_attr for weights
                    num_nodes=self.num_nodes,
                )

                # Instantiate and apply the transform
                # is_undirected=True assumes your graph definition is symmetric
                lambda_calculator = LaplacianLambdaMax(
                    normalization="sym", is_undirected=True
                )
                transformed_data = lambda_calculator(temp_data)

                # Check if lambda_max was computed
                if (
                    not hasattr(transformed_data, "lambda_max")
                    or transformed_data.lambda_max is None
                ):
                    raise RuntimeError(
                        "LaplacianLambdaMax transform did not compute lambda_max."
                    )

                lambda_max_val = transformed_data.lambda_max
                # Clamp to ensure positivity (optional, but safe)
                lambda_max_val = max(lambda_max_val, 1e-5)
                self._lambda_max = torch.tensor(lambda_max_val, dtype=torch.float)
                if self.logger:
                    self.logger.info(
                        f"Calculated lambda_max via PyG: {self._lambda_max.item()}"
                    )

            except Exception as e:
                if self.logger:
                    self.logger.error(
                        f"Failed to calculate lambda_max using PyG transform: {e}",
                        exc_info=True,
                    )
                raise RuntimeError(f"Failed to calculate lambda_max: {e}")

        self._get_edges()
        self._get_edge_weights()
        edges_tensor = torch.tensor(self._edges, dtype=torch.long)
        edge_weights_tensor = torch.tensor(self._edge_weights, dtype=torch.float)
        if horizon is None:
            raise ValueError("Horizon must be specified.")
        train_dataset = self.IndexDataset(
            train_seq_indices,
            data_train,
            horizon,
            gpu=(allGPU != -1),
            lazy=dask_batching,
            lags=self.lags,
            target_offset=target_offset,
        )
        val_dataset = self.IndexDataset(
            val_seq_indices,
            data_val_test,
            horizon,
            gpu=(allGPU != -1),
            lazy=dask_batching,
            lags=self.lags,
            target_offset=target_offset,
        )
        test_dataset = self.IndexDataset(
            test_seq_indices,
            data_val_test,
            horizon,
            gpu=(allGPU != -1),
            lazy=dask_batching,
            lags=self.lags,
            target_offset=target_offset,
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, drop_last=True
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
        )

        #  Step 9: Cache and Return (Unchanged)
        result = (
            train_dataloader,
            val_dataloader,
            test_dataloader,
            self._edges,
            self._edge_weights,
            self._lambda_max,
        )
        if cache_path:
            self._save_cache(cache_path, result)
            self._save_scalers()  # Save train-fitted scalers

        self.force_reload = False
        self.logger.info("Dataset processing complete (Normalization: Train Set Only).")
        return (
            train_dataloader,
            val_dataloader,
            test_dataloader,
            edges_tensor,
            edge_weights_tensor,
            self._lambda_max,
        )
