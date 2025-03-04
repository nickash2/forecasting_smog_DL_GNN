from pipeline import (
    read_meteo_csv_from_data_raw,
    read_four_contaminants,
    get_metadata,
    tidy_raw_contaminant_data,
    tidy_raw_meteo_data,
    subset_sensors,
    perform_data_split,
    perform_data_split_without_train,
    calc_combined_min_max_params,
    normalise_linear,
    print_pollutant_extremes,
    export_minmax,
    concat_frames_horizontally,
)


# --------------------------
# Pipeline Stage Functions
# --------------------------


def load_raw_data(years, contaminants, city, city_station):
    """Load raw contaminant and meteorological data."""
    raw_data = {}
    meteo_data = {}
    for year in years:
        raw_data[year] = read_four_contaminants(year, contaminants)
        meteo_data[year] = read_meteo_csv_from_data_raw(year, city, city_station)
    return raw_data, meteo_data


def process_metadata(raw_data, years, contaminants):
    """Extract metadata from raw contaminant data."""
    metadata = {}
    for year in years:
        metadata[year] = {}
        for i, cont in enumerate(contaminants):
            metadata[year][cont] = get_metadata(raw_data[year][i])
    return metadata


def process_contaminants(
    raw_data, years, contaminants, subset_months, start_mon, end_mon
):
    """Process and tidy contaminant data."""
    tidy_data = {}
    for year in years:
        tidy_data[year] = {}
        for i, cont in enumerate(contaminants):
            tidy_data[year][cont] = tidy_raw_contaminant_data(
                raw_data[year][i],
                str(year),
                subset_months,
                start_mon,
                end_mon,
                fill_NaNs=True,
            )
    return tidy_data


def process_meteo(
    meteo_data, years, meteo_vars, stations, subset_months, start_mon, end_mon
):
    """Process and tidy meteorological data."""
    tidy_meteo = {}
    for year in years:
        tidy_meteo[year] = {}
        for var_name, var_info in meteo_vars.items():
            tidy_meteo[year][var_name] = tidy_raw_meteo_data(
                meteo_data[year],
                var_info["code"],
                stations,
                str(year),
                subset_months,
                start_mon,
                end_mon,
            )
    return tidy_meteo


def create_subsets(tidy_data, years, contaminants, sensors):
    """Create location-based subsets of contaminant data."""
    tidy_subset = {}
    for year in years:
        tidy_subset[year] = {}
        for cont in contaminants:
            tidy_subset[year][cont] = subset_sensors(tidy_data[year][cont], sensors)
    return tidy_subset


def split_dataset(
    tidy_subset,
    tidy_meteo,
    years,
    meteo_vars,
    contaminants,
    days_vali,
    days_test,
    days_vali_final,
    days_test_final,
):
    """Split data into train/val/test sets based on year."""
    split_data = {"train": {}, "val": {}, "test": {}}

    for year in years:
        for split in ["train", "val", "test"]:
            split_data[split][year] = {}

        # Early years: training only
        if year in [2017, 2018, 2020]:
            for cont in contaminants:
                split_data["train"][year][cont] = tidy_subset[year][cont].copy()
            for var in meteo_vars:
                split_data["train"][year][var] = tidy_meteo[year][var].copy()

        # Middle years: full split
        elif year in [2021, 2022]:
            for cont in contaminants:
                train, val, test = perform_data_split(
                    tidy_subset[year][cont], days_vali, days_test
                )
                split_data["train"][year][cont] = train
                split_data["val"][year][cont] = val
                split_data["test"][year][cont] = test
            for var in meteo_vars:
                train, val, test = perform_data_split(
                    tidy_meteo[year][var], days_vali, days_test
                )
                split_data["train"][year][var] = train
                split_data["val"][year][var] = val
                split_data["test"][year][var] = test

        # Final year: val/test only
        elif year == 2023:
            for cont in contaminants:
                val, test = perform_data_split_without_train(
                    tidy_subset[year][cont], days_vali_final, days_test_final
                )
                split_data["val"][year][cont] = val
                split_data["test"][year][cont] = test
            for var in meteo_vars:
                val, test = perform_data_split_without_train(
                    tidy_meteo[year][var], days_vali_final, days_test_final
                )
                split_data["val"][year][var] = val
                split_data["test"][year][var] = test

    return split_data


def calculate_normalization_params(
    split_data, train_years, contaminants, meteo_vars, city_name
):
    """Calculate normalization parameters from training data and export the minmax values."""
    min_max_params = {}

    for cont in contaminants:
        train_frames = [
            split_data["train"][y][cont]
            for y in train_years
            if y in split_data["train"]
        ]
        min_max_params[cont] = dict(
            zip(["min", "max"], calc_combined_min_max_params(train_frames))
        )

    for var in meteo_vars:
        train_frames = [
            split_data["train"][y][var] for y in train_years if y in split_data["train"]
        ]
        min_max_params[var] = dict(
            zip(["min", "max"], calc_combined_min_max_params(train_frames))
        )

    # Create dataframe for pollutant extremes
    print("\nPollutant min/max values:")
    pollutant_values = []
    for cont in contaminants:
        pollutant_values.extend(
            [min_max_params[cont]["min"], min_max_params[cont]["max"]]
        )

    df_minmax = print_pollutant_extremes(pollutant_values, contaminants)
    print()
    export_minmax(df_minmax, "contaminant_minmax", city_name)
    return min_max_params


def normalize_dataset(split_data, min_max_params, years, contaminants, meteo_vars):
    """Normalize dataset using pre-calculated parameters."""
    normalized_data = {"train": {}, "val": {}, "test": {}}

    for split in ["train", "val", "test"]:
        for year in years:
            normalized_data[split][year] = {}
            print(split_data[split])
            if year not in split_data[split]:
                continue
            # Normalize contaminants
            for cont in contaminants:
                if cont in split_data[split][year]:
                    normalized_data[split][year][cont] = normalise_linear(
                        split_data[split][year][cont],
                        min_max_params[cont]["min"],
                        min_max_params[cont]["max"],
                    )
            # Normalize meteo variables
            for var in meteo_vars:
                if var in split_data[split][year]:
                    normalized_data[split][year][var] = normalise_linear(
                        split_data[split][year][var],
                        min_max_params[var]["min"],
                        min_max_params[var]["max"],
                    )
    return normalized_data


def prepare_io_data(
    normalized_data, years, splits, sensors, contaminants, meteo_target
):
    """Prepare input-output data structures for model training."""
    frames = {"u": {s: {} for s in splits}, "y": {s: {} for s in splits}}

    for split in splits:
        for year in years:
            if year == 2023 and split == "train":
                continue

            # Input data (u)
            frames["u"][split][year] = []
            for cont in contaminants:
                if cont in normalized_data[split][year]:
                    frames["u"][split][year].append(
                        normalized_data[split][year][cont].loc[:, [sensors[0]]]
                    )
            for var in meteo_target:
                if var in normalized_data[split][year]:
                    frames["u"][split][year].append(normalized_data[split][year][var])

            # Target data (y)
            frames["y"][split][year] = []
            for cont in contaminants:
                if cont in normalized_data[split][year]:
                    frames["y"][split][year].append(
                        normalized_data[split][year][cont].loc[:, [sensors[1]]]
                    )
    return frames


def export_combined_data(
    frames,
    output_dir="data/data_combined",
    meteo_target: list = None,
    contaminants: list = None,
):
    """Combine and export final datasets."""
    csv_params = {"index": True, "sep": ";", "decimal": ".", "encoding": "utf-8"}

    for data_type in ["u", "y"]:
        for split in frames[data_type]:
            for year in frames[data_type][split]:
                if not frames[data_type][split][year]:
                    continue

                df = concat_frames_horizontally(
                    frames[data_type][split][year],
                    contaminants + meteo_target if data_type == "u" else contaminants,
                )
                df.to_csv(
                    f"{output_dir}/{split}_{year}_combined_{data_type}.csv",
                    **csv_params,
                )
