# src/pipeline/pipeline.py

from .stage import (
    load_raw_data,
    process_metadata,
    process_contaminants,
    process_meteo,
    create_subsets,
    split_dataset,
    calculate_normalization_params,
    normalize_dataset,
    prepare_io_data,
    export_combined_data,
)
from .tests import assert_equal_shape

import pickle

# --------------------------
# Helper Functions
# --------------------------


def _log_message(message: str, step: int, total_steps: int) -> None:
    """Log a message with step numbering."""
    print(f"({step}/{total_steps}): {message}")


def _validate_data_shapes(
    frames, allow_column_variance, allow_index_variance, context_message
):
    """Validate shapes of multiple dataframes."""
    assert_equal_shape(
        frames, allow_column_variance, allow_index_variance, context_message
    )


def _cleanup_objects(*objects) -> None:
    """Explicitly delete objects from memory."""
    for obj in objects:
        del obj


def validate_processed_data(tidy_data, years, contaminants):
    """Validate processed data for NaNs and other issues"""
    for year in years:
        for contaminant in contaminants:
            if contaminant in tidy_data[year]:
                df = tidy_data[year][contaminant]
                nan_count = df.isnull().sum()
                # if nan_count.any():
                #     # print(f"\nWarning: NaNs found in {year} {contaminant}:")
                #     print(nan_count[nan_count > 0])

                #     # Show where the NaNs are
                # nan_rows = df[df.isnull().any(axis=1)]
                # if not nan_rows.empty:
                # print("\nFirst few rows with NaNs:")
                # print(nan_rows.head())


# --------------------------
# Main Pipeline Function
# --------------------------


def execute_all_cities_pipeline(
    contaminants: list = ["O3", "NO2"],
    years: list = [2017, 2018, 2020, 2021, 2022, 2023],
    LOG: bool = True,
    meteo_target: list = ["temp", "dewP", "WD", "Wvh", "P", "SQ"],
    **kwargs,
) -> None:
    """Execute complete pipeline across all cities simultaneously."""
    execute_pipeline(
        contaminants=contaminants,
        locations=None,  # Not needed when process_all is True
        city_name=None,  # Not needed when process_all is True
        years=years,
        LOG=LOG,
        meteo_target=meteo_target,
        process_all=True,  # Set the new parameter
        **kwargs,
    )


def execute_pipeline(
    contaminants: list = ["PM25", "PM10", "O3", "NO2"],
    locations: list = ["NL10636", "NL10641", 260],
    city_name: str = "Utrecht",
    years: list = [2017, 2018],
    LOG: bool = True,
    meteo_target: list = ["temp", "dewP", "WD", "Wvh", "P", "SQ"],
    SUBSET_MONTHS: bool = True,
    START_MON: str = "08",
    END_MON: str = "12",
    days_vali: int = 21,
    days_test: int = 21,
    days_vali_final_yrs: int = 63,
    days_test_final_yrs: int = 63,
    process_all: bool = False,  # New parameter
) -> None:
    """
    Main pipeline execution function.
    """
    TOTAL_STEPS = 9
    print("-----------------------------------")
    print("Executing the pipeline\n")

    sensors = locations[:2]
    stations = locations[2]
    stations_to_city = {stations: city_name}
    output_dir = city_name.lower()

    # Rest of initialization
    meteo_vars = {
        "temp": {"code": "T"},
        "dewP": {"code": "TD"},
        "WD": {"code": "DD"},
        "Wvh": {"code": "FH"},
        "Wmax": {"code": "FX"},
        "preT": {"code": "DR"},
        "P": {"code": "P"},
        "preS": {"code": "RH"},
        "SQ": {"code": "SQ"},
        "Q": {"code": "Q"},
    }
    train_years = [2017, 2018, 2020, 2021, 2022]

    # Step 1: Load raw data
    raw_data = {}
    meteo_data = {}

    raw_data, meteo_data = load_raw_data(years, contaminants, city_name, stations)

    if LOG:
        _log_message("Data read successfully", 1, TOTAL_STEPS)

    # Step 2: Process contaminants
    metadata = process_metadata(raw_data, years, contaminants)
    tidy_data = process_contaminants(
        raw_data, years, contaminants, SUBSET_MONTHS, START_MON, END_MON
    )
    if LOG:
        validate_processed_data(tidy_data, years, contaminants)
    _cleanup_objects(raw_data)
    _cleanup_objects(meteo_data)

    if LOG:
        _validate_data_shapes(
            [v for y in years for v in tidy_data[y].values()],
            True,
            False,
            "Tidying of pollutant data",
        )
        _log_message("Pollutant data tidied successfully", 2, TOTAL_STEPS)

    # Step 3: Process meteorological data
    tidy_meteo = process_meteo(
        meteo_data, years, meteo_vars, stations, SUBSET_MONTHS, START_MON, END_MON
    )
    if LOG:
        _validate_data_shapes(
            [v for y in years for v in tidy_meteo[y].values()],
            True,
            True,
            "Meteorological data",
        )
        _log_message("Meteorological data tidied successfully", 3, TOTAL_STEPS)

    # Step 4: Create subsets
    tidy_subset = create_subsets(tidy_data, years, contaminants, sensors)
    if LOG:
        _validate_data_shapes(
            [tidy_subset[y][c] for y in years for c in contaminants],
            True,
            True,
            "Subset data",
        )
        _log_message("Subsetting successful", 4, TOTAL_STEPS)
    _cleanup_objects(tidy_data)

    # Step 5: Split data
    split_data = split_dataset(
        tidy_subset,
        tidy_meteo,
        years,
        meteo_vars,
        contaminants,
        days_vali,
        days_test,
        days_vali_final_yrs,
        days_test_final_yrs,
    )
    if LOG:
        _log_message("Data splitting successful", 5, TOTAL_STEPS)

    # Step 6: Normalize data
    min_max_params = calculate_normalization_params(
        split_data, train_years, contaminants, meteo_vars, city_name.lower()
    )

    if process_all:
        # Save train frames for this city as a pickle file
        with open(
            f"data/data_combined/all/split_dataframes_{city_name.lower()}.pkl",
            "wb",
        ) as f:
            pickle.dump(split_data, f)
        if LOG:
            _log_message("Train frames saved successfully", 6, TOTAL_STEPS)

    # pre preprocessing step to calculate minmax params of all cities by reading all csvs and finding the norm params
    normalized_data = normalize_dataset(
        split_data, min_max_params, years, contaminants, meteo_vars
    )
    if LOG:
        _log_message("Normalization successful", 7, TOTAL_STEPS)

    # Step 7: Prepare IO data
    io_frames = prepare_io_data(
        normalized_data,
        years,
        ["train", "val", "test"],
        sensors,
        contaminants,
        meteo_target,
    )
    if LOG:
        _log_message("IO preparation successful", 8, TOTAL_STEPS)

    # Step 8: Export data
    export_combined_data(
        io_frames,
        output_dir=f"data/data_combined/{city_name.lower()}",
        contaminants=contaminants,
        meteo_target=meteo_target,
    )
    if LOG:
        _log_message("Data exported successfully", 9, TOTAL_STEPS)

    print("\nPipeline execution completed successfully")
