# src/pipeline/pipeline.py

# This file contains one function that excecutes the entire pipeline.
# It is similar to the preprocess.ipynb notebook, but this one can be
# ran easily from the command line and main.py. It uses just functions
# from the pipeline modules, which are imported below.
#
# For modifications, e.g. when changing the dimensions of time, space,
# and/or variables, using the notebook is probably easy for experimentation,
# and (copies of) this file would be good for "production" runs. So, to
# change the set-up, I'd recommend making a copy and going through the
# the code one-by-one, testing it, and then moving on to the next part

from pipeline import read_meteo_csv_from_data_raw
from pipeline import read_four_contaminants
from pipeline import get_metadata
from pipeline import tidy_raw_contaminant_data
from pipeline import tidy_raw_meteo_data
from pipeline import print_aggegrated_sensor_metrics
from pipeline import subset_sensors
from pipeline import perform_data_split
from pipeline import perform_data_split_without_train
from pipeline import print_split_ratios
from pipeline import calc_combined_min_max_params
from pipeline import normalise_linear
from pipeline import print_pollutant_extremes
from pipeline import export_minmax
from pipeline import plot_distributions_KDE
from pipeline import concat_frames_horizontally
from pipeline import delete_timezone_from_index
from pipeline import assert_equal_shape
from pipeline import assert_equal_index
from pipeline import assert_no_NaNs
from pipeline import assert_range
import pandas as pd
import numpy as np


def execute_pipeline(
    contaminants: list = ["PM25", "PM10", "O3", "NO2"],
    locations: list = ["NL10636", "NL10641", 260],
    city_name: str = "Utrecht",
    years: list = [2017, 2018],
    LOG: bool = True,
    SUBSET_MONTHS: bool = True,
    START_MON: str = "08",
    END_MON: str = "12",
    days_vali: int = 21,
    days_test: int = 21,
    days_vali_final_yrs: int = 63,
    days_test_final_yrs: int = 63,
) -> None:
    """Main pipeline execution function."""
    print("-----------------------------------")
    print("Executing the pipeline\n")

    TUINDORP = locations[0]  # starting location for contamination data
    BREUKELEN = locations[1]  # 'goal' location for contamination data
    stations = locations[2]

    # First step, load in the raw data
    raw_data = {}
    meteo_data = {}

    for year in years:
        raw_data[year] = read_four_contaminants(year, contaminants)
        meteo_data[year] = read_meteo_csv_from_data_raw(year)

    if LOG:
        print("(1/8): Data read successfully")

    # First, tidy the contamination data
    metadata = {}

    # Get metadata for each contaminant and year
    for year in years:
        metadata[year] = {}
        for i, contaminant in enumerate(contaminants):
            metadata[year][contaminant] = get_metadata(raw_data[year][i])

    if LOG:
        print(f"Got metadata for years {years} and contaminants {contaminants}")

    # Create tidied data structures
    tidy_data = {}

    # Tidy data for each contaminant and year
    for year in years:
        tidy_data[year] = {}
        for i, contaminant in enumerate(contaminants):
            tidy_data[year][contaminant] = tidy_raw_contaminant_data(
                raw_data[year][i], str(year), SUBSET_MONTHS, START_MON, END_MON
            )

    if LOG:
        # Verify shape of tidied data
        tidy_frames = []
        for year in years:
            tidy_frames.extend(list(tidy_data[year].values()))

        assert_equal_shape(tidy_frames, True, False, "Tidying of pollutant data")
        print("(2/8): Pollutant data tidied successfully")

    # Second, tidy the meteorological data
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

    tidy_meteo = {}

    # Process each year's meteorological data
    for year in years:
        tidy_meteo[year] = {}
        for var_name, var_info in meteo_vars.items():
            tidy_meteo[year][var_name] = tidy_raw_meteo_data(
                meteo_data[year],
                var_info["code"],
                stations,
                str(year),
                SUBSET_MONTHS,
                START_MON,
                END_MON,
            )

    if LOG:
        # Verify shape of tidied meteorological data
        tidy_meteo_frames = []
        for year in years:
            tidy_meteo_frames.extend(list(tidy_meteo[year].values()))

        assert_equal_shape(
            tidy_meteo_frames, True, True, "Tidying of meteorological data"
        )

        # Verify no NaNs in meteorological data
        assert_no_NaNs(tidy_meteo_frames, "Tidying of meteorological data")
        print("(3/8): Meteorological data tidied successfully")

    # Clean up raw meteorological data
    del meteo_data
    del raw_data

    # Here, we'll select the locations we want to use. The
    # I/O-task can be either 0-dimensional, or 1-dimensional.
    sensors_1D = [TUINDORP, BREUKELEN]

    # Create dictionary for subset data
    tidy_subset = {}

    # Subset sensors for each year and contaminant
    for year in years:
        tidy_subset[year] = {}
        for contaminant in contaminants:
            tidy_subset[year][contaminant] = subset_sensors(
                tidy_data[year][contaminant], sensors_1D
            )

    if LOG:
        # Verify shape of subset data
        subset_frames = []
        for year in years:
            print(f"Year {year}, {contaminant}: {tidy_subset[year][contaminant].shape}")
            subset_frames.append(tidy_subset[year][contaminant])
        # print(subset_frames)
        assert_equal_shape(
            subset_frames, True, True, "Location-wise subsetting of pollutant data"
        )

        # Verify no NaNs in subset data
        assert_no_NaNs(subset_frames, "Location-wise subsetting of pollutant data")
        print("(4/8): Location-wise subsetting of pollutant data successful")

    # Clean up original tidy data
    del tidy_data

    print("Printing some basic statistics for the pollutants:")
    print(f"(Sensor {TUINDORP} is TUINDORP)\n")

    # Aggregate sensor metrics
    for contaminant in contaminants:
        # Collect all frames for this contaminant across years
        frames = []
        for year in years:
            if year in tidy_subset:
                frames.append(tidy_subset[year][contaminant])

        # Use metadata from the first available year for this contaminant
        first_year = min(years)
        meta = metadata[first_year][contaminant]

        # Print metrics for this contaminant
        print(f"\n{contaminant} statistics:")
        print_aggegrated_sensor_metrics(frames, TUINDORP, meta)

    # Initialize dictionary to store split data
    split_data = {"train": {}, "val": {}, "test": {}}

    # Process each year
    for year in years:
        # Initialize year in split_data if needed
        for split in ["train", "val", "test"]:
            split_data[split][year] = {}

        # Handle early years (2017, 2018, 2020) - only training data
        if year in [2017, 2018, 2020]:
            # Process contaminants
            for cont in contaminants:
                if year in tidy_subset and cont in tidy_subset[year]:
                    split_data["train"][year][cont] = tidy_subset[year][cont].copy()

            # Process meteorological variables
            for var in meteo_vars:
                if year in tidy_meteo and var in tidy_meteo[year]:
                    split_data["train"][year][var] = tidy_meteo[year][var].copy()

        # Handle middle years (2021, 2022) - split into train/val/test
        elif year in [2021, 2022]:
            # Process contaminants
            for cont in contaminants:
                if year in tidy_subset and cont in tidy_subset[year]:
                    train, val, test = perform_data_split(
                        tidy_subset[year][cont], days_vali, days_test
                    )
                    split_data["train"][year][cont] = train
                    split_data["val"][year][cont] = val
                    split_data["test"][year][cont] = test

            # Process meteorological variables
            for var in meteo_vars:
                if year in tidy_meteo and var in tidy_meteo[year]:
                    train, val, test = perform_data_split(
                        tidy_meteo[year][var], days_vali, days_test
                    )
                    split_data["train"][year][var] = train
                    split_data["val"][year][var] = val
                    split_data["test"][year][var] = test

        # Handle final year (2023) - only val/test data
        elif year == 2023:
            # Process contaminants
            for cont in contaminants:
                if year in tidy_subset and cont in tidy_subset[year]:
                    val, test = perform_data_split_without_train(
                        tidy_subset[year][cont],
                        days_vali_final_yrs,
                        days_test_final_yrs,
                    )
                    split_data["val"][year][cont] = val
                    split_data["test"][year][cont] = test

            # Process meteorological variables
            for var in meteo_vars:
                if year in tidy_meteo and var in tidy_meteo[year]:
                    val, test = perform_data_split_without_train(
                        tidy_meteo[year][var], days_vali_final_yrs, days_test_final_yrs
                    )
                    split_data["val"][year][var] = val
                    split_data["test"][year][var] = test

    # Create individual variables for backward compatibility
    for split in ["train", "val", "test"]:
        for year in years:
            if year in split_data[split]:
                for var in contaminants + list(meteo_vars.keys()):
                    if var in split_data[split][year]:
                        suffix = "_1D" if var in contaminants else ""
                        var_name = f"df_{var}_{year}_{split}{suffix}"
                        globals()[var_name] = split_data[split][year][var]

    if LOG:
        # First, check for equal shape of pollutant data of unsplitted years
        assert_equal_shape(
            [
                df_PM25_2017_train_1D,
                df_PM10_2017_train_1D,
                df_NO2_2017_train_1D,
                df_O3_2017_train_1D,
                df_PM25_2018_train_1D,
                df_PM10_2018_train_1D,
                df_NO2_2018_train_1D,
                df_O3_2018_train_1D,
                df_PM25_2020_train_1D,
                df_PM10_2020_train_1D,
                df_NO2_2020_train_1D,
                df_O3_2020_train_1D,
            ],
            True,
            True,
            "Split of pollutant train set for 2017, 2018 and 2020",
        )
        # Second, check for equal shape of meteorological data of unsplitted years
        assert_equal_shape(
            [
                df_temp_2017_train,
                df_dewP_2017_train,
                df_WD_2017_train,
                df_Wvh_2017_train,
                df_P_2017_train,
                df_SQ_2017_train,
                df_temp_2018_train,
                df_dewP_2018_train,
                df_WD_2018_train,
                df_Wvh_2018_train,
                df_P_2018_train,
                df_SQ_2018_train,
                df_temp_2020_train,
                df_dewP_2020_train,
                df_WD_2020_train,
                df_Wvh_2020_train,
                df_P_2020_train,
                df_SQ_2020_train,
            ],
            True,
            True,
            "Split of meteorological train set for 2017, 2018 and 2020",
        )
        # Third, check for equal row number of training set in 2021 and 2022
        assert_equal_shape(
            [
                df_PM25_2021_train_1D,
                df_PM10_2021_train_1D,
                df_NO2_2021_train_1D,
                df_O3_2021_train_1D,
                df_temp_2021_train,
                df_dewP_2021_train,
                df_WD_2021_train,
                df_Wvh_2021_train,
                df_P_2021_train,
                df_SQ_2021_train,
                df_PM25_2022_train_1D,
                df_PM10_2022_train_1D,
                df_NO2_2022_train_1D,
                df_O3_2022_train_1D,
                df_temp_2022_train,
                df_dewP_2022_train,
                df_WD_2022_train,
                df_Wvh_2022_train,
                df_P_2022_train,
                df_SQ_2022_train,
                # They should be of the same length, meaning they're split over the
                # same timeframe. Columns can vary, because meteorological data is
                # not used for the location where the predictions are made, i.e. Breukelen
            ],
            True,
            False,
            "Split of training data for 2021 and 2022",
        )
        # Fourth, check for equal row number of validation set in 2021 and 2022
        assert_equal_shape(
            [
                df_PM25_2021_val_1D,
                df_PM10_2021_val_1D,
                df_NO2_2021_val_1D,
                df_O3_2021_val_1D,
                df_temp_2021_val,
                df_dewP_2021_val,
                df_WD_2021_val,
                df_Wvh_2021_val,
                df_P_2021_val,
                df_SQ_2021_val,
                df_PM25_2022_val_1D,
                df_PM10_2022_val_1D,
                df_NO2_2022_val_1D,
                df_O3_2022_val_1D,
                df_temp_2022_val,
                df_dewP_2022_val,
                df_WD_2022_val,
                df_Wvh_2022_val,
                df_P_2022_val,
                df_SQ_2022_val,
            ],
            True,
            False,
            "Split of validation data for 2021 and 2022",
        )
        # Fifth, check for equal row number of test set in 2021 and 2022
        assert_equal_shape(
            [
                df_PM25_2021_test_1D,
                df_PM10_2021_test_1D,
                df_NO2_2021_test_1D,
                df_O3_2021_test_1D,
                df_temp_2021_test,
                df_dewP_2021_test,
                df_WD_2021_test,
                df_Wvh_2021_test,
                df_P_2021_test,
                df_SQ_2021_test,
                df_PM25_2022_test_1D,
                df_PM10_2022_test_1D,
                df_NO2_2022_test_1D,
                df_O3_2022_test_1D,
                df_temp_2022_test,
                df_dewP_2022_test,
                df_WD_2022_test,
                df_Wvh_2022_test,
                df_P_2022_test,
                df_SQ_2022_test,
            ],
            True,
            False,
            "Split of test data for 2021 and 2022",
        )
        # Sixth, check for equal row number of validation set in 2023
        assert_equal_shape(
            [
                df_PM25_2023_val_1D,
                df_PM10_2023_val_1D,
                df_NO2_2023_val_1D,
                df_O3_2023_val_1D,
                df_temp_2023_val,
                df_dewP_2023_val,
                df_WD_2023_val,
                df_Wvh_2023_val,
                df_P_2023_val,
                df_SQ_2023_val,
            ],
            True,
            False,
            "Split of validation data for 2023",
        )
        # Seventh, check for equal row number of test set in 2023
        assert_equal_shape(
            [
                df_PM25_2023_test_1D,
                df_PM10_2023_test_1D,
                df_NO2_2023_test_1D,
                df_O3_2023_test_1D,
                df_temp_2023_test,
                df_dewP_2023_test,
                df_WD_2023_test,
                df_Wvh_2023_test,
                df_P_2023_test,
                df_SQ_2023_test,
            ],
            True,
            False,
            "Split of test data for 2023",
        )
        print("(5/8): Train-validation-test split successful")

    print_split_ratios(
        [
            df_PM25_2017_train_1D,
            df_PM25_2018_train_1D,
            df_PM25_2020_train_1D,
            df_PM25_2021_train_1D,
            df_PM25_2022_train_1D,
        ],
        [df_PM25_2021_val_1D, df_PM25_2022_val_1D, df_PM25_2023_val_1D],
        [df_PM25_2021_test_1D, df_PM25_2022_test_1D, df_PM25_2023_test_1D],
        "the",
    )  # Could also print the pollutants here or any other string

    # Initialize dictionary to store min/max parameters
    min_max_params = {}

    # Training years to use for normalization
    train_years = [2017, 2018, 2020, 2021, 2022]

    # Calculate min/max for contaminants
    for cont in contaminants:
        train_frames = [
            split_data["train"][year][cont]
            for year in train_years
            if year in split_data["train"] and cont in split_data["train"][year]
        ]
        min_val, max_val = calc_combined_min_max_params(train_frames)
        min_max_params[cont] = {"min": min_val, "max": max_val}

    # Calculate min/max for meteorological variables
    for var in meteo_vars:
        train_frames = [
            split_data["train"][year][var]
            for year in train_years
            if year in split_data["train"] and var in split_data["train"][year]
        ]
        min_val, max_val = calc_combined_min_max_params(train_frames)
        min_max_params[var] = {"min": min_val, "max": max_val}

    # Create dataframe for pollutant extremes
    if LOG:
        print("\nPollutant min/max values:")
        pollutant_values = []
        for cont in ["NO2", "O3", "PM10", "PM25"]:
            pollutant_values.extend(
                [min_max_params[cont]["min"], min_max_params[cont]["max"]]
            )

        df_minmax = print_pollutant_extremes(pollutant_values)
        print()
        export_minmax(df_minmax, "contaminant_minmax")

    # Create variables for backward compatibility
    for var_name, params in min_max_params.items():
        globals()[f"{var_name}_min_train"] = params["min"]
        globals()[f"{var_name}_max_train"] = params["max"]

    # Initialize dictionary to store normalized data
    normalized_data = {"train": {}, "val": {}, "test": {}}

    # Normalize all data using the calculated parameters
    for split_type in ["train", "val", "test"]:
        for year in years:
            normalized_data[split_type][year] = {}
            if year in split_data[split_type]:
                # Normalize contaminants
                for cont in contaminants:
                    if cont in split_data[split_type][year]:
                        normalized_data[split_type][year][cont] = normalise_linear(
                            split_data[split_type][year][cont],
                            min_max_params[cont]["min"],
                            min_max_params[cont]["max"],
                        )

                # Normalize meteorological variables
                for var in meteo_vars:
                    if var in split_data[split_type][year]:
                        normalized_data[split_type][year][var] = normalise_linear(
                            split_data[split_type][year][var],
                            min_max_params[var]["min"],
                            min_max_params[var]["max"],
                        )

    # Create individual variables for backward compatibility
    for split_type in ["train", "val", "test"]:
        for year in years:
            if year in normalized_data[split_type]:
                for var in contaminants + list(meteo_vars.keys()):
                    if var in normalized_data[split_type][year]:
                        suffix = "_1D" if var in contaminants else ""
                        var_name = f"df_{var}_{year}_{split_type}_norm{suffix}"
                        globals()[var_name] = normalized_data[split_type][year][var]
    if LOG:
        # Assert range only for training frames, validation and test
        # frames can, very theoretically, have unlimited values
        assert_range(
            [
                df_NO2_2017_train_norm_1D,
                df_NO2_2018_train_norm_1D,
                df_NO2_2020_train_norm_1D,
                df_NO2_2021_train_norm_1D,
                df_NO2_2022_train_norm_1D,
            ],
            0,
            1,
            "Normalisation of NO2 data",
        )
        assert_range(
            [
                df_O3_2017_train_norm_1D,
                df_O3_2018_train_norm_1D,
                df_O3_2020_train_norm_1D,
                df_O3_2021_train_norm_1D,
                df_O3_2022_train_norm_1D,
            ],
            0,
            1,
            "Normalisation of O3 data",
        )
        assert_range(
            [
                df_PM10_2017_train_norm_1D,
                df_PM10_2018_train_norm_1D,
                df_PM10_2020_train_norm_1D,
                df_PM10_2021_train_norm_1D,
                df_PM10_2022_train_norm_1D,
            ],
            0,
            1,
            "Normalisation of PM10 data",
        )
        assert_range(
            [
                df_PM25_2017_train_norm_1D,
                df_PM25_2018_train_norm_1D,
                df_PM25_2020_train_norm_1D,
                df_PM25_2021_train_norm_1D,
                df_PM25_2022_train_norm_1D,
            ],
            0,
            1,
            "Normalisation of PM25 data",
        )
        assert_range(
            [
                df_temp_2017_train_norm,
                df_temp_2018_train_norm,
                df_temp_2020_train_norm,
                df_temp_2021_train_norm,
                df_temp_2022_train_norm,
            ],
            0,
            1,
            "Normalisation of temperature data",
        )
        assert_range(
            [
                df_dewP_2017_train_norm,
                df_dewP_2018_train_norm,
                df_dewP_2020_train_norm,
                df_dewP_2021_train_norm,
                df_dewP_2022_train_norm,
            ],
            0,
            1,
            "Normalisation of dew point data",
        )
        assert_range(
            [
                df_WD_2017_train_norm,
                df_WD_2018_train_norm,
                df_WD_2020_train_norm,
                df_WD_2021_train_norm,
                df_WD_2022_train_norm,
            ],
            0,
            1,
            "Normalisation of wind direction data",
        )
        assert_range(
            [
                df_Wvh_2017_train_norm,
                df_Wvh_2018_train_norm,
                df_Wvh_2020_train_norm,
                df_Wvh_2021_train_norm,
                df_Wvh_2022_train_norm,
            ],
            0,
            1,
            "Normalisation of wind velocity data",
        )
        assert_range(
            [
                df_P_2017_train_norm,
                df_P_2018_train_norm,
                df_P_2020_train_norm,
                df_P_2021_train_norm,
                df_P_2022_train_norm,
            ],
            0,
            1,
            "Normalisation of pressure data",
        )
        assert_range(
            [
                df_SQ_2017_train_norm,
                df_SQ_2018_train_norm,
                df_SQ_2020_train_norm,
                df_SQ_2021_train_norm,
                df_SQ_2022_train_norm,
            ],
            0,
            1,
            "Normalisation of solar radiation data",
        )
        print("(6/8): Normalisation successful")

    # Define variables
    keys = ["PM25", "PM10", "O3", "NO2", "temp", "dewP", "WD", "Wvh", "P", "SQ"]
    splits = ["train", "val", "test"]

    # Initialize dictionaries to store frame lists
    frames_u = {split: {} for split in splits}
    frames_y = {split: {} for split in splits}

    # Create input dataframes for Utrecht (u) and Breukelen (y)
    for split in splits:
        for year in years:
            # Skip train split for 2023
            if year == 2023 and split == "train":
                continue

            # Create frame list for Utrecht (input data)
            if split in normalized_data and year in normalized_data[split]:
                frames_u[split][year] = []

                # Add pollutant data for Tuindorp
                for cont in ["PM25", "PM10", "O3", "NO2"]:
                    if cont in normalized_data[split][year]:
                        frames_u[split][year].append(
                            normalized_data[split][year][cont].loc[:, [TUINDORP]]
                        )

                # Add meteorological data
                for var in ["temp", "dewP", "WD", "Wvh", "P", "SQ"]:
                    if var in normalized_data[split][year]:
                        frames_u[split][year].append(normalized_data[split][year][var])

            # Create frame list for Breukelen (output data)
            if split in normalized_data and year in normalized_data[split]:
                frames_y[split][year] = []

                # Add only pollutant data for Breukelen
                for cont in ["PM25", "PM10", "O3", "NO2"]:
                    if cont in normalized_data[split][year]:
                        frames_y[split][year].append(
                            normalized_data[split][year][cont].loc[:, [BREUKELEN]]
                        )

    # Create individual variables for backward compatibility
    for split in splits:
        for year in years:
            if year == 2023 and split == "train":
                continue
            if year in frames_u[split]:
                globals()[f"frames_{split}_{year}_1D_u"] = frames_u[split][year]
            if year in frames_y[split]:
                globals()[f"frames_{split}_{year}_1D_y"] = frames_y[split][year]

    input_keys = ["PM25", "PM10", "O3", "NO2", "temp", "dewP", "WD", "Wvh", "p", "SQ"]
    target_keys = ["PM25", "PM10", "O3", "NO2"]

    # Define the years and splits to process
    years = [2017, 2018, 2020, 2021, 2022, 2023]
    splits = ["train", "val", "test"]

    # Create dictionaries to store the horizontal concatenated frames
    horizontal_u = {}
    horizontal_y = {}

    # Process each year and split combination
    for year in years:
        for split in splits:
            # Skip train split for 2023
            if year == 2023 and split == "train":
                continue

            # Skip val and test splits for 2017, 2018, and 2020
            if year in [2017, 2018, 2020] and split in ["val", "test"]:
                continue

            # Create the variable names
            frames_var_name = f"frames_{split}_{year}_1D"

            # Process input (u) frames
            if globals().get(f"{frames_var_name}_u") is not None:
                df_name = f"df_{split}_{year}_horizontal_u"
                horizontal_u[df_name] = concat_frames_horizontally(
                    globals()[f"{frames_var_name}_u"], input_keys
                )
                globals()[df_name] = horizontal_u[df_name]

            # Process target (y) frames
            if globals().get(f"{frames_var_name}_y") is not None:
                df_name = f"df_{split}_{year}_horizontal_y"
                horizontal_y[df_name] = concat_frames_horizontally(
                    globals()[f"{frames_var_name}_y"], target_keys
                )
                globals()[df_name] = horizontal_y[df_name]

    # At last, a final check before exporting

    if LOG:
        # First, check if u-dataframes of unsplitted years have same shape
        assert_equal_shape(
            [
                df_train_2017_horizontal_u,
                df_train_2018_horizontal_u,
                df_train_2020_horizontal_u,
            ],
            True,
            True,
            "Shape of u-dataframes of 2017, 2018 and 2020",
        )
        # Second, check if y-dataframes of unsplitted years have same shape
        assert_equal_shape(
            [
                df_train_2017_horizontal_y,
                df_train_2018_horizontal_y,
                df_train_2020_horizontal_y,
            ],
            True,
            True,
            "Shape of y-dataframes of 2017, 2018 and 2020",
        )
        # Third, check if validation/test u-dataframes of splitted years
        # have the same shape
        assert_equal_shape(
            [
                df_val_2021_horizontal_u,
                df_test_2021_horizontal_u,
                df_val_2022_horizontal_u,
                df_test_2022_horizontal_u,
            ],
            True,
            True,
            "Shape of u-dataframes of 2021 and 2022",
        )
        # Fourth, check if validation/test y-dataframes of splitted years
        # have the same shape
        assert_equal_shape(
            [
                df_val_2021_horizontal_y,
                df_test_2021_horizontal_y,
                df_val_2022_horizontal_y,
                df_test_2022_horizontal_y,
            ],
            True,
            True,
            "Shape of y-dataframes of 2021 and 2022",
        )
        # Fifth, check if 2023 dataframes have the same shape
        assert_equal_shape(
            [
                df_val_2023_horizontal_u,
                df_test_2023_horizontal_u,
                df_val_2023_horizontal_y,
                df_test_2023_horizontal_y,
            ],
            True,
            False,
            "Shape of 2023 dataframes",
        )

        print("(7/8): All data concatenations successful")

    csv_params = {"index": True, "sep": ";", "decimal": ".", "encoding": "utf-8"}

    # Export the dataframes

    # Save both input (u) and target (y) dataframes
    for data_type in ["u", "y"]:
        for year in years:
            for split in splits:
                # Skip train split for 2023
                if year == 2023 and split == "train":
                    continue

                # Skip val and test splits for 2017, 2018, and 2020
                if year in [2017, 2018, 2020] and split in ["val", "test"]:
                    continue

                # Create variable and filename
                df_name = f"df_{split}_{year}_horizontal_{data_type}"
                filename = f"data/data_combined/{split}_{year}_combined_{data_type}.csv"

                # Export dataframe if it exists in globals
                if df_name in globals():
                    globals()[df_name].to_csv(filename, **csv_params)
        print("(8/8): All data done exporting")
