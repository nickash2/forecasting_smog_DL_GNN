# src/process_global
import pandas as pd
import pickle
from typing import Dict, List

from pipeline.stage import (
    normalize_dataset,
    prepare_io_data,
    export_combined_data,
)
from pipeline.extract import read_all_city_train_frames
import pandas as pd

import sys

sys.path.append("../")
from config import ALL_DIR


def find_global_min_max(city_train_frames: List[pd.DataFrame]) -> dict:
    """Find global min/max values for each pollutant across all cities and years"""
    global_min_max = {}

    for city_dict in city_train_frames:
        for year, pollutant_dict in city_dict["train"].items():
            for pollutant, df in pollutant_dict.items():
                if pollutant not in global_min_max:
                    global_min_max[pollutant] = {
                        "min": float("inf"),
                        "max": float("-inf"),
                    }

                curr_min = df.min().min()
                curr_max = df.max().max()

                if curr_min < global_min_max[pollutant]["min"]:
                    global_min_max[pollutant]["min"] = curr_min

                if curr_max > global_min_max[pollutant]["max"]:
                    global_min_max[pollutant]["max"] = curr_max

    return global_min_max


def normalize_city_frames(
    city_train_frames: List[pd.DataFrame],
    global_min_max: dict,
    years: List[int],
    contaminants: List[str],
    meteo_vars: Dict[str, Dict[str, str]],
) -> List[pd.DataFrame]:
    """Normalize all city frames using global min/max values"""
    normalized_frames = []
    for city_frame in city_train_frames:
        norm = normalize_dataset(
            city_frame,
            global_min_max,
            years=years,
            contaminants=contaminants,
            meteo_vars=meteo_vars,
        )
        normalized_frames.append(norm)
    return normalized_frames


def prepare_all_io_data(
    normalized_frames: List[pd.DataFrame],
    years: List[int],
    sensors: List[int],
    contaminants: List[str],
    meteo_vars: Dict[str, Dict[str, str]],
) -> List[pd.DataFrame]:
    """Prepare input/output data for all cities"""
    io_frames = []
    for idx, io_frame in enumerate(normalized_frames):
        io_frames.append(
            prepare_io_data(
                io_frame,
                years,
                ["train", "val", "test"],
                sensors[idx],
                contaminants,
                meteo_vars,
            )
        )
    return io_frames


def export_all_data(
    io_frames: List[pd.DataFrame],
    output_dir: str,
    contaminants: List[str],
    meteo_target: List[str],
    cities: List[str],
):
    """Export all prepared data frames"""

    for frame, city in zip(io_frames, cities):
        export_combined_data(
            frame,
            output_dir=f"{output_dir}/{city}",
            contaminants=contaminants,
            meteo_target=meteo_target,
        )


def rescale(sensors: List[str], years: List[int], cities: List[str]):
    """
    Rescales the data for the given sensors, from their previous normalization (city-specific)
    to the global normalization (city-wide).
    :param sensors: List of sensors for each city
    :param years: List of years to consider
    :param cities: List of cities to consider in the same order as sensors

    """

    print("ALL_DIR for rescaling", ALL_DIR)
    # years = [2017, 2018, 2020, 2021, 2022, 2023]
    contaminants = ["NO2", "O3"]
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
    # sensors = [
    #     ["NL01485", "NL01494"],  # Rotterdam
    #     ["NL10636", "NL10641"],  # Utrecht
    #     ["NL49003", "NL49012"],  # Amsterdam
    # ]
    meteo_target = ["temp", "dewP", "WD", "Wvh", "P", "SQ"]

    cities = ["rotterdam", "utrecht", "amsterdam"]

    # Process pipeline
    city_train_frames = read_all_city_train_frames(ALL_DIR / "pkls")

    print("Processed city train frames (1/5)")

    global_min_max = find_global_min_max(city_train_frames)

    print(global_min_max)

    print("Found global min/max values (2/5)")

    normalized_frames = normalize_city_frames(
        city_train_frames, global_min_max, years, contaminants, meteo_vars
    )

    print("Re-Normalized all city train frames (3/5)")

    io_frames = prepare_all_io_data(
        normalized_frames, years, sensors, contaminants, meteo_vars
    )

    print("Prepared all input/output data (4/5)")

    export_all_data(io_frames, ALL_DIR, contaminants, meteo_target, cities)

    print("Exported all rescaled data ready for GNN (5/5)")
