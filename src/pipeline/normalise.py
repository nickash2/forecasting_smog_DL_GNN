# src/pipeline/normalise.py

# (Helper) functions for (inverse) linear scaling used for normalisation

from typing import List
import numpy as np
import pandas as pd


def get_df_minimum(df: pd.DataFrame) -> float:
    """
    Returns minimum of entire dataframe, thus
    the minimum of the minimums of all columns

    :param df: dataframe
    :return: minimum of entire dataframe
    """
    return np.min(df.min())


def get_df_maximum(df: pd.DataFrame) -> float:
    """
    Returns maximum of entire dataframe, thus
    the maximum of the maximums of all columns

    :param df: dataframe
    :return: maximum of entire dataframe
    """
    return np.max(df.max())


def calc_combined_min_max_params(dfs: list) -> tuple:
    """ "
    Returns min and max of two dataframes combined

    :param dfs: list of dataframes
    :return: tuple of min and max
    """
    min = np.min([get_df_minimum(df) for df in dfs])
    max = np.max([get_df_maximum(df) for df in dfs])
    return min, max


def normalise_linear(df: pd.DataFrame, min: float, max: float) -> pd.DataFrame:
    """
    Performs linear scaling (minmax) on dataframe:

    x' = (x - x_min) / (x_max - x_min),

    where x is the original value, and x_min and x_max
    are the minimum and maximum values of the associated
    training data, respectively

    :param df: dataframe
    :param min: minimum value of training data
    :param max: maximum value of training data
    :return: normalised dataframe
    """
    return (df - min) / (max - min)


def normalise_linear_inv(df_norm: pd.DataFrame, min: float, max: float) -> pd.DataFrame:
    """
    Performs inverse linear scaling (minmax) on dataframe:

    x = x' * (x_max - x_min) + x_min,

    where x' is the normalised value, and x_min and x_max
    are the minimum and maximum values of the associated
    training data, respectively

    :param df_norm: normalised dataframe
    :param min: minimum value of training data
    :param max: maximum value of training data
    :return: inverse normalised dataframe
    """
    return df_norm * (max - min) + min


def print_pollutant_extremes(
    extremes: List[float], contaminants: List[str] = None, bool_print=True
) -> pd.DataFrame:
    """
    Takes a list of minimum and maximum values for contaminants and creates a DataFrame.

    :param extremes: List of floats containing min/max values in pairs [cont1_min, cont1_max, cont2_min, cont2_max, ...]
    :param contaminants: List of contaminant names. If None, defaults to ["NO2", "O3", "PM10", "PM25"]
    :param bool_print: Whether to print the resulting DataFrame
    :return: DataFrame with minimum and maximum values for each contaminant
    """
    if contaminants is None:
        contaminants = ["NO2", "O3", "PM10", "PM25"]

    # Validate input length
    if len(extremes) != len(contaminants) * 2:
        raise ValueError(
            f"Expected {len(contaminants) * 2} values in extremes list (min/max for each contaminant)"
        )

    # Create dictionary for DataFrame
    data = {}
    for i, cont in enumerate(contaminants):
        data[cont] = [extremes[i * 2], extremes[i * 2 + 1]]  # min and max values

    df_minmax = pd.DataFrame(data, index=["min", "max"]).T

    if bool_print:
        print(df_minmax)

    return df_minmax
