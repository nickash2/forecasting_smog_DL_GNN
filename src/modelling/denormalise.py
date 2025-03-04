# src/modelling/denormalise.py

# Functions that denormalise the data back to original scale

import torch
import pandas as pd
from pipeline.normalise import calc_combined_min_max_params


def retrieve_min_max(path: str, conts=["NO2", "O3", "PM10", "PM25"]) -> dict:
    """
    Retrieves the min and max values for each contaminant from the given path by:
    - retrieving and reading the .csv file
    - creating a dictionary with contaminant names as keys and min/max values as values
    - returning it

    :param path: str, path to the .csv file
    :param conts: list, list of contaminant names (order is important!)
    :return: dict, dictionary with contaminant names as keys and min/max values as values
    """
    df_minmax = pd.read_csv(path, sep=";", encoding="utf-8", index_col=0)
    min = {f"{cont}_min": df_minmax.loc[cont, "min"] for cont in conts}
    max = {f"{cont}_max": df_minmax.loc[cont, "max"] for cont in conts}
    # ** unpacks the dictionaries into a single dictionary
    return {**min, **max}


def normalise_linear_inv(tensor: torch.tensor, min: float, max: float) -> torch.tensor:
    """
    Performs inverse linear scaling (minmax) on a tensor,
    so the values get restored to their original range

    :param tensor: tensor to be denormalised
    :param min: minimum value of the original range
    :param max: maximum value of the original range
    :return: denormalised tensor
    """
    return (tensor * (max - min)) + min


def denormalise(
    tensor_3D: torch.tensor, path: str, contaminants: list = ["NO2", "O3"]
) -> torch.tensor:
    """
    Helper function for denormalising the predictions:
    - clones the tensor (because it's passed by reference)
    - retrieves the min and max values for each contaminant
    - denormalises the tensor per pollutant (because they have
      different min and max values)
    - returns the denormalised tensor

    :param tensor_3D: 3D tensor to be denormalised
    :param path: path to the .csv file with min and max values
    :param contaminants: list of contaminants (order is important!)
    :return: denormalised 3D tensor
    """
    tensor_3D_copy = tensor_3D.clone().detach()
    dict_minmax = retrieve_min_max(path, conts=contaminants)

    for idx, cont in enumerate(contaminants):
        min_val = dict_minmax[f"{cont}_min"]
        max_val = dict_minmax[f"{cont}_max"]
        # take first and only batch, all
        # time steps, current contaminant
        tensor_3D_copy[:, :, idx] = normalise_linear_inv(
            tensor_3D[:, :, idx], min_val, max_val
        )
    return tensor_3D_copy


def denormalise_dataframes(dfs_list, minmax_path, contaminants=["NO2", "O3"]):
    """
    Denormalizes a list of DataFrames

    :param dfs_list: List of pandas DataFrames
    :param minmax_path: Path to min/max values file
    :param contaminants: List of contaminant columns to denormalize
    :return: List of denormalized DataFrames
    """
    dict_minmax = retrieve_min_max(minmax_path, conts=contaminants)
    denormalized_list = []

    for df in dfs_list:
        df_copy = df.copy()

        for cont in contaminants:
            if cont in df_copy.columns:
                min_val = dict_minmax[f"{cont}_min"]
                max_val = dict_minmax[f"{cont}_max"]
                df_copy[cont] = df_copy[cont] * (max_val - min_val) + min_val

        denormalized_list.append(df_copy)

    return denormalized_list


def denormalize_then_normalize_with_target(
    df,
    input_minmax_path,
    target_minmax_path=None,
    no_target=False,
    contaminants=["NO2", "O3"],
):
    # Get min/max values
    input_params = retrieve_min_max(input_minmax_path, conts=contaminants)
    if no_target:
        target_params = calc_combined_min_max_params(df)
        print("Total params calculated", target_params)
    else:
        target_params = retrieve_min_max(target_minmax_path, conts=contaminants)

    # Create a copy to avoid modifying original
    df_copy = df.copy()

    # Process each contaminant
    for cont in contaminants:
        if cont in df_copy.columns:
            # Step 1: Denormalize using input parameters
            r_min = input_params[f"{cont}_min"]
            r_max = input_params[f"{cont}_max"]
            denormalized = df_copy[cont] * (r_max - r_min) + r_min

            # Step 2: Normalize using target parameters
            u_min = target_params[f"{cont}_min"]
            u_max = target_params[f"{cont}_max"]
            df_copy[cont] = (denormalized - u_min) / (u_max - u_min)

    return df_copy
