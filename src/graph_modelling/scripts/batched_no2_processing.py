import torch
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import os
import sys
import datetime
from torch.utils.tensorboard import SummaryWriter
from torch_geometric_temporal.signal import temporal_signal_split
import json

# Updated import paths using relative paths within graph_modelling
from ..training.train_utils import train_model_index, evaluate_index
from ..visualization.visualization import (
    plot_training_history,
    plot_predictions,
    set_base_dir,
)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_TRT_ALLOW_GROWTH"] = "true"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """Main function to process data and train the model using built-in batching"""
    # Initialize the NO2DatasetLoader with index=True to enable batched processing
    N_LAGS = 72
    N_HORIZON = 24
    BATCH_SIZE = 16
    N_STEP = 24
    N_EPOCHS = 2
    PATIENCE = 5
    SAMPLE_SIZE = 1.0
    ONLY_NO2 = True
    ATTENTION = True
    ATTENTION_BOTH = False
    K = 1

    print("Initializing NO2 dataset loader with batching capability...")
    # Set force_reload=False to ensure the cached dataset is used
    loader = NO2DatasetLoader(
        index=True,
        only_no2=ONLY_NO2,
        force_reload=False,
        cache_file="no2_dataset_cache.pkl",
    )

    print(
        f"Loading dataset with {N_LAGS} lags, {N_HORIZON} horizon, and {N_STEP} step size..."
    )
    # Get batched dataset using our utility function
    train_loader, val_loader, test_loader, edges, edge_weights = (
        loader.get_index_dataset(
            lags=N_LAGS,
            batch_size=BATCH_SIZE,
            shuffle=True,
            allGPU=device,
            ratio=(0.7, 0.1, 0.2),
            only_no2=ONLY_NO2,
            sample_size=SAMPLE_SIZE,
            horizon=N_HORIZON,
            cache=True,
            cache_suffix=None,
            step_size=N_STEP,
        )
    )

    print(len(train_loader))
    print(len(val_loader))
    print(len(test_loader))

    print("Train, validation, and test loaders created successfully.")

    # Calculate number of variables and features
    # For NO2 dataset with default settings, we have 7 variables (NO2 + weather vars)
    # Each with N_LAGS timesteps
    num_vars = 7 if not loader.only_no2 else 1

    if ATTENTION:
        model = AttentionGConvGRU(
            num_nodes=3,
            num_vars=num_vars,
            lags=N_LAGS,
            hidden_channels=32,
            horizon=N_HORIZON,
            K=K,
        ).to(device)
    elif ATTENTION_BOTH:
        model = ASTGCN_Like(
            num_nodes=3,
            num_vars=num_vars,
            lags=N_LAGS,
            horizon=N_HORIZON,
            # Adjust channels/blocks as needed
            block_channels=32,
            gru_channels=32,
            K=K,
            num_blocks=1,
        ).to(device)
    else:
        model = BatchedGConvGRUIndex(
            num_nodes=3,
            num_vars=num_vars,
            lags=N_LAGS,
            hidden_channels=32,
            horizon=N_HORIZON,
            K=K,
        ).to(device)

    print(model)

    # Set up TensorBoard for visualization
    log_dir = (
        BASE_DIR
        / "results"
        / "logs"
        / f"batched_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # Train the model using the batched data loaders
    try:
        print("Starting model training with batched data...")
        model, history = train_model_index(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            edge_index=edges,
            edge_weight=edge_weights,
            device=device,
            epochs=N_EPOCHS,
            patience=PATIENCE,
            writer=writer,
        )
    except KeyboardInterrupt:
        print("Cancelling training, continuing onto evaluation.")

    # Close TensorBoard writer
    writer.close()

    # Set base directory for visualization functions
    set_base_dir(BASE_DIR)

    # Plot training history
    plot_training_history(history)

    # Evaluate model on test data
    print("Evaluating model with batched test data...")
    test_loss, predictions, targets = evaluate_index(
        model=model,
        test_loader=test_loader,
        edge_index=edges,
        edge_weight=edge_weights,
        device=device,
        loader=loader,
        cities=["amsterdam", "rotterdam", "utrecht"],
    )

    # Save the final model
    torch.save(model.state_dict(), MODEL_PATH / "final_batched_gnn_model.pt")
    print("Model evaluation complete!")


if __name__ == "__main__":
    sys.path.append(str(Path.cwd()))
    from ..datasets.no2_dataset import NO2DatasetLoader
    from ..models.batched_gconvgru import BatchedGConvGRUIndex
    from ..models.batched_gconvgru_atn import AttentionGConvGRU
    from ..models.astgcn import ASTGCN_Like

    # Set paths and constants
    BASE_DIR = Path.cwd()
    MODEL_PATH = BASE_DIR / "results" / "models"
    DATA_DIR = BASE_DIR / "data" / "data_gnn"
    ALL_DIR = DATA_DIR / "all"
    RAW_DATA_DIR = BASE_DIR / "data" / "data_raw"

    print("BASE_DIR: ", BASE_DIR)
    print("MODEL_PATH: ", MODEL_PATH)
    print("ALL_DIR: ", ALL_DIR)

    # Set random seed for reproducibility
    torch.manual_seed(34)

    # Constants
    N_HOURS_U = 72  # number of hours to use for input
    N_HOURS_Y = 24  # number of hours to predict
    N_HOURS_STEP = 24  # "sampling rate" in hours
    CONTAMINANTS = ["NO2"]  # Only predicting NO2
    WEATHER_VARS = [
        "P",
        "SQ",
        "WD",
        "Wvh",
        "dewP",
        "temp",
    ]  # Weather variables to include
    cities = ["amsterdam", "rotterdam", "utrecht"]
    SAVE_DIR = ALL_DIR / "geometric_pkl"

    # Ensure directories exist
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "scalers").mkdir(parents=True, exist_ok=True)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "results").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "results" / "graph_plots").mkdir(parents=True, exist_ok=True)

    main()
