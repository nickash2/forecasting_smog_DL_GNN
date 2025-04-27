import torch
import numpy as np
from pathlib import Path
from torch_geometric_temporal.signal import StaticGraphTemporalSignalBatch
from tqdm import tqdm


def get_temporal_signal_batch(
    loader,
    lags=24,
    batch_size=32,
    only_no2=None,
    sample_size=None,
    cache=True,
    cache_suffix=None,
):
    """
    Get a StaticGraphTemporalSignalBatch for batched temporal graph processing

    Args:
        loader: NO2DatasetLoader instance
        lags: Number of lag timesteps to include
        batch_size: Batch size for training
        only_no2: Whether to use only NO2 or include weather variables
        sample_size: Fraction or number of samples to use
        cache: Whether to use cached data
        cache_suffix: Suffix for cache filename

    Returns:
        StaticGraphTemporalSignalBatch object
    """
    # This is a placeholder for the implementation
    # You will need to implement this in the NO2DatasetLoader class
    # For now, this just shows the expected function signature
    if hasattr(loader, "get_batched_dataset"):
        return loader.get_batched_dataset(
            lags=lags,
            batch_size=batch_size,
            only_no2=only_no2,
            sample_size=sample_size,
            cache=cache,
            cache_suffix=cache_suffix,
        )
    else:
        raise NotImplementedError(
            "The loader does not implement get_batched_dataset for StaticGraphTemporalSignalBatch"
        )


def train_with_temporal_batch(
    model,
    train_dataset,
    val_dataset,
    device,
    epochs=100,
    patience=10,
    model_save_path=None,
    writer=None,
):
    """
    Train a model using StaticGraphTemporalSignalBatch with validation

    Args:
        model: The GNN model to train
        train_dataset: Training StaticGraphTemporalSignalBatch object
        val_dataset: Validation StaticGraphTemporalSignalBatch object
        device: Device to train on (cuda/cpu)
        epochs: Maximum number of training epochs
        patience: Early stopping patience
        model_save_path: Path to save the best model
        writer: TensorBoard summary writer

    Returns:
        Trained model and training history
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()

    num_batches = len(train_dataset)
    # For tracking metrics
    best_val_loss = float("inf")
    counter = 0
    best_model_state = None
    history = {"train_loss": [], "val_loss": []}

    print(f"Training on {device}...")

    for epoch in tqdm(range(epochs), desc="Epochs", total=epochs):
        # Training phase
        model.train()
        total_loss = 0.0
        batch_count = 0

        # Process each batch (snapshot) from the dataset
        train_iterator = tqdm(
            enumerate(train_dataset),
            desc=f"Training",
            total=len(list(train_dataset)),
            leave=False,
        )
        for t, snapshot in train_iterator:
            # Move snapshot to device
            snapshot = snapshot.to(device)

            # Forward pass
            optimizer.zero_grad()

            # Get predictions from model
            predictions = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)

            # Calculate loss
            loss = criterion(predictions, snapshot.y.unsqueeze(1))

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update metrics
            total_loss += loss.item()
            batch_count += 1

            # Update progress bar description
            train_iterator.set_postfix({"loss": f"{loss.item():.6f}"})

        # Calculate average training loss
        avg_train_loss = total_loss / batch_count if batch_count > 0 else float("inf")
        history["train_loss"].append(avg_train_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        val_batch_count = 0

        with torch.no_grad():
            val_iterator = tqdm(
                enumerate(val_dataset),
                desc=f"Validation",
                total=len(list(val_dataset)),
                leave=False,
            )
            for t, snapshot in val_iterator:
                # Move snapshot to device
                snapshot = snapshot.to(device)

                # Get predictions from model
                predictions = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)

                # Calculate loss
                val_loss = criterion(predictions, snapshot.y.unsqueeze(1))

                # Update metrics
                total_val_loss += val_loss.item()
                val_batch_count += 1

                # Update progress bar description
                val_iterator.set_postfix({"val_loss": f"{val_loss.item():.6f}"})

        # Calculate average validation loss
        avg_val_loss = (
            total_val_loss / val_batch_count if val_batch_count > 0 else float("inf")
        )
        history["val_loss"].append(avg_val_loss)

        # Early stopping logic based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            counter = 0
            # Save best model if path provided
            if model_save_path:
                torch.save(model.state_dict(), model_save_path)
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Log to TensorBoard if provided
        if writer is not None:
            writer.add_scalar("Loss/train", avg_train_loss, epoch)
            writer.add_scalar("Loss/val", avg_val_loss, epoch)

        print(
            f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}"
        )

    # Load best model if found
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history
