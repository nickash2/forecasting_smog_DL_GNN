import torch
import optuna
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Callable, Optional
from torch.nn import Module
from torch.optim import Optimizer
from torch.nn import Module, MSELoss
from torch.utils.data import Dataset
import os
import sys

print(os.getcwd())
from graph_modelling.models.temporalgnn import TemporalGNN
from graph_modelling.models.basicgnn import BasicGNN
from graph_modelling.models.attentiongnn import AttentionGNN
from graph_modelling.models.temporalattentiongnn import GATGRUGNN
from torch_geometric.data import Batch


def train_epoch_optuna(
    model: Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    criterion: Module,
    device: torch.device,
    output_dim: int,
    trial: optuna.Trial,
):
    """Trains the model for a single epoch within the Optuna optimization loop.

    Args:
        model: The PyTorch model (any `torch.nn.Module`).
        train_loader: The data loader for the training data.
        optimizer: The optimizer used for training.
        criterion: The loss function (any `torch.nn.Module`).
        device: The device to run the training on (CPU or GPU).
        output_dim: The dimension of the output layer.
        trial: The Optuna trial object.

    Returns:
        The average training loss for the epoch.
    """
    model.train()
    epoch_train_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        batch.x_seq = batch.x_seq.to(device)
        optimizer.zero_grad()
        out = model(batch)
        y_target = batch["y"].view(-1, output_dim)

        if out.shape != y_target.shape:
            trial_number = trial.number
            epoch_num = trial.number + 1
            print(
                f"Warning: Shape mismatch in trial {trial_number}, epoch {epoch_num}. Output: {out.shape}, Target: {y_target.shape}. Skipping batch."
            )
            continue

        loss = criterion(out, y_target)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    avg_train_loss = epoch_train_loss / len(train_loader)
    return avg_train_loss


def validate_epoch_optuna(
    model: Module,
    val_loader: DataLoader,
    criterion: Module,
    device: torch.device,
    output_dim: int,
    trial: optuna.Trial,
):
    """Validates the model for a single epoch within the Optuna optimization loop.

    Args:
        model: The PyTorch model (any `torch.nn.Module`).
        val_loader: The data loader for the validation data.
        criterion: The loss function (any `torch.nn.Module`).
        device: The device to run the training on (CPU or GPU).
        output_dim: The dimension of the output layer.
        trial: The Optuna trial object.

    Returns:
        The average validation loss for the epoch.
    """
    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            batch.x_seq = batch.x_seq.to(device)
            out = model(batch)
            y_target = batch["y"].view(
                -1, output_dim
            )  # Access target using consistent key

            if out.shape != y_target.shape:
                trial_number = trial.number
                epoch_num = trial.number + 1
                print(
                    f"Warning: Shape mismatch during validation trial {trial_number}, epoch {epoch_num}. Skipping batch."
                )
                continue

            loss = criterion(out, y_target)
            epoch_val_loss += loss.item()

    avg_val_loss = epoch_val_loss / len(val_loader)
    return avg_val_loss


def objective(
    trial: optuna.Trial,
    model_type: str,
    train_dataset: Dataset,
    val_dataset: Dataset,
    input_dim: int,
    output_dim: int,
    *,
    device: torch.device,
    num_epochs: int = 500,
    patience: int = 10,
    use_lr_scheduler: bool = True,
    N_HOURS_U=72,
    N_HOURS_Y=24,
):
    """Objective function for Optuna optimization.

    Args:
        trial: The Optuna trial object.
        model: The PyTorch model (you create this *outside* the objective function).
        train_dataset: The training dataset.
        val_dataset: The validation dataset.
        input_dim: The input dimension of the model.
        output_dim: The output dimension of the model.
        device: The device to run the training on (CPU or GPU).
        num_epochs: The maximum number of epochs to train for (default: 500).
        patience: The patience for early stopping (default: 10).
        use_lr_scheduler: Whether to use a learning rate scheduler (ReduceLROnPlateau). Defaults to True.

    Returns:
        The best validation loss achieved during training.
    """

    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    hidden_dim = trial.suggest_int("hidden_dim", 16, 128, step=16)
    num_gcn = trial.suggest_int("num_gcn", 1, 4)

    if model_type == "basicgnn":
        model = BasicGNN(
            seq_len=N_HOURS_U,
            num_features=input_dim,
            forecast_horizon=N_HOURS_Y,
            hidden_dim=hidden_dim,
            num_gcn=num_gcn,
        ).to(device)

    elif model_type == "temporalgnn":
        # TemporalGNN-specific parameters
        rnn_layers = trial.suggest_int("rnn_layers", 1, 4)
        rnn_dropout = trial.suggest_float("rnn_dropout", 0.1, 0.5, step=0.1)
        model = TemporalGNN(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            rnn_layers=rnn_layers,
            rnn_dropout=rnn_dropout,
            gcn_layers=num_gcn,
        ).to(device)

    elif model_type == "attentiongnn":
        heads = trial.suggest_int("heads", 1, 8)
        dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
        model = AttentionGNN(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gcn,
            heads=heads,
            dropout=dropout,
        ).to(device)

    elif model_type == "temporalattentiongnn":
        valid_head_options = [
            1,
            2,
            4,
            7,
            8,
        ]
        gat_heads = trial.suggest_categorical(
            "gat_heads", [h for h in valid_head_options if hidden_dim % h == 0]
        )
        gat_layers = trial.suggest_int("gat_layers", 1, 4)
        gru_layers = trial.suggest_int("gru_layers", 1, 4)
        dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)

        model = GATGRUGNN(
            input_features=input_dim,
            seq_len=N_HOURS_U,
            forecast_horizon=N_HOURS_Y,
            hidden_dim=hidden_dim,
            gat_heads=gat_heads,
            gat_layers=gat_layers,
            rnn_layers=gru_layers,
            dropout=dropout,
        ).to(device)

    print(model)

    def custom_collate(data_list):
        # First create the batch as usual
        batch = Batch.from_data_list(data_list)

        # For x_seq, we need to maintain the 4D structure
        batch_size = len(data_list)
        num_nodes = data_list[0].x_seq.size(0)  # Typically 3
        seq_len = data_list[0].x_seq.size(1)  # 72
        n_features = data_list[0].x_seq.size(2)  # 7

        # Stack the x_seq tensors properly to get (batch_size, num_nodes, seq_len, features)
        x_seq_stacked = torch.stack([d.x_seq for d in data_list], dim=0)

        # Make sure it has the right shape
        assert x_seq_stacked.shape == (batch_size, num_nodes, seq_len, n_features), (
            f"Expected shape ({batch_size}, {num_nodes}, {seq_len}, {n_features}), got {x_seq_stacked.shape}"
        )

        # Assign to the batch
        batch.x_seq = x_seq_stacked

        return batch

    print("Creating DataLoaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=0,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = MSELoss()

    if use_lr_scheduler:
        factor = trial.suggest_float("plateau_factor", 0.1, 0.7, step=0.1)
        patience_sched = trial.suggest_int("plateau_patience", 3, 10)
        min_lr = trial.suggest_float("plateau_min_lr", 1e-8, 1e-6, log=True)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience_sched,
            min_lr=min_lr,
            verbose=False,
        )
        print(
            f"  Using ReduceLROnPlateau: factor={factor}, patience={patience_sched}, min_lr={min_lr}"
        )
    else:
        scheduler = None  # No scheduler

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training and validation steps
        avg_train_loss = train_epoch_optuna(
            model, train_loader, optimizer, criterion, device, output_dim, trial
        )
        avg_val_loss = validate_epoch_optuna(
            model, val_loader, criterion, device, output_dim, trial
        )

        if scheduler:
            scheduler.step(avg_val_loss)
            lr = optimizer.param_groups[0]["lr"]  # access current learning rate
            print(
                f"Trial {trial.number} - Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} LR: {lr:.8f}"
            )
        else:
            print(
                f"Trial {trial.number} - Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
            )

        trial.report(avg_val_loss, epoch)

        if trial.should_prune():
            print(f"Trial {trial.number} pruned at epoch {epoch + 1}.")
            raise optuna.exceptions.TrialPruned()

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"Validation loss did not improve. Patience counter: {patience_counter}/{patience}"
            )
        if patience_counter >= patience:
            print(
                f"Trial {trial.number} early stopping triggered at epoch {epoch + 1}."
            )
            break

    return best_val_loss
