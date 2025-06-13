import torch
import numpy as np
from tqdm import tqdm
from ..visualization.visualization import plot_predictions
from codecarbon import EmissionsTracker
import haversine as hs
from haversine import Unit
import math

# array of distances and also the latitude and longitude of the stations

dict_cities = {
    "amsterdam": (52.3172, 4.7897),
    "rotterdam": (51.9606, 4.4469),
    "utrecht": (52.10503, 5.12448),
}


dict_distances = {
    "amsterdam": {
        hs.haversine(
            dict_cities["amsterdam"],
            dict_cities["utrecht"],
            unit=Unit.KILOMETERS,
        )
    },
    "rotterdam": {
        hs.haversine(
            dict_cities["rotterdam"],
            dict_cities["utrecht"],
            unit=Unit.KILOMETERS,
        )
    },
}


def latlon_to_xy(lat1lon1, lat2lon2):
    """Convert lat/lon to approximate x, y in km using the equirectangular projection."""
    lat1, lon1 = lat1lon1
    lat2, lon2 = lat2lon2
    R = 6371  # Radius of Earth in km
    x = (lon2 - lon1) * (math.pi / 180) * R * math.cos(math.radians((lat1 + lat2) / 2))
    y = (lat2 - lat1) * (math.pi / 180) * R
    return x, y


utr_rot = latlon_to_xy(dict_cities["utrecht"], dict_cities["rotterdam"])

utr_amst = latlon_to_xy(dict_cities["utrecht"], dict_cities["amsterdam"])

coords = torch.tensor(
    [
        latlon_to_xy(dict_cities["utrecht"], dict_cities["rotterdam"]),
        latlon_to_xy(dict_cities["utrecht"], dict_cities["amsterdam"]),
    ],
    dtype=torch.float32,
)  # [2, 2]


def get_scaled_vx_vy(input_matrix):
    Wvh_utrecht = input_matrix[:, -24:, 11]
    WD_utrecht = input_matrix[:, -24:, 10] * 360  # denormalize wind direction

    Wvx_utrecht = Wvh_utrecht * torch.cos(torch.deg2rad(WD_utrecht))
    Wvy_utrecht = Wvh_utrecht * torch.sin(torch.deg2rad(WD_utrecht))

    return Wvx_utrecht, Wvy_utrecht


class PINNLoss(torch.nn.MSELoss):
    def __init__(self, reg_coef=None):
        super(PINNLoss, self).__init__()
        self.reg_coef = reg_coef

    def forward(self, y_hat, y_true, input_matrix=None):
        # Calculate the MSE loss
        mse_loss = torch.nn.functional.mse_loss(
            y_hat[:, :, 2], y_true[:, :, 2]
        )  # mse of utrecht

        # If reg_coef is provided, apply the PINN regularization term
        if self.reg_coef is not None:
            Wvx, Wvy = get_scaled_vx_vy(input_matrix)

            # phy loss = something, L  = mse + reg_coef * phy_loss
            # phy loss = dc/dt + dc/dx + dc/dy

            # dcdt = current prediction - prediction one hour ago, but the zeroth pred one will be compared with the concentration before pred starts
            # y0 = the last observed value before predictions begin
            # Get y0: last known value before prediction starts (at hour 47)
            # Indices of the 3 cities in the input features (example)
            city_feature_indices = [0, 1, 2]  # adjust as needed

            # Extract last known true values for these cities at last time step
            y0 = input_matrix[:, -1, city_feature_indices]  # shape: [B, 3]

            # Add time dim
            y0 = y0.unsqueeze(1)  # shape: [B, 1, 3]

            # Now concatenate with y_hat along time dim
            y_combined = torch.cat((y0, y_hat), dim=1)  # shape: [B, T+1, 3]

            # Compute temporal difference: dcdt = c(t) - c(t-1)
            dcdt = torch.diff(y_combined, dim=1)  # shape: [B, T, N]
            dcdt = dcdt[:, :, 2]

            c_pred = y_hat[:, :, 2]  # shape: [B, N*F] (current utrecht prediction)
            c_ams = y_hat[:, :, 0]  # shape: [B, N*F] (Amsterdam prediction)
            c_rot = y_hat[:, :, 1]  # shape: [B, N*F] (Rotterdam prediction)

            delta_c_ams = c_ams - c_pred  # shape: [B, N*F] (Amsterdam - Utrecht)
            delta_c_rot = c_rot - c_pred  # shape: [B, N*F] (Rotterdam - Utrecht)

            delta_c = torch.stack([delta_c_ams, delta_c_rot], dim=2)
            coords_T = coords.T  # shape: [2, N-1]

            # Compute A = (X^T X)^(-1) X^T
            A = torch.inverse(coords_T @ coords) @ coords_T  # shape: [2, N-1]
            A = A.to(y_hat.device)
            delta_c = delta_c.to(y_hat.device)

            # delta_c_spatial shape: [B, T, N-1]

            # Compute gradients: grad_c shape [B, T, 2]
            grad_c = torch.einsum("ij,btk->bti", A, delta_c)

            # Extract spatial derivatives
            dcdx = grad_c[:, :, 0]  # shape: [B, T]
            dcdy = grad_c[:, :, 1]  # shape: [B, T]

            residual = dcdt + Wvx * dcdx + Wvy * dcdy
            phy_loss = torch.mean(residual**2)  # shape: [B, T]
            total_loss = mse_loss + self.reg_coef * phy_loss
            total_loss = total_loss.to(y_hat.device)
            return total_loss
        raise ValueError("reg_coef must be provided for PINN loss calculation")
        # Compute spatial derivatives from input_matri

        # dcdx and dcdy
        # get latlon coordinatese of all nodes
        # get the distance of each node from each other

        # loop for each node
        # get prediction of that node (c_pred)
        # get predictions of other two nodes (c_other_pred)
        # calculate dc by differencing prediction of that node with the other two nodes c_other_pred - c_pred

        # use approximation equation for dcdx and dcdy

        # city0_NO2 (Amsterdam NO2)
        # city0_P (Amsterdam Pressure)
        # city0_SQ (Amsterdam Solar radiation)
        # city0_WD (Amsterdam Wind direction)
        # city0_Wvh (Amsterdam Wind speed)
        # city0_dewP (Amsterdam Dew point)
        # city0_temp (Amsterdam Temperature)
        # city1_NO2 (Utrecht NO2)
        # city1_P (Utrecht Pressure)
        # city1_SQ (Utrecht Solar radiation)
        # city1_WD (Utrecht Wind direction)
        # city1_Wvh (Utrecht Wind speed)
        # city1_dewP (Utrecht Dew point)
        # city1_temp (Utrecht Temperature)
        # city2_NO2 (Rotterdam NO2)
        # city2_P (Rotterdam Pressure)
        # city2_SQ (Rotterdam Solar radiation)
        # city2_WD (Rotterdam Wind direction)
        # city2_Wvh (Rotterdam Wind speed)
        # city2_dewP (Rotterdam Dew point)
        # city2_temp (Rotterdam Temperature)


def train_model_index(
    model,
    train_loader,
    val_loader,
    edge_index,
    edge_weight,
    device,
    epochs=50,
    patience=5,
    writer=None,
    learning_rate=1e-5,
    weight_decay=1e-6,
    lambda_max=None,
):
    """
    Train model using index-based dataloaders

    Args:
        model: Model to train
        train_loader, val_loader: DataLoaders with training and validation data
        device: PyTorch device
        epochs: Max number of epochs to train
        patience: Early stopping patience (epochs without improvement)
        only_no2: Whether only NO2 values should be used as targets
    """
    criterion = PINNLoss(reg_coef=1e-5)
    # criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-8,
    )
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "epochs": [], "lr": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        train_pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
        for x_batch, y_batch in train_pbar:
            # Move data to device
            x_batch, y_batch = x_batch.to(device).float(), y_batch.to(device).float()

            edge_index, edge_weight = edge_index.to(device), edge_weight.to(device)

            optimizer.zero_grad()
            y_hat = model(x_batch, edge_index, edge_weight, lambda_max=lambda_max)

            # Extract only NO2 values as targets when using all variables
            if y_batch.shape[2] != y_hat.shape[2]:
                # Model is expecting (B, horizon, num_nodes=3)
                # Target is (B, horizon, num_nodes*num_vars=21)

                # Reshape to (B, horizon, num_nodes=3, num_vars=7)
                B, H, NF = y_batch.shape
                num_nodes = 3  # Assuming 3 cities
                num_vars = NF // num_nodes  # Calculate based on actual dimensions

                # Reshape and extract just NO2 (first variable) for each node
                y_batch_reshaped = y_batch.view(B, H, num_nodes, num_vars)
                # Take only NO2 (index 0) for all nodes
                y_batch_no2 = y_batch_reshaped[:, :, :, 0]

                # Calculate loss using only NO2 values
                loss = criterion(y_hat, y_batch_no2, x_batch)

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            train_pbar.set_postfix(
                {"train_loss": sum(train_losses) / len(train_losses)}
            )

        # Validation phase
        val_loss = validate_model(
            model, val_loader, criterion, device, edge_index, edge_weight, lambda_max
        )
        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step(val_loss)

        # Track current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)

        # Update history
        avg_train_loss = sum(train_losses) / len(train_losses)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["epochs"].append(epoch)

        writer.add_scalars("Loss", {"train": avg_train_loss, "val": val_loss}, epoch)

        print(
            f"Epoch {epoch}: Train Loss {avg_train_loss:.6f}, Val Loss {val_loss:.6f} - LR {current_lr:.3g}"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model if needed
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            print(f"Validation loss did not improve {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping after {epoch} epochs")
                model.load_state_dict(best_model_state)
                break

    return model, history


def validate_model(
    model, val_loader, criterion, device, edge_index, edge_weight, lambda_max
):
    """Run validation and return average loss"""
    model.eval()
    val_losses = []
    edge_index, edge_weight = edge_index.to(device), edge_weight.to(device)

    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device).float(), y_batch.to(device).float()

            y_hat = model(x_batch, edge_index, edge_weight, lambda_max=lambda_max)

            if y_batch.shape[2] != y_hat.shape[2]:
                # Reshape to get just NO2 values for each node
                B, H, NF = y_batch.shape
                num_nodes = 3
                num_vars = NF // num_nodes  # Calculate based on actual dimensions

                y_batch_reshaped = y_batch.view(B, H, num_nodes, num_vars)
                y_batch_no2 = y_batch_reshaped[:, :, :, 0]

                loss = criterion(y_hat, y_batch_no2)  # , x_batch)

            val_losses.append(loss.item())

    return sum(val_losses) / len(val_losses)


def evaluate_index(
    model,
    test_loader,
    edge_index,
    edge_weight,
    device,
    loader=None,
    cities=["amsterdam", "rotterdam", "utrecht"],
    lambda_max=None,
):
    model.eval()
    edge_index, edge_weight = edge_index.to(device), edge_weight.to(device)
    criterion = torch.nn.MSELoss()

    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device).float(), y_batch.to(device).float()

            y_hat = model(x_batch, edge_index, edge_weight, lambda_max=lambda_max)

            if y_batch.shape[2] != y_hat.shape[2]:
                B, H, NF = y_batch.shape
                num_nodes = 3
                num_vars = NF // num_nodes  # Calculate based on actual dimensions

                y_batch_reshaped = y_batch.view(B, H, num_nodes, num_vars)
                y_batch_no2 = y_batch_reshaped[:, :, :, 0]

                loss = criterion(y_hat, y_batch_no2)

                all_preds.append(y_hat.cpu().numpy())
                all_targets.append(y_batch_no2.cpu().numpy())

            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f"Test MSE (scaled): {avg_loss:.6f}")
    print(f"Test RMSE (scaled): {np.sqrt(avg_loss):.6f}")

    all_preds_array = np.vstack(all_preds)
    all_targets_array = np.vstack(all_targets)

    # Unscale predictions if loader is provided
    if loader is not None:
        try:
            unscaled_preds = loader.denormalize_no2(all_preds_array)
            unscaled_targets = loader.denormalize_no2(all_targets_array)

            unscaled_mse = np.mean((unscaled_preds - unscaled_targets) ** 2)
            unscaled_rmse = np.sqrt(unscaled_mse)

            print(f"Test MSE (unscaled): {unscaled_mse:.4f}")
            print(f"Test RMSE (unscaled): {unscaled_rmse:.4f} μg/m³")

        except Exception as e:
            print(f"Error during denormalization or plotting: {e}")

    return avg_loss, all_preds_array, all_targets_array
