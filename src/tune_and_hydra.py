import hydra
from omegaconf import DictConfig
import torch
import pickle
from pathlib import Path
from hydra.utils import instantiate, to_absolute_path
from torch_geometric.loader import DataLoader
from graph_modelling.utils.train_gnn import train
from graph_modelling.utils.test_gnn import predict_and_evaluate
from graph_modelling.utils.tune_gnn import objective
import optuna
import datetime
import pandas as pd
import matplotlib.pyplot as plt

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

@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    model_path = Path(to_absolute_path(cfg.paths.model_path))
    data_dir = Path(to_absolute_path(cfg.paths.data_dir))

    with open(data_dir / "geometric_pkl/train_dataset.pkl", "rb") as f:
        train_dataset = pickle.load(f)
    with open(data_dir / "geometric_pkl/val_dataset.pkl", "rb") as f:
        val_dataset = pickle.load(f)
    with open(data_dir / "geometric_pkl/test_dataset.pkl", "rb") as f:
        test_dataset = pickle.load(f)

    # Add x_seq to datasets
    num_nodes = 3
    window_size = 72
    n_features = 7
    for ds in (train_dataset, val_dataset, test_dataset):
        for data in ds:
            data.x_seq = data.x.view(num_nodes, window_size, n_features)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 7
    output_dim = 24

    for model_name in cfg.model_list:
        print(f"\n=== Running model: {model_name} ===")
        model_cfg = cfg.model
        model_cfg.name = model_name
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        study = optuna.create_study(
            direction="minimize",
            study_name=f"{model_name}-gnn-tuning-{current_time}",
            storage="sqlite:///gnn_tuning.db",
            load_if_exists=True,
            pruner=optuna.pruners.HyperbandPruner(),
        )

        study.optimize(
            lambda trial: objective(
                trial,
                model_name,
                train_dataset,
                val_dataset,
                input_dim,
                output_dim,
                device=device,
                num_epochs=cfg.n_epochs,
                N_HOURS_U=72,
                N_HOURS_Y=24,
            ),
            n_trials=cfg.n_trials,
        )

        best_trial = study.best_trial

        model = instantiate(model_cfg, **best_trial.params).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=best_trial.params["lr"],
            weight_decay=best_trial.params["weight_decay"],
        )

        criterion = torch.nn.MSELoss()
        train_loader = DataLoader(
            train_dataset,
            batch_size=best_trial.params["batch_size"],
            shuffle=False,
            collate_fn=custom_collate,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=best_trial.params["batch_size"],
            shuffle=False,
            collate_fn=custom_collate,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=best_trial.params["batch_size"],
            shuffle=False,
            collate_fn=custom_collate,
        )

        train_losses, val_losses = train(
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            device,
            cfg.n_epochs,
            cfg.patience,
        )

        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.legend()
        plt.title(f"Loss for {model_name}")
        plt.savefig(model_path / f"loss_{model_name}_{current_time}.png")

        torch.save(
            model.state_dict(),
            model_path / f"model_{model_name}_{current_time}.pt",
        )

        with open(data_dir / "geometric_pkl/y_min_max.pkl", "rb") as f:
            y_min, y_max = pickle.load(f)

        global_rmse, rmse_per_node, mean_no2_rmse = predict_and_evaluate(
            model, test_loader, device, output_dim, y_min, y_max, 24
        )

        results = {
            "model_type": model_name,
            "global_rmse": global_rmse,
            "mean_no2_rmse": mean_no2_rmse,
            "rmse_per_node": rmse_per_node,
        }
        pd.DataFrame([results]).to_csv(
            model_path / f"results_{model_name}_{current_time}.csv", index=False
        )


if __name__ == "__main__":
    main()
