import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Base directory path - will be set from main file
BASE_DIR = None


def set_base_dir(path):
    """Set the base directory for saving plots"""
    global BASE_DIR
    BASE_DIR = path


def plot_training_history(history):
    """Plot training and validation loss"""
    if BASE_DIR is None:
        raise ValueError("BASE_DIR not set. Call set_base_dir() first.")

    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        str(BASE_DIR / "results" / "batched_training_history.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_predictions(predictions, targets, city_names=None):
    """
    Plot test predictions against actual values for each city

    Args:
        predictions: Model predictions (denormalized)
        targets: Actual target values (denormalized)
        city_names: Names of cities for the plot labels
    """
    if BASE_DIR is None:
        raise ValueError("BASE_DIR not set. Call set_base_dir() first.")

    if city_names is None:
        city_names = ["Amsterdam", "Rotterdam", "Utrecht"]

    # Determine the number of cities (nodes) from the data shape
    n_cities = predictions.shape[1] if len(predictions.shape) > 1 else 1
    n_samples = len(predictions) // n_cities if n_cities > 1 else len(predictions)

    # Create time steps for x-axis
    time_steps = np.arange(n_samples)

    # Create subplots - one for each city
    fig, axes = plt.subplots(n_cities, 1, figsize=(12, 4 * n_cities))

    # Make axes iterable even if there's only one city
    if n_cities == 1:
        axes = [axes]

    # Plot predictions vs actual values for each city
    for i in range(n_cities):
        city_name = city_names[i] if i < len(city_names) else f"City {i}"

        if n_cities > 1:
            # Extract predictions and targets for this city
            city_preds = predictions[:, i]
            city_targets = targets[:, i]
        else:
            city_preds = predictions
            city_targets = targets

        # Plot the data
        axes[i].plot(time_steps, city_targets, "b-", label="Actual")
        axes[i].plot(time_steps, city_preds, "r--", label="Predicted")
        axes[i].set_title(f"NO2 Predictions for {city_name}")
        axes[i].set_xlabel("Time (hours)")
        axes[i].set_ylabel("NO2 (μg/m³)")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

        # Add error metrics in the plot
        mse = np.mean((city_preds - city_targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(city_preds - city_targets))

        # Display metrics on the plot
        axes[i].text(
            0.02,
            0.92,
            f"RMSE: {rmse:.2f} μg/m³\nMAE: {mae:.2f} μg/m³",
            transform=axes[i].transAxes,
            bbox=dict(facecolor="white", alpha=0.7),
        )

    plt.tight_layout()
    plt.savefig(
        str(BASE_DIR / "results" / "batched_test_predictions.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Also create a scatter plot of predicted vs actual values
    plt.figure(figsize=(10, 8))

    # Different colors for each city
    colors = ["blue", "red", "green", "orange", "purple"]

    for i in range(n_cities):
        if n_cities > 1:
            city_preds = predictions[:, i]
            city_targets = targets[:, i]
        else:
            city_preds = predictions
            city_targets = targets

        plt.scatter(
            city_targets,
            city_preds,
            alpha=0.5,
            color=colors[i % len(colors)],
            label=city_names[i] if i < len(city_names) else f"City {i}",
        )

    # Add reference line (perfect predictions)
    max_val = max(np.max(predictions), np.max(targets))
    min_val = min(np.min(predictions), np.min(targets))
    plt.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.8)

    plt.xlabel("Actual NO2 (μg/m³)")
    plt.ylabel("Predicted NO2 (μg/m³)")
    plt.title("Predicted vs Actual NO2 Values")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")

    plt.savefig(
        str(BASE_DIR / "results" / "batched_prediction_scatter.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
