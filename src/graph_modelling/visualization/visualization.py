import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.subplots as sp
from pathlib import Path
import datetime
import json
import os

# Base directory path - will be set from main file
BASE_DIR = None
# Timestamp for file naming - automatically generated when module is imported
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def set_base_dir(path):
    """Set the base directory for saving plots"""
    global BASE_DIR
    BASE_DIR = path

    # Create results directory if it doesn't exist
    results_dir = BASE_DIR / "results"
    if not results_dir.exists():
        results_dir.mkdir(parents=True)


def get_file_name(base_name, model_name=None):
    """
    Generate filename with timestamp and optional model name
    Returns None if model_name is "unknown" or None to prevent file creation

    Args:
        base_name: Base filename (e.g., 'training_history')
        model_name: Optional model name to include in filename

    Returns:
        str: Filename with timestamp and model name, or None if model_name is invalid
    """
    # If model_name is "unknown" or None, return None to prevent file creation
    if not model_name or model_name == "unknown":
        return None

    # Otherwise, create a filename with the model name
    return f"{model_name}_{base_name}_{TIMESTAMP}.png"


def plot_training_history(history, model_name=None, use_plotly=False):
    """
    Plot training and validation loss

    Args:
        history: Dictionary containing training and validation loss history
        model_name: Optional model name for file naming
        use_plotly: Whether to use Plotly for interactive plots instead of matplotlib
    """
    if BASE_DIR is None:
        raise ValueError("BASE_DIR not set. Call set_base_dir() first.")

    # Only include model name in title if it's valid
    title_suffix = f" - {model_name}" if model_name and model_name != "unknown" else ""

    if use_plotly:
        # Create plotly figure
        fig = go.Figure()

        # Add traces for training and validation loss
        fig.add_trace(
            go.Scatter(y=history["train_loss"], mode="lines", name="Training Loss")
        )

        fig.add_trace(
            go.Scatter(y=history["val_loss"], mode="lines", name="Validation Loss")
        )

        # Update layout
        fig.update_layout(
            title=f"Training and Validation Loss Over Time{title_suffix}",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            legend_title="Legend",
            template="plotly_white",
            width=800,
            height=500,
            hovermode="x unified",
        )

        # Generate filename with timestamp
        filename = get_file_name("training_history", model_name)
        if filename:
            # Save as HTML for interactive viewing
            html_filename = filename.replace(".png", ".html")
            html_path = str(BASE_DIR / "results" / html_filename)
            fig.write_html(html_path)

            # Also save as PNG for backward compatibility
            fig.write_image(str(BASE_DIR / "results" / filename), scale=2)
            print(f"Interactive plot saved to {html_path}")
    else:
        # Original matplotlib code
        plt.figure(figsize=(10, 6))
        plt.plot(history["train_loss"], label="Training Loss")
        plt.plot(history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training and Validation Loss Over Time{title_suffix}")
        plt.legend()
        plt.grid(True)

        # Generate filename with timestamp
        filename = get_file_name("training_history", model_name)
        plt.savefig(
            str(BASE_DIR / "results" / filename),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def plot_predictions(
    predictions,
    targets,
    city_names=None,
    model_name=None,
    save_metrics=True,
    use_plotly=False,
    energy_metrics=None,
):
    """
    Plot test predictions against actual values for each city and save metrics to log file

    Args:
        predictions: Model predictions (denormalized) - shape should be (timesteps, n_cities)
        targets: Actual target values (denormalized) - shape should be (timesteps, n_cities)
        city_names: Names of cities for the plot labels
        model_name: Optional model name for file naming
        save_metrics: Whether to save metrics to a JSON file
        use_plotly: Whether to use Plotly for interactive plots instead of matplotlib
        energy_metrics: Optional dictionary with energy consumption and inference time metrics
    """
    if BASE_DIR is None:
        raise ValueError("BASE_DIR not set. Call set_base_dir() first.")

    if city_names is None:
        city_names = ["Amsterdam", "Rotterdam", "Utrecht"]

    # Debug shape information
    print(f"Predictions shape: {predictions.shape}, Targets shape: {targets.shape}")

    # Ensure predictions and targets are proper numpy arrays
    predictions = np.array(predictions)
    targets = np.array(targets)

    # Check if we need to reshape the arrays - they might be coming in as (timesteps*n_cities,)
    # or as (timesteps, n_cities) where n_cities=1 for a single city
    if len(predictions.shape) == 1:
        # If we have a flat array and know the number of cities
        n_cities = len(city_names)
        n_timesteps = len(predictions) // n_cities

        # Reshape to (timesteps, n_cities)
        predictions = predictions.reshape(n_timesteps, n_cities)
        targets = targets.reshape(n_timesteps, n_cities)
        print(f"Reshaped arrays to: {predictions.shape}")

    # Now determine the number of cities from the data shape
    if len(predictions.shape) == 1:
        # Still a flat array, assume one city
        n_cities = 1
        n_samples = len(predictions)
        # Reshape to 2D array with one column
        predictions = predictions.reshape(-1, 1)
        targets = targets.reshape(-1, 1)
    else:
        # 2D array with shape (timesteps, n_cities)
        n_samples, n_cities = predictions.shape

    print(f"Working with {n_cities} cities, {n_samples} time points")

    # Create time steps for x-axis
    time_steps = np.arange(n_samples)

    # Dictionary to store metrics for each city
    city_metrics = {}

    # Calculate metrics for each city (for both plotting methods)
    for i in range(n_cities):
        city_name = city_names[i] if i < len(city_names) else f"City {i}"

        # Extract predictions and targets for this city
        city_preds = predictions[:, i]
        city_targets = targets[:, i]

        # Calculate error metrics
        mse = np.mean((city_preds - city_targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(city_preds - city_targets))

        # Store metrics for this city
        city_metrics[city_name] = {
            "MSE": float(mse),
            "RMSE": float(rmse),
            "MAE": float(mae),
        }
    if use_plotly:
        # Create interactive plotly figure
        fig = sp.make_subplots(
            rows=n_cities,
            cols=1,
            subplot_titles=[
                f"NO2 Predictions for {city_names[i] if i < len(city_names) else f'City {i}'}"
                for i in range(n_cities)
            ],
            vertical_spacing=0.1,
        )

        # Add traces for each city
        for i in range(n_cities):
            city_name = city_names[i] if i < len(city_names) else f"City {i}"

            # Extract predictions and targets for this city
            city_preds = predictions[:, i]
            city_targets = targets[:, i]

            # Get metrics for this city
            rmse = city_metrics[city_name]["RMSE"]
            mae = city_metrics[city_name]["MAE"]

            # Add actual values
            fig.add_trace(
                go.Scatter(
                    x=time_steps,
                    y=city_targets,
                    mode="lines",
                    name=f"Actual ({city_name})",
                    line=dict(color="blue"),
                    legendgroup=city_name,
                ),
                row=i + 1,
                col=1,
            )

            # Add predicted values
            fig.add_trace(
                go.Scatter(
                    x=time_steps,
                    y=city_preds,
                    mode="lines",
                    name=f"Predicted ({city_name})",
                    line=dict(color="red", dash="dash"),
                    legendgroup=city_name,
                ),
                row=i + 1,
                col=1,
            )

            # Add metrics annotation
            fig.add_annotation(
                text=f"RMSE: {rmse:.2f} μg/m³<br>MAE: {mae:.2f} μg/m³",
                xref=f"x{i + 1}",
                yref=f"y{i + 1}",
                x=0.02,
                y=0.95,
                showarrow=False,
                bgcolor="white",
                opacity=0.8,
                xanchor="left",
                yanchor="top",
            )

        # Update layout
        fig.update_layout(
            height=300 * n_cities,
            width=1000,
            title_text=f"NO2 Prediction Results{' - ' + model_name if model_name and model_name != 'unknown' else ''}",
            template="plotly_white",
            hovermode="x unified",
            legend_tracegroupgap=10,
        )

        # Update x and y axis titles
        for i in range(n_cities):
            fig.update_xaxes(title_text="Time (hours)", row=i + 1, col=1)
            fig.update_yaxes(title_text="NO2 (μg/m³)", row=i + 1, col=1)

        # Generate filename with timestamp - only include valid model name
        predictions_filename = get_file_name("test_predictions", model_name)
        if predictions_filename:
            # Save as HTML for interactive viewing
            html_filename = predictions_filename.replace(".png", ".html")

            html_path = str(BASE_DIR / "results" / html_filename)
            fig.write_html(html_path, include_plotlyjs="cdn")

            # Also save as PNG for backward compatibility
            fig.write_image(str(BASE_DIR / "results" / predictions_filename), scale=2)
            print(f"Interactive plot saved to {html_path}")
        else:
            return  # ignore unknown file name
    else:
        # Original matplotlib code
        # Create subplots - one for each city
        fig, axes = plt.subplots(n_cities, 1, figsize=(12, 4 * n_cities))

        # Make axes iterable even if there's only one city
        if n_cities == 1:
            axes = [axes]

        # Plot predictions vs actual values for each city
        for i in range(n_cities):
            city_name = city_names[i] if i < len(city_names) else f"City {i}"

            # Extract predictions and targets for this city
            city_preds = predictions[:, i]
            city_targets = targets[:, i]

            # Get metrics for this city
            rmse = city_metrics[city_name]["RMSE"]
            mae = city_metrics[city_name]["MAE"]

            # Plot the data
            axes[i].plot(time_steps, city_targets, "b-", label="Actual")
            axes[i].plot(time_steps, city_preds, "r--", label="Predicted")
            axes[i].set_title(f"NO2 Predictions for {city_name}")
            axes[i].set_xlabel("Time (hours)")
            axes[i].set_ylabel("NO2 (μg/m³)")
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

            # Display metrics on the plot
            axes[i].text(
                0.02,
                0.92,
                f"RMSE: {rmse:.2f} μg/m³\nMAE: {mae:.2f} μg/m³",
                transform=axes[i].transAxes,
                bbox=dict(facecolor="white", alpha=0.7),
            )

        plt.tight_layout()

        # Generate filename with timestamp - only include valid model name
        predictions_filename = get_file_name("test_predictions", model_name)
        plt.savefig(
            str(BASE_DIR / "results" / predictions_filename),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # Save metrics to JSON file code remains the same
    if save_metrics:
        # Calculate overall metrics
        overall_mse = np.mean((predictions - targets) ** 2)
        overall_rmse = np.sqrt(overall_mse)
        overall_mae = np.mean(np.abs(predictions - targets))

        # Create metrics dictionary with overall and per-city metrics
        if model_name and model_name != "unknown":
            metrics = {
                "timestamp": TIMESTAMP,
                "model_name": model_name,
                "overall": {
                    "MSE": float(overall_mse),
                    "RMSE": float(overall_rmse),
                    "MAE": float(overall_mae),
                },
                "per_city": city_metrics,
            }

            # Add energy metrics if provided
            if energy_metrics:
                metrics["energy"] = energy_metrics

            # Save metrics to JSON file
            metrics_filename = f"{model_name}_metrics_{TIMESTAMP}.json"
            metrics_path = BASE_DIR / "results" / metrics_filename

            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)

            print(f"Metrics saved to {metrics_path}")

            # Also save a summary to a common log file for easy comparison across runs
            log_file = BASE_DIR / "results" / "model_performance_log_.csv"

            # Create header if file doesn't exist
            if not log_file.exists():
                with open(log_file, "w") as f:
                    header = "timestamp,model_name,overall_rmse,overall_mae"
                    for city in city_names:
                        header += f",{city}_rmse"
                    # Add energy metrics columns to header
                    header += (
                        ",inference_time_s,inference_energy_kWh,training_energy_kWh"
                    )
                    # Add new CO2 and GPU metrics
                    header += ",training_co2_kg,inference_co2_kg"
                    f.write(header + "\n")

            # Append metrics to log file (outside the header check)
            with open(log_file, "a") as f:
                line = f"{TIMESTAMP},{model_name},{overall_rmse:.4f},"
                for city in city_names:
                    if city in city_metrics:
                        line += f",{city_metrics[city]['RMSE']:.4f}"
                    else:
                        line += ",NA"

                # Add energy metrics if available, rounded to 4 decimal places
                if energy_metrics:
                    # Format existing energy metrics
                    inference_time = energy_metrics.get("inference_time_s")
                    inference_time_str = (
                        f"{inference_time:.4g}"
                        if isinstance(inference_time, (float, int))
                        else "NA"
                    )

                    inference_energy = energy_metrics.get("inference_energy_kWh")
                    inference_energy_str = (
                        f"{inference_energy:.4g}"
                        if isinstance(inference_energy, (float, int))
                        else "NA"
                    )

                    training_energy = energy_metrics.get("training_energy_kWh")
                    training_energy_str = (
                        f"{training_energy:.4g}"
                        if isinstance(training_energy, (float, int))
                        else "NA"
                    )

                    # Format new CO2 and GPU metrics
                    training_co2 = energy_metrics.get("training_emissions_gCO2")
                    training_co2_str = (
                        f"{training_co2:.4g}"
                        if isinstance(training_co2, (float, int))
                        else "NA"
                    )

                    inference_co2 = energy_metrics.get("inference_emissions_gCO2")
                    inference_co2_str = (
                        f"{inference_co2:.4g}"
                        if isinstance(inference_co2, (float, int))
                        else "NA"
                    )

                    line += f",{inference_time_str},{inference_energy_str},{training_energy_str}"
                    line += f",{training_co2_str},{inference_co2_str},"
                else:
                    line += ",NA,NA,NA,NA,NA"  # Add NA for all missing metrics

                f.write(line + "\n")

            return metrics

    return None
