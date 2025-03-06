import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from zeus.monitor import ZeusMonitor
from torch.utils.tensorboard import SummaryWriter
import datetime
import torch
import os
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional
from functools import wraps


class MetricsTracker:
    """Wrapper class for tracking training metrics, energy consumption and logging."""

    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "logs",
        track_energy: bool = True,
        track_memory: bool = True,
        track_tensorboard: bool = True,
        verbose: bool = True,
    ):
        """Initialize metrics tracker."""
        self.verbose = verbose
        self.experiment_name = experiment_name
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(log_dir) / f"{experiment_name}_{self.timestamp}"

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Initialize trackers
        self.writer = SummaryWriter(self.log_dir) if track_tensorboard else None
        self.zeus_monitor = (
            ZeusMonitor(log_file=self.log_dir / f"energy_{self.timestamp}.csv")
            if track_energy
            else None
        )

        # Tracking flags
        self.track_energy = track_energy
        self.track_memory = track_memory and torch.cuda.is_available()
        self.track_tensorboard = track_tensorboard

        # Storage for different metrics
        self._init_storage()

        self.total_energy = 0.0

    def _init_storage(self):
        """Initialize storage for different types of metrics."""
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "energy": {"epoch": [], "step": []},
            "memory": {"allocated": [], "reserved": [], "max_allocated": []},
        }

    def track_window(self, window_name: str):
        """
        Decorator for tracking metrics in a specific window.
        Handles nested measurement windows correctly.
        """

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Track active windows to ensure proper nesting
                if not hasattr(self, "_active_windows"):
                    self._active_windows = set()

                # Start tracking if window not already active
                if self.track_energy and window_name not in self._active_windows:
                    self.zeus_monitor.begin_window(window_name)
                    self._active_windows.add(window_name)

                try:
                    result = func(*args, **kwargs)

                    # Collect measurements
                    measurements = {}

                    # Energy measurements - only end if this is our window
                    if self.track_energy and window_name in self._active_windows:
                        energy_measurement = self.zeus_monitor.end_window(window_name)
                        measurements["energy"] = energy_measurement
                        self.total_energy += energy_measurement.total_energy

                        if self.verbose and window_name == "epoch":
                            epoch = result.get("epoch", 0)
                            print(f"\nEpoch {epoch}:")
                            print(f"  Train Loss: {result['train_loss']:.4f}")
                            print(f"  Val Loss: {result['val_loss']:.4f}")
                            print(
                                f"  Epoch Energy: {energy_measurement.total_energy:.2f}J"
                            )
                            print(f"  Epoch Time: {energy_measurement.time:.2f}s")
                            # Calculate and print average step energy
                            if "steps_data" in result:
                                avg_step_energy = np.mean(
                                    [
                                        s["energy"].total_energy
                                        for s in result["steps_data"]
                                    ]
                                )
                                print(f"  Avg Step Energy: {avg_step_energy:.2f}J")

                        # Memory measurements
                        if self.track_memory:
                            memory_stats = self._get_gpu_memory_stats()
                            measurements["memory"] = memory_stats

                        self._active_windows.remove(window_name)

                    # Log to TensorBoard
                    if self.track_tensorboard:
                        self._log_to_tensorboard(window_name, result, measurements)

                    # Store measurements
                    self._store_measurements(window_name, result, measurements)

                    return result, measurements

                except Exception as e:
                    # Clean up only if this is our window
                    if self.track_energy and window_name in self._active_windows:
                        self.zeus_monitor.end_window(window_name)
                        self._active_windows.remove(window_name)
                    raise e

            return wrapper

        return decorator

    def _get_gpu_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics using PyTorch."""
        return {
            "allocated": torch.cuda.memory_allocated() / 1024**2,  # Convert to MB
            "reserved": torch.cuda.memory_reserved() / 1024**2,  # Convert to MB
            "max_allocated": torch.cuda.max_memory_allocated()
            / 1024**2,  # Convert to MB
        }

    def _log_to_tensorboard(self, window_name: str, result: Dict, measurements: Dict):
        """Log metrics to TensorBoard."""
        if not self.writer:
            return

        step = (
            result["epoch"]
            if isinstance(result, dict) and "epoch" in result
            else len(self.metrics["train_loss"])
        )

        # Log energy metrics
        if "energy" in measurements:
            energy = measurements["energy"]
            # Log energy consumption - this is already GPU energy
            self.writer.add_scalar(
                f"Energy/{window_name}/gpu_energy", energy.total_energy, step
            )
            self.writer.add_scalar(f"Energy/{window_name}/time", energy.time, step)

            # Calculate and log power consumption
            if energy.time > 0:
                power = energy.total_energy / energy.time
                self.writer.add_scalar(f"Energy/{window_name}/gpu_power", power, step)

        # Log memory metrics
        if "memory" in measurements:
            for name, value in measurements["memory"].items():
                self.writer.add_scalar(f"Memory/{name}", value, step)

        # Log loss metrics if available
        if isinstance(result, dict):
            if "train_loss" in result:
                self.writer.add_scalar("Loss/train", result["train_loss"], step)
            if "val_loss" in result:
                self.writer.add_scalar("Loss/val", result["val_loss"], step)

    def _store_measurements(self, window_name: str, result: Dict, measurements: Dict):
        """Store measurements in internal storage."""
        # Store losses
        if isinstance(result, dict):
            if "train_loss" in result:
                self.metrics["train_loss"].append(result["train_loss"])
            if "val_loss" in result:
                self.metrics["val_loss"].append(result["val_loss"])

        # Store energy measurements
        if "energy" in measurements:
            self.metrics["energy"][window_name].append(
                {
                    "step": result.get(
                        "step", len(self.metrics["energy"][window_name])
                    ),
                    "epoch": result.get("epoch", -1),
                    "energy": measurements["energy"].total_energy,
                    "time": measurements["energy"].time,
                }
            )

        # Store memory measurements
        if "memory" in measurements:
            for name, value in measurements["memory"].items():
                self.metrics["memory"][name].append(value)

    def plot_metrics(self, save: bool = True):
        """Plot all tracked metrics."""
        num_plots = 2  # Losses and steps
        if self.track_energy:
            num_plots += 1
        if self.track_memory:
            num_plots += 1

        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))
        if num_plots == 1:
            axes = [axes]

        plot_idx = 0

        # Plot losses
        epochs = range(len(self.metrics["train_loss"]))
        axes[plot_idx].plot(epochs, self.metrics["train_loss"], label="Training Loss")
        axes[plot_idx].plot(epochs, self.metrics["val_loss"], label="Validation Loss")
        axes[plot_idx].set_xlabel("Epoch")
        axes[plot_idx].set_ylabel("Loss")
        axes[plot_idx].set_title("Training Progress")
        axes[plot_idx].legend()
        axes[plot_idx].grid(True)

        # Plot energy if available
        step_energies = [m["energy"] for m in self.metrics["energy"]["step"]]

        if self.track_energy:
            plot_idx += 1
            # Extract energy values from dictionaries
            epoch_energies = [m["energy"] for m in self.metrics["energy"]["epoch"]]

            axes[plot_idx].plot(
                epochs, epoch_energies, label="Energy (Epoch)", color="blue"
            )

            axes[plot_idx].set_xlabel("Epoch")
            axes[plot_idx].set_ylabel("Energy (J)")
            axes[plot_idx].set_title("GPU Energy Consumption per Epoch")
            axes[plot_idx].legend()
            axes[plot_idx].grid(True)

        if step_energies:  # Only plot steps if we have data
            plot_idx += 1
            axes[plot_idx].plot(
                step_energies,
                label="Energy (Step)",
                color="green",
                alpha=0.5,
            )

            axes[plot_idx].set_xlabel("Steps")
            axes[plot_idx].set_ylabel("Energy (J)")
            axes[plot_idx].set_title("GPU Energy Consumption per Step")
            axes[plot_idx].legend()
            axes[plot_idx].grid(True)

        # Plot memory if available

        if self.track_memory:
            plot_idx += 1
            for name in self.metrics["memory"]:
                memory_data = self.metrics["memory"][name]
                # Create proper indices for memory data
                memory_x = np.linspace(0, len(epochs) - 1, len(memory_data))
                axes[plot_idx].plot(memory_x, memory_data, label=name)

            axes[plot_idx].set_xlabel("Epoch")
            axes[plot_idx].set_ylabel("Memory (MB)")
            axes[plot_idx].set_title("GPU Memory Usage")
            axes[plot_idx].legend()
            axes[plot_idx].grid(True)

        plt.tight_layout()
        if save:
            plt.savefig(self.log_dir / "metrics.png")
        plt.show()

    def save_metrics(self):
        """Save all metrics to CSV files."""
        # Save losses
        pd.DataFrame(
            {
                "train_loss": self.metrics["train_loss"],
                "val_loss": self.metrics["val_loss"],
            }
        ).to_csv(self.log_dir / "losses.csv")

        # Save memory metrics
        if self.track_memory:
            memory_df = pd.DataFrame(
                {name: values for name, values in self.metrics["memory"].items()}
            )
            memory_df.to_csv(self.log_dir / "memory.csv")

    def end_tracking(self):
        """Clean up and close all trackers."""
        if self.writer:
            self.writer.close()
