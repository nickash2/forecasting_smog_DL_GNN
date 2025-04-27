#!/usr/bin/env python3

import argparse


def main():
    """Main function to parse arguments and run the appropriate component"""
    parser = argparse.ArgumentParser(description="NO2 prediction with GNNs")
    parser.add_argument(
        "--mode",
        choices=[
            "preprocess",
            "train",
            "train_basic",
            "train_recurrent",
            "train_structured",
            "evaluate",
            "visualize",
        ],
        required=True,
        help="Mode to run the program in",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=150, help="Number of epochs for training"
    )
    parser.add_argument(
        "--lags", type=int, default=72, help="Number of lags (past timesteps) to use"
    )
    parser.add_argument(
        "--horizon", type=int, default=24, help="Prediction horizon (future timesteps)"
    )
    parser.add_argument(
        "--only_no2",
        action="store_true",
        help="Use only NO2 data (no weather variables)",
    )
    args = parser.parse_args()

    if args.mode == "preprocess":
        from graph_modelling.data.preprocess_gnn import main as preprocess_main

        preprocess_main()

    elif args.mode == "train":
        if hasattr(args, "model") and args.model == "recurrent_gnn":
            from graph_modelling.training.train_recurrent_gnn import train_model

            train_model(
                batch_size=args.batch_size,
                epochs=args.epochs,
                lags=args.lags,
                horizon=args.horizon,
                only_no2=args.only_no2,
            )
        elif hasattr(args, "model") and args.model == "structured_gnn":
            from graph_modelling.training.train_structured_gnn import train_model

            train_model(
                batch_size=args.batch_size,
                epochs=args.epochs,
                lags=args.lags,
                horizon=args.horizon,
                only_no2=args.only_no2,
            )
        else:
            from graph_modelling.training.train_basic_gnn import train_model

            train_model(
                batch_size=args.batch_size,
                epochs=args.epochs,
                lags=args.lags,
                horizon=args.horizon,
                only_no2=args.only_no2,
            )

    elif args.mode == "evaluate":
        from graph_modelling.training.evaluation import evaluate_model

        model_type = args.model if hasattr(args, "model") else "default_model_type"
        evaluate_model(model_type=model_type, only_no2=args.only_no2)

    elif args.mode == "visualize":
        from graph_modelling.visualization.plot_utils import create_visualizations

        create_visualizations()


if __name__ == "__main__":
    main()
