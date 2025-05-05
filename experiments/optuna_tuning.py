import optuna
import torch
from Utils.plotting import plot_all_comparisons
from Features.cpu_optimized import train_cpu_model

def objective(trial, model_variant):
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)

    result = train_cpu_model(
        batch_size=batch_size,
        epochs=10,
        learning_rate=learning_rate,
        verbose=False,
        model_variant=model_variant,
        quantize=False,
        subset=True,
        dataset_size=5000
    )
    return result["accuracy"]

import optuna
from Features.cpu_optimized import train_cpu_model
from Utils.plotting import plot_all_comparisons


def objective(trial, model_variant):
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-1)

    result = train_cpu_model(
        batch_size=batch_size,
        learning_rate=learning_rate,
        model_variant=model_variant,
        epochs=10,
        verbose=False,
        quantize=False,
        subset=True,
        dataset_size=5000,
    )
    return result["accuracy"]


def run_optuna_tuning():
    model_variants = ["resnet18", "mobilenetv2"]
    all_results = []
    comparisons = []

    for model_variant in model_variants:
        print(f"\n=== Tuning {model_variant.upper()} ===")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, model_variant), n_trials=3)

        print("\nBest Hyperparameters Found:")
        for key, value in study.best_params.items():
            print(f"{key}: {value}")
        print(f"\nBest Validation Accuracy: {study.best_value:.2f}%")

        print("\nTraining with Default Settings...")
        default_result = train_cpu_model(
            batch_size=32,
            epochs=10,
            learning_rate=0.001,
            model_variant=model_variant,
            verbose=True,
            quantize=True,
            subset=True,
            dataset_size=5000
        )

        print("\nTraining with Best Tuned Settings...")
        tuned_result = train_cpu_model(
            batch_size=study.best_params["batch_size"],
            epochs=10,
            learning_rate=study.best_params["learning_rate"],
            model_variant=model_variant,
            verbose=True,
            quantize=True,
            subset=True,
            dataset_size=5000
        )

        all_results.append({
            "model_variant": model_variant,
            "batch_size": study.best_params["batch_size"],
            "learning_rate": study.best_params["learning_rate"],
            "accuracy": tuned_result["accuracy"],
            "quantized_accuracy": tuned_result["quantized_accuracy"],
            "avg_epoch_time": tuned_result["avg_epoch_time"],
            "inference_latency": tuned_result["inference_latency"],
            "peak_memory_MB": tuned_result["peak_memory_MB"]
        })

        # ✅ Updated format for plot_all_comparisons
        comparisons.append((
            model_variant.upper(),
            [
                ("Default", default_result),
                ("Tuned", tuned_result)
            ]
        ))

    plot_all_comparisons(comparisons)
    return all_results
