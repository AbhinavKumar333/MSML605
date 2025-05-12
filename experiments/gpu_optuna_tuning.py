import optuna
from Features.gpu_optimized import train_gpu_model
from Utils.plotting import plot_all_comparisons, plot_optimization_history


def objective(trial, model_variant):
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)

    result = train_gpu_model(
        batch_size=batch_size,
        learning_rate=learning_rate,
        model_variant=model_variant,
        quantize=False,
        amp=True,
        subset=True,
        epochs=10,
        verbose=False,
        dataset_size=5000
    )
    return result["accuracy"]


def run_gpu_optuna_tuning():
    model_variants = ['resnet18', 'mobilenetv2']
    comparisons = []

    for model_variant in model_variants:
        print(f"\n=== Tuning {model_variant.upper()} ===")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, model_variant), n_trials=3)

        print("\nBest Hyperparameters:")
        print(study.best_params)
        print(f"Best Validation Accuracy: {study.best_value:.2f}%")

        print("\nTraining with Default Settings...")
        default_result = train_gpu_model(
            batch_size=32,
            learning_rate=0.001,
            model_variant=model_variant,
            quantize=False,
            amp=True,
            subset=True,
            epochs=10,
            dataset_size=5000
        )

        print("\nTraining with Best Tuned Settings...")
        tuned_result = train_gpu_model(
            batch_size=study.best_params['batch_size'],
            learning_rate=study.best_params['learning_rate'],
            model_variant=model_variant,
            quantize=False,
            amp=True,
            subset=True,
            epochs=10,
            dataset_size=5000
        )

        comparisons.append((
            model_variant.upper(),
            [
                ("Default", default_result),
                ("Tuned", tuned_result)
            ]
        ))

        # Optional per-model optimization history
        plot_optimization_history(study)

    plot_all_comparisons(comparisons)


if __name__ == "__main__":
    run_gpu_optuna_tuning()
