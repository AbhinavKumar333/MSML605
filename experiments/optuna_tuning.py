import optuna
from Utils.plotting import plot_comparison
from Features.cpu_optimized import train_cpu_model


def objective(trial):
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)
    model_variant = trial.suggest_categorical('model_variant', ["vgg16", "resnet18", "mobilenetv2"])

    result = train_cpu_model(
        batch_size=batch_size,
        epochs=2,
        learning_rate=learning_rate,
        verbose=False,
        model_variant=model_variant,
        quantize=False
    )

    return result["accuracy"]


def run_optuna_tuning():
    study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=20)
    study.optimize(objective, n_trials=5)  # Only 5 trials temporarily

    print("\nBest Hyperparameters Found:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")

    print(f"\nBest Validation Accuracy: {study.best_value:.2f}%")

    # === Retrain Default Settings
    print("\nTraining with Default Settings...")
    default_result = train_cpu_model(
        batch_size=32,
        epochs=2,
        learning_rate=0.001,
        model_variant="vgg16",
        verbose=True,
        quantize=False
    )

    # === Retrain Best Tuned Settings
    print("\nTraining with Best Tuned Settings...")
    tuned_result = train_cpu_model(
        batch_size=study.best_params['batch_size'],
        epochs=2,
        learning_rate=study.best_params['learning_rate'],
        model_variant=study.best_params['model_variant'],
        verbose=True,
        quantize=False
    )

    # === Plot Comparison
    plot_comparison(default_result, tuned_result)


if __name__ == "__main__":
    run_optuna_tuning()

