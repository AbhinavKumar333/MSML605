import optuna
from Features.gpu_optimized import train_gpu_model
from Utils.plotting import plot_optimization_history


def objective(trial, subset=False, dataset_size=5000, epochs=2, learning_rate=0.001):
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    model_variant = trial.suggest_categorical('model_variant', ['resnet18', 'mobilenetv2'])

    result = train_gpu_model(
        subset=subset,
        batch_size=batch_size,
        learning_rate=learning_rate,
        model_variant=model_variant,
        epochs=2,
        subset=True,
        dataset_size=5000
    )
    
    accuracy = result["accuracy"]
    return accuracy


def run_gpu_optuna_tuning(subset=False, dataset_size=5000, epochs=2, learning_rate=0.001):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=3, subset=subset, dataset_size=dataset_size, epochs=epochs, learning_rate=learning_rate)

    print("\nBest Hyperparameters:")
    print(study.best_params)
    print(f"Best Validation Accuracy: {study.best_value:.2f}%")

    plot_optimization_history(study)


if __name__ == "__main__":
    run_gpu_optuna_tuning()
