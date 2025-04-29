import optuna
from Features.gpu_optimized import train_gpu_model
from Utils.plotting import plot_optimization_history

def objective(trial):
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    model_variant = trial.suggest_categorical('model_variant', ['vgg16', 'resnet18', 'mobilenetv2'])

    result = train_gpu_model(
        batch_size=batch_size,
        epochs=2,
        learning_rate=learning_rate,
        model_variant=model_variant,
        verbose=False,
        subset=True,  # Use small subset to speed up tuning
        quantize=False,
        amp=True,     # Use Automatic Mixed Precision
    )
    
    accuracy = result["accuracy"]
    return accuracy

def run_gpu_optuna_tuning():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("\nBest Hyperparameters:")
    print(study.best_params)
    print(f"Best Validation Accuracy: {study.best_value:.2f}%")

    plot_optimization_history(study)

if __name__ == "__main__":
    run_gpu_optuna_tuning()