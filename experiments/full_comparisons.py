# File: experiments/full_comparison.py
import optuna
import torch
from Utils.plotting import plot_all_comparisons
from Features.cpu_optimized import train_cpu_model
from Features.gpu_optimized import train_gpu_model

def run_full_optimization_comparison(hardware):
    model_variants = ["resnet18", "mobilenetv2"]
    comparisons = []

    for model in model_variants:
        print(f"\n==========================")
        print(f" Starting Benchmark: {model.upper()}")
        print(f"==========================")
        runs = []

        if hardware.lower() == 'cpu':

            # CPU Default
            print(" Running CPU Default...")
            cpu_default = train_cpu_model(
                model_variant=model, subset=True, dataset_size=5000,
                quantize=False, verbose=True
            )
            runs.append(("CPU Default", cpu_default))

            # CPU Quantized
            print(" Running CPU Quantized...")
            cpu_quant = train_cpu_model(
                model_variant=model, subset=True, dataset_size=5000,
                quantize=True, verbose=True
            )
            runs.append(("CPU Quantized", cpu_quant))

        if hardware.lower() == 'gpu':

            # GPU Default
            print(" Running GPU Default...")
            gpu_default = train_gpu_model(
                model_variant=model, subset=True, dataset_size=5000,
                amp=False, verbose=True
            )
            runs.append(("GPU", gpu_default))

            # GPU AMP
            print(" Running GPU with AMP...")
            gpu_amp = train_gpu_model(
                model_variant=model, subset=True, dataset_size=5000,
                amp=True, verbose=True
            )
            runs.append(("GPU AMP", gpu_amp))

        if hardware.lower() == 'cpu':
        
            # CPU Tuning
            print(" Running Optuna Tuning for CPU...")
            def cpu_objective(trial):
                bs = trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128])
                lr = trial.suggest_loguniform('lr', 1e-4, 1e-1)
                result = train_cpu_model(
                    model_variant=model,
                    batch_size=bs, learning_rate=lr,
                    subset=True, dataset_size=5000,
                    quantize=False, verbose=True
                )
                return result["accuracy"]

            cpu_study = optuna.create_study(direction="maximize")
            cpu_study.optimize(cpu_objective, n_trials=5)
            best_cpu_params = cpu_study.best_params
            print(f" Best CPU Params: {best_cpu_params}")
            cpu_tuned = train_cpu_model(
                model_variant=model,
                batch_size=best_cpu_params['batch_size'],
                learning_rate=best_cpu_params['lr'],
                subset=True, dataset_size=5000,
                quantize=False, verbose=True
            )
            runs.append(("CPU Tuned", cpu_tuned))


        if hardware.lower() == 'gpu':
            # GPU Tuning
            print(" Running Optuna Tuning for GPU...")
            def gpu_objective(trial):
                bs = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
                lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
                result = train_gpu_model(
                    model_variant=model,
                    batch_size=bs, learning_rate=lr,
                    subset=True, dataset_size=5000,
                    amp=False, verbose=True
                )
                return result["accuracy"]

            gpu_study = optuna.create_study(direction="maximize")
            gpu_study.optimize(gpu_objective, n_trials=5)
            best_gpu_params = gpu_study.best_params
            print(f" Best GPU Params: {best_gpu_params}")
            gpu_tuned = train_gpu_model(
                model_variant=model,
                batch_size=best_gpu_params['batch_size'],
                learning_rate=best_gpu_params['lr'],
                subset=True, dataset_size=5000,
                amp=False, verbose=True
            )
            runs.append(("GPU Tuned", gpu_tuned))

            comparisons.append((model.upper(), runs))
            print(f" Finished Benchmark for {model.upper()}")

    print("\n All model comparisons completed.")
    plot_all_comparisons(comparisons)
    return comparisons
