# colab.py - Optuna Hyperparameter Search for Glacier Segmentation
# This script is designed to be run in a single Google Colab cell.

print("--- Setting up Environment for Optuna Search ---")

# 1. Install Optuna
!pip install optuna -q

# 2. Standard Imports
import os
import torch
import torch.optim as optim
import optuna

# 3. Import from your project scripts
# Ensure your project files (train_utils.py, etc.) are accessible in the Colab environment.
from data_utils import create_segmentation_dataloaders
from models import UNet
from train_utils import train_model, TverskyLoss

# --- Configuration ---
# Directory where your 'Train' data is located in Colab
DATA_DIR = "/content/Train"
# Number of optimization trials to run
N_TRIALS = 15
# Number of epochs for each trial (keep this low to search faster)
TRIAL_EPOCHS = 25 

# --- Optuna Objective Function ---
def objective(trial: optuna.trial.Trial) -> float:
    """
    Defines a single training trial. 
    Optuna will call this function multiple times to find the best hyperparameters.
    """
    print(f"\n--- Starting Trial {trial.number} ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 1. Suggest Hyperparameters ---
    # We define the search space for Optuna here.
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    tversky_alpha = trial.suggest_float("tversky_alpha", 0.3, 0.7)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 24])

    # Tversky beta is dependent on alpha
    tversky_beta = 1.0 - tversky_alpha

    # --- 2. Set up Model, Data, and Loss ---
    # Dataloaders
    train_loader, val_loader = create_segmentation_dataloaders(
        DATA_DIR,
        batch_size=batch_size,
        num_workers=2,
        use_global_stats=True,
        augment=True
    )

    # Model
    model = UNet(in_channels=5, out_channels=1).to(device)

    # Loss Function
    criterion = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)

    # Optimizer
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    # --- 3. Run Training ---
    # We wrap the training in a try...except block to handle CUDA Out-of-Memory errors.
    try:
        _, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=TRIAL_EPOCHS,
            device=device,
            model_save_path=f"/content/optuna_trial_{trial.number}",
            early_stopping_patience=7,
            use_amp=True
        )
        # --- 4. Report Result ---
        best_val_mcc = max(history.get("val_mcc", [0]))

    except RuntimeError as e:
        # If CUDA runs out of memory, we prune the trial
        if "out of memory" in str(e):
            print(f"--- Trial {trial.number} ran out of memory and will be pruned. ---")
            raise optuna.exceptions.TrialPruned()
        else:
            # Re-raise any other runtime error
            raise e
    print(f"--- Trial {trial.number} Finished | Best Val MCC: {best_val_mcc:.4f} ---")
    
    # Optuna Pruning (optional, for advanced use)
    # trial.report(best_val_mcc, step=TRIAL_EPOCHS)
    # if trial.should_prune():
    #     raise optuna.exceptions.TrialPruned()

    return best_val_mcc

# --- Main Execution Block ---
if __name__ == '__main__':
    # Create a study object and specify we want to maximize the MCC
    study = optuna.create_study(direction="maximize")

    # Start the optimization
    study.optimize(objective, n_trials=N_TRIALS)

    # --- Print Final Results ---
    print("\n\n--- Optuna Search Complete ---")
    print(f"Number of finished trials: {len(study.trials)}")
    
    best_trial = study.best_trial
    print(f"Best trial achieved a validation MCC of: {best_trial.value:.4f}")

    print("\nOptimal hyperparameters found:")
    for key, value in best_trial.params.items():
        print(f"  --{key}: {value}")

    print("\n" + "*"*80)
    print("COPY AND RUN THE FOLLOWING COMMAND TO TRAIN YOUR FINAL MODEL:")
    print("*"*80 + "\n")

    # Construct and print the final training command
    final_command = f"!python train_model.py \\\n" \
                    f"  --data_dir \"{DATA_DIR}\" \\\n" \
                    f"  --model_type unet \\\n" \
                    f"  --loss tversky \\\n" \
                    f"  --tversky_alpha {best_trial.params['tversky_alpha']:.4f} \\\n" \
                    f"  --tversky_beta {1.0 - best_trial.params['tversky_alpha']:.4f} \\\n" \
                    f"  --learning_rate {best_trial.params['learning_rate']:.6f} \\\n" \
                    f"  --weight_decay {best_trial.params['weight_decay']:.6f} \\\n" \
                    f"  --batch_size {best_trial.params['batch_size']} \\\n" \
                    f"  --epochs 120 \\\n" \
                    f"  --optimizer {best_trial.params['optimizer'].lower()} \\\n" \
                    f"  --scheduler plateau \\\n" \
                    f"  --amp \\\n" \
                    f"  --global_stats \\\n" \
                    f"  --threshold_sweep \\\n" \
                    f"  --early_stopping_patience 15 \\\n" \
                    f"  --num_workers 2 \\\n" \
                    f"  --model_save_path \"/content/drive/MyDrive/glacier_hack/models/unet_optuna_final\""
    
    print(final_command)
    print("\n" + "*"*80)