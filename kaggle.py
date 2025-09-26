# kaggle.py - FULLY AUTOMATED v3 - Optuna-controlled Augmentations

print("--- Starting Fully Automated Kaggle Workflow ---")

# 1. Install Optuna
!pip install optuna -q

# 2. Standard Imports
import os
import torch
import torch.optim as optim
import torch.nn as nn
import optuna
import shutil

# --- This cell assumes you have already run your setup cell to clone the repo and download data ---

# --- Configuration ---
# Paths updated for the user's Kaggle environment structure
ORIGINAL_DATA_DIR = "/kaggle/working/Train"
PROCESSED_DATA_DIR = "/kaggle/working/Train_processed"
STATS_PATH = os.path.join(PROCESSED_DATA_DIR, "stats.json")
KAGGLE_WORKING_DIR = "/kaggle/working/"
N_TRIALS = 25
TRIAL_EPOCHS = 25

# --- STEP 1: Pre-process the dataset and calculate global stats ---
print("\n" + "*"*80)
print("STEP 1: PRE-PROCESSING DATASET FOR FASTER TRAINING")
print("*"*80 + "\n")

preprocess_command = f"python preprocess_data.py --input_dir {ORIGINAL_DATA_DIR} --output_dir {PROCESSED_DATA_DIR}"
!{preprocess_command}

print("\n--- Pre-processing complete. ---")

# --- STEP 2: Run Optuna Hyperparameter Search ---
print("\n" + "*"*80)
print("STEP 2: STARTING OPTUNA HYPERPARAMETER SEARCH")
print("This will take several hours.")
print("*"*80 + "\n")

# Import from your project scripts AFTER they are available
from data_utils_combo import create_segmentation_dataloaders_combo
from models import UNet
from train_utils import train_model

def objective(trial: optuna.trial.Trial) -> float:
    print(f"\n--- Starting Trial {trial.number} ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    tversky_alpha = trial.suggest_float("tversky_alpha", 0.3, 0.7)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 24])
    augment = trial.suggest_categorical("augment", [True, False]) # Let Optuna decide

    train_loader, val_loader = create_segmentation_dataloaders_combo(
        PROCESSED_DATA_DIR, batch_size=batch_size, num_workers=4, augment=augment
    )
    
    model = UNet(in_channels=5, out_channels=1)
    if torch.cuda.device_count() > 1:
      print(f"--- Using {torch.cuda.device_count()} GPUs via DataParallel ---")
      model = nn.DataParallel(model)
    model.to(device)

    from train_utils import TverskyLoss
    criterion = TverskyLoss(alpha=tversky_alpha, beta=(1.0-tversky_alpha))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) if optimizer_name == "Adam" else optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    try:
        trial_save_path = os.path.join(KAGGLE_WORKING_DIR, f"optuna_trial_{trial.number}")
        _, history = train_model(
            model=model, train_loader=train_loader, val_loader=val_loader,
            criterion=criterion, optimizer=optimizer, num_epochs=TRIAL_EPOCHS,
            device=device, model_save_path=trial_save_path,
            stats_path=STATS_PATH, 
            early_stopping_patience=7, use_amp=True, augment=augment # Pass augment flag
        )
        best_val_mcc = max(history.get("val_mcc", [0]))
        shutil.rmtree(trial_save_path, ignore_errors=True)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"--- Trial {trial.number} ran out of memory and will be pruned. ---")
            raise optuna.exceptions.TrialPruned()
        else:
            raise e

    return best_val_mcc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS)

# --- Print Final Results ---
print("\n\n--- Optuna Search Complete ---")
best_trial = study.best_trial
print(f"Best trial achieved a validation MCC of: {best_trial.value:.4f}")

print("\nOptimal hyperparameters found:")
for key, value in best_trial.params.items():
    print(f"  --{key}: {value}")

# Conditionally add the --augment flag based on the best trial's parameters
augment_flag = "--augment" if best_trial.params.get("augment", False) else ""

print("\n" + "*"*80)
print("COPY AND RUN THIS COMMAND TO TRAIN YOUR FINAL MODEL:")
print("*"*80 + "\n")

final_model_path = "/kaggle/working/final_model"

final_command = f"python train_model.py " \
                f"--data_dir '{PROCESSED_DATA_DIR}' " \
                f"--use_combo_loader " \
                f"--stats_path '{STATS_PATH}' " \
                f"--model_type unet " \
                f"--loss tversky " \
                f"--tversky_alpha {best_trial.params['tversky_alpha']:.4f} " \
                f"--tversky_beta {1.0 - best_trial.params['tversky_alpha']:.4f} " \
                f"--learning_rate {best_trial.params['learning_rate']:.6f} " \
                f"--weight_decay {best_trial.params['weight_decay']:.6f} " \
                f"--batch_size {best_trial.params['batch_size']} " \
                f"--epochs 150 " \
                f"--optimizer {best_trial.params['optimizer'].lower()} " \
                f"--scheduler plateau " \
                f"--amp " \
                f"{augment_flag} " \
                f"--threshold_sweep " \
                f"--early_stopping_patience 20 " \
                f"--num_workers 4 " \
                f"--model_save_path '{final_model_path}'"

print(final_command)
print("\n" + "*"*80)