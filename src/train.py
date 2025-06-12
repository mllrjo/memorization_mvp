# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import math
from datetime import datetime
import os # Import the 'os' module for file path manipulation

from src.data_generator import generate_exponential_correlated_bitstring
from src.model import BareBonesTransformer

# --- MVP Parameters ---
MVP_SEQUENCE_LENGTH = 32
MVP_CORRELATION_STRENGTH_PARAM = 16 # This is the correlation length
MVP_NUM_SEQUENCES_TRAIN = 100 # Reverted to 1000 as per discussion, adjust for your scale
MVP_NUM_SEQUENCES_TEST = 100   # Test set size

MVP_VOCAB_SIZE = 2 # Binary (0 or 1)

# Larger Model Capacity (as discussed for better memorization)
MVP_D_MODEL = 512
MVP_NHEAD = 8
MVP_NUM_LAYERS = 4

MVP_LEARNING_RATE = 0.0001
MVP_BATCH_SIZE = 32
MVP_TRAINING_STEPS = 100000 # Increased for potential grokking observation

EVAL_INTERVAL = 200 # Evaluate and log every 200 steps
CHECKPOINT_SAVE_INTERVAL = 5000 # Save checkpoint every 5000 steps

def calculate_true_per_bit_entropy(correlation_strength_param, sequence_length):
    """
    Calculates the theoretical true per-bit entropy (lower bound NLL)
    for the exponentially correlated bitstring based on the generation parameters.
    This assumes a first-order Markov chain.
    """
    # Replicate the p_same_as_prev calculation from data_generator
    p_same_as_prev = 0.5 + (correlation_strength_param / (2.0 * sequence_length))
    p_same_as_prev = np.clip(p_same_as_prev, 0.55, 0.95) # Apply clipping as in generator

    p_diff_as_prev = 1.0 - p_same_as_prev

    # Conditional entropy for a binary variable
    # H(X_i | X_{i-1}) = -p_same * log(p_same) - p_diff * log(p_diff)
    
    entropy = 0.0
    if p_same_as_prev > 0:
        entropy -= p_same_as_prev * math.log(p_same_as_prev)
    if p_diff_as_prev > 0:
        entropy -= p_diff_as_prev * math.log(p_diff_as_prev)
        
    return entropy # This is the theoretical minimum NLL per bit (in nats)

def run_mvp_training():
    print("--- Starting MVP Training ---")
    print(f"Sequence Length: {MVP_SEQUENCE_LENGTH}")
    print(f"Correlation Strength Param: {MVP_CORRELATION_STRENGTH_PARAM}")
    print(f"Train Dataset Size: {MVP_NUM_SEQUENCES_TRAIN}")
    print(f"Model: {MVP_NUM_LAYERS} layer(s), d_model={MVP_D_MODEL}, heads={MVP_NHEAD}")
    print(f"Training Steps: {MVP_TRAINING_STEPS}, Eval Interval: {EVAL_INTERVAL}, Checkpoint Save Interval: {CHECKPOINT_SAVE_INTERVAL}")
    print(f"Batch Size: {MVP_BATCH_SIZE}, Learning Rate: {MVP_LEARNING_RATE}")

    # Calculate true per-bit entropy
    true_per_bit_entropy = calculate_true_per_bit_entropy(MVP_CORRELATION_STRENGTH_PARAM, MVP_SEQUENCE_LENGTH)
    print(f"Theoretical True Per-Bit Entropy (Lower Bound NLL): {true_per_bit_entropy:.4f}")

    # Setup logging to a new file each time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_log_{timestamp}.txt"
    log_file = open(log_filename, "w")
    log_file.write(f"Training Log - {timestamp}\n")
    log_file.write(f"Parameters:\n")
    log_file.write(f"  Sequence Length: {MVP_SEQUENCE_LENGTH}\n")
    log_file.write(f"  Correlation Strength Param: {MVP_CORRELATION_STRENGTH_PARAM}\n")
    log_file.write(f"  Train Dataset Size: {MVP_NUM_SEQUENCES_TRAIN}\n")
    log_file.write(f"  Test Dataset Size: {MVP_NUM_SEQUENCES_TEST}\n")
    log_file.write(f"  Model: {MVP_NUM_LAYERS} layers, d_model={MVP_D_MODEL}, nhead={MVP_NHEAD}\n")
    log_file.write(f"  Training Steps: {MVP_TRAINING_STEPS}\n")
    log_file.write(f"  Batch Size: {MVP_BATCH_SIZE}\n")
    log_file.write(f"  Learning Rate: {MVP_LEARNING_RATE}\n")
    log_file.write(f"Theoretical True Per-Bit Entropy (Lower Bound NLL): {true_per_bit_entropy:.4f}\n\n")
    log_file.write("Step,Model_NLL,Memorization_Accuracy(Train),Generalization_Accuracy(Test)\n")


    # 1. Data Generation
    print("\nGenerating data...")
    train_data_np = generate_exponential_correlated_bitstring(
        MVP_SEQUENCE_LENGTH, MVP_CORRELATION_STRENGTH_PARAM, MVP_NUM_SEQUENCES_TRAIN
    )
    test_data_np = generate_exponential_correlated_bitstring(
        MVP_SEQUENCE_LENGTH, MVP_CORRELATION_STRENGTH_PARAM, MVP_NUM_SEQUENCES_TEST
    )

    # Convert to PyTorch Tensors
    train_inputs = torch.from_numpy(train_data_np[:, :-1]).long()
    train_targets = torch.from_numpy(train_data_np[:, 1:]).long()
    test_inputs = torch.from_numpy(test_data_np[:, :-1]).long()
    test_targets = torch.from_numpy(test_data_np[:, 1:]).long()

    train_dataset = TensorDataset(train_inputs, train_targets)
    train_dataloader = DataLoader(train_dataset, batch_size=MVP_BATCH_SIZE, shuffle=True)
    test_dataset = TensorDataset(test_inputs, test_targets)
    test_dataloader = DataLoader(test_dataset, batch_size=MVP_BATCH_SIZE, shuffle=False)

    # 2. Model Initialization
    print("\nInitializing model...")
    model = BareBonesTransformer(
        vocab_size=MVP_VOCAB_SIZE,
        d_model=MVP_D_MODEL,
        nhead=MVP_NHEAD,
        num_layers=MVP_NUM_LAYERS,
        sequence_length=MVP_SEQUENCE_LENGTH -1
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=MVP_LEARNING_RATE)

    # --- Checkpoint Loading Logic ---
    CHECKPOINT_BASE_PATH = "/content/drive/My Drive/memorization_mvp_checkpoints/" # Path for Colab/Drive
    # If running on local machine, you might want to adjust this path
    CHECKPOINT_BASE_PATH = "./checkpoints/" 
    
    start_step = 0
    
    # Create the base directory for checkpoints if it doesn't exist
    if not os.path.exists(CHECKPOINT_BASE_PATH):
        os.makedirs(CHECKPOINT_BASE_PATH, exist_ok=True)

    latest_checkpoint = None
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_BASE_PATH) if f.startswith("model_checkpoint_step_")]
    if checkpoint_files:
        # Sort files by step number to get the latest one
        checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        latest_checkpoint = os.path.join(CHECKPOINT_BASE_PATH, checkpoint_files[-1])

    if latest_checkpoint:
        print(f"Loading checkpoint from {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=device) # map_location to ensure it loads to correct device
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step']
        print(f"Resuming training from step {start_step}")
    else:
        print("No checkpoint found, starting training from scratch.")

    global_step = start_step # Initialize global_step based on loaded checkpoint or 0

    # 3. Training Loop
    print("\nStarting training loop...")
    
    # Create an iterator for the train_dataloader that can be reset
    # This is to get a 'random' batch for train accuracy evaluation.
    train_iter = iter(train_dataloader)

    # Calculate initial epoch to start from
    initial_epoch = start_step // len(train_dataloader)

    for epoch in range(initial_epoch, MVP_TRAINING_STEPS // len(train_dataloader) + 2): # Add 2 for safety
        if global_step >= MVP_TRAINING_STEPS: # Check here to break from outer loop
            break
        
        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            if global_step >= MVP_TRAINING_STEPS:
                break # Break from inner loop

            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            
            logits = model(inputs)
            loss = criterion(logits.view(-1, MVP_VOCAB_SIZE), targets.view(-1))
            
            loss.backward()
            optimizer.step()

            global_step += 1

            # --- Evaluation and Logging ---
            if global_step % EVAL_INTERVAL == 0:
                model.eval() # Set model to evaluation mode
                
                # Evaluate Memorization (Training Accuracy) on a batch
                train_correct_predictions = 0
                train_total_predictions = 0
                try:
                    eval_train_inputs, eval_train_targets = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_dataloader) # Reset iterator if end of epoch
                    eval_train_inputs, eval_train_targets = next(train_iter)

                eval_train_inputs, eval_train_targets = eval_train_inputs.to(device), eval_train_targets.to(device)
                with torch.no_grad():
                    train_predicted_logits = model(eval_train_inputs)
                    train_predicted_bits = torch.argmax(train_predicted_logits, dim=-1)
                    train_correct_predictions = (train_predicted_bits == eval_train_targets).sum().item()
                    train_total_predictions = eval_train_targets.numel()
                
                memorization_accuracy = (train_correct_predictions / train_total_predictions) * 100 if train_total_predictions > 0 else 0


                # Evaluate Generalization (Test Accuracy) on the entire test_dataloader
                generalization_accuracy = 0
                test_correct_predictions = 0
                test_total_predictions = 0
                with torch.no_grad():
                    for test_inputs_batch, test_targets_batch in test_dataloader:
                        test_inputs_batch, test_targets_batch = test_inputs_batch.to(device), test_targets_batch.to(device)
                        test_predicted_logits = model(test_inputs_batch)
                        test_predicted_bits = torch.argmax(test_predicted_logits, dim=-1)
                        test_correct_predictions += (test_predicted_bits == test_targets_batch).sum().item()
                        test_total_predictions += test_targets_batch.numel()
                    generalization_accuracy = (test_correct_predictions / test_total_predictions) * 100 if test_total_predictions > 0 else 0

                # Log results to console and file
                log_message = (
                    f"Step {global_step}/{MVP_TRAINING_STEPS}, "
                    f"Model NLL: {loss.item():.4f}, "
                    f"Memorization Accuracy (Train): {memorization_accuracy:.2f}%, "
                    f"Generalization Accuracy (Test): {generalization_accuracy:.2f}%"
                )
                print(log_message)
                log_file.write(f"{global_step},{loss.item():.4f},{memorization_accuracy:.2f},{generalization_accuracy:.2f}\n")
                
                model.train() # Set model back to train mode

            # --- Checkpoint Saving ---
            if global_step % CHECKPOINT_SAVE_INTERVAL == 0 and global_step > start_step: # Only save new checkpoints
                checkpoint_file = os.path.join(CHECKPOINT_BASE_PATH, f"model_checkpoint_step_{global_step}.pth")
                torch.save({
                    'step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_file)
                print(f"Checkpoint saved to {checkpoint_file}")
        
    print(f"Reached {MVP_TRAINING_STEPS} training steps. Stopping.")


    # 5. Final Absolute Minimum Evaluation (Memorization Check on a training sample for confirmation)
    print("\n--- Final Memorization Check on a Single Training Sample ---")
    model.eval()
    
    sample_idx = np.random.randint(0, MVP_NUM_SEQUENCES_TRAIN)
    sample_input = torch.from_numpy(train_data_np[sample_idx, :-1]).long().unsqueeze(0).to(device)
    sample_target = torch.from_numpy(train_data_np[sample_idx, 1:]).long().unsqueeze(0).to(device)

    print(f"Original Training Sequence (Input): {train_data_np[sample_idx, :-1].tolist()}")
    print(f"Original Training Sequence (Target): {train_data_np[sample_idx, 1:].tolist()}")
    
    with torch.no_grad():
        predicted_logits = model(sample_input)
        predicted_bits = torch.argmax(predicted_logits, dim=-1).squeeze(0)
    
    print(f"Model Predicted Sequence: {predicted_bits.tolist()}")

    final_matches = (predicted_bits == sample_target.squeeze(0)).sum().item()
    final_total_bits = sample_target.numel()
    final_accuracy = final_matches / final_total_bits * 100

    print(f"Accuracy on single training sequence: {final_accuracy:.2f}% ({final_matches}/{final_total_bits} bits correct)")
    
    if final_accuracy > 90:
        print("MVP achieved: Model shows ability to memorize parts of training data!")
    else:
        print("MVP status: Model is training, but memorization not strongly evident yet. Consider more steps or debugging.")

    log_file.close()
    print(f"Results logged to {log_filename}")

if __name__ == '__main__':
    run_mvp_training()
