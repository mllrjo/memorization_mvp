# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

from src.data_generator import generate_exponential_correlated_bitstring
from src.model import BareBonesTransformer, PositionalEncoding # PositionalEncoding needed if run standalone

# --- MVP Parameters ---
MVP_SEQUENCE_LENGTH = 32
MVP_CORRELATION_STRENGTH_PARAM = 8
MVP_NUM_SEQUENCES_TRAIN = 1000
MVP_NUM_SEQUENCES_TEST = 100 # Using test as the primary evaluation set for MVP
MVP_VOCAB_SIZE = 2 # Binary (0 or 1)

MVP_D_MODEL = 256
MVP_NHEAD = 4
MVP_NUM_LAYERS = 2

MVP_LEARNING_RATE = 0.0001
MVP_BATCH_SIZE = 32
MVP_TRAINING_STEPS = 20000 # Very short run for MVP

def run_mvp_training():
    print("--- Starting MVP Training ---")
    print(f"Sequence Length: {MVP_SEQUENCE_LENGTH}")
    print(f"Correlation Strength: {MVP_CORRELATION_STRENGTH_PARAM}")
    print(f"Model: {MVP_NUM_LAYERS} layer(s), d_model={MVP_D_MODEL}, heads={MVP_NHEAD}")
    print(f"Training Steps: {MVP_TRAINING_STEPS}")
    print(f"Batch Size: {MVP_BATCH_SIZE}")

    # 1. Data Generation
    print("\nGenerating data...")
    train_data_np = generate_exponential_correlated_bitstring(
        MVP_SEQUENCE_LENGTH, MVP_CORRELATION_STRENGTH_PARAM, MVP_NUM_SEQUENCES_TRAIN
    )
    test_data_np = generate_exponential_correlated_bitstring(
        MVP_SEQUENCE_LENGTH, MVP_CORRELATION_STRENGTH_PARAM, MVP_NUM_SEQUENCES_TEST
    )

    # Convert to PyTorch Tensors
    # For language modeling, input is sequence[:-1], target is sequence[1:]
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
        sequence_length=MVP_SEQUENCE_LENGTH -1 # Input sequence is one shorter
    )
    
    # Check if CUDA is available (though we're aiming for CPU first)
    device = torch.device("cpu") # Explicitly use CPU for MVP
    model.to(device)

    # 3. Training Setup
    criterion = nn.CrossEntropyLoss() # For binary classification per bit
    optimizer = optim.Adam(model.parameters(), lr=MVP_LEARNING_RATE)

    # 4. Training Loop
    print("\nStarting training loop...")
    global_step = 0
    for epoch in range(100000): # Iterate over epochs, but break early based on steps
        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            if global_step >= MVP_TRAINING_STEPS:
                break # Exit if we hit total steps
            
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            
            # Model predicts next bit for each position in the input sequence
            # logits: (batch_size, sequence_length-1, vocab_size)
            logits = model(inputs) 
            
            # Flatten logits and targets for CrossEntropyLoss
            # targets are (batch_size, sequence_length-1)
            loss = criterion(logits.view(-1, MVP_VOCAB_SIZE), targets.view(-1))
            
            loss.backward()
            optimizer.step()

            global_step += 1
            if global_step % 10 == 0: # Log every 10 steps
                print(f"Step {global_step}/{MVP_TRAINING_STEPS}, Loss: {loss.item():.4f}")
        
        if global_step >= MVP_TRAINING_STEPS:
            print(f"Reached {MVP_TRAINING_STEPS} training steps. Stopping.")
            break

    # 5. Absolute Minimum Evaluation (Memorization Check)
    print("\n--- Evaluating Memorization on a Training Sample ---")
    model.eval() # Set model to evaluation mode
    
    # Pick one sequence from the training set to check
    sample_idx = np.random.randint(0, MVP_NUM_SEQUENCES_TRAIN)
    sample_input = torch.from_numpy(train_data_np[sample_idx, :-1]).long().unsqueeze(0).to(device) # Add batch dim
    sample_target = torch.from_numpy(train_data_np[sample_idx, 1:]).long().unsqueeze(0).to(device)

    print(f"Original Training Sequence (Input): {train_data_np[sample_idx, :-1].tolist()}")
    print(f"Original Training Sequence (Target): {train_data_np[sample_idx, 1:].tolist()}")
    
    with torch.no_grad():
        predicted_logits = model(sample_input) # (1, sequence_length-1, vocab_size)
        # Get the bit with the highest probability
        predicted_bits = torch.argmax(predicted_logits, dim=-1).squeeze(0) # (sequence_length-1)
    
    print(f"Model Predicted Sequence: {predicted_bits.tolist()}")

    # Check if the predicted sequence matches the target for basic memorization
    matches = (predicted_bits == sample_target.squeeze(0)).sum().item()
    total_bits = sample_target.numel()
    accuracy = matches / total_bits * 100

    print(f"Accuracy on single training sequence: {accuracy:.2f}% ({matches}/{total_bits} bits correct)")
    
    if accuracy > 90: # Simple threshold for MVP
        print("MVP achieved: Model shows ability to memorize parts of training data!")
    else:
        print("MVP status: Model is training, but memorization not strongly evident yet. Consider more steps or debugging.")


if __name__ == '__main__':
    run_mvp_training()
