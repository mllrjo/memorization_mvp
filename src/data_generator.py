# src/data_generator.py

import numpy as np

def generate_exponential_correlated_bitstring(sequence_length, correlation_strength_param, num_sequences):
    """
    Generates synthetic binary bitstrings with exponential correlation.
    The correlation_strength_param controls the decay rate. Higher values mean stronger correlation.
    For simplicity, this example will use a fixed probability for the first bit,
    and then a conditional probability for subsequent bits.
    """
    data = []
    
    # A simple way to model exponential correlation:
    # Probability of the next bit being the same as the previous one.
    # Higher correlation_strength_param means higher P(same_as_prev).
    # Let's map correlation_strength_param from a conceptual value (e.g., 8 for length)
    # to a probability. For a 'length' of L, P(same) = 1 - 1/L roughly.
    # Or more directly:
    # Let's define p_same_as_prev. If 0.5, no correlation. If 0.9, strong correlation.
    # We will use correlation_strength_param to influence p_same.
    
    # For correlation_length = 8, p_same might be around 0.6-0.7
    # This is a simplification for a prototype, actual generation would involve more
    # rigorous methods for exponential correlation.
    
    # Let's map correlation_length to p_same_as_prev
    # E.g., if correlation_length is desired, we can make p_same_as_prev related to 1 - 1/correlation_length
    # For a barebones MVP, let's just pick a fixed p_same for now to demonstrate correlation.
    # A p_same_as_prev of 0.7 means 70% chance next bit is same as prev.
    p_same_as_prev = 0.5 + (correlation_strength_param / (2.0 * sequence_length)) # Crude mapping
    p_same_as_prev = np.clip(p_same_as_prev, 0.55, 0.95) # Ensure it's somewhat correlated but not fixed
    
    for _ in range(num_sequences):
        sequence = np.zeros(sequence_length, dtype=int)
        # First bit is random
        sequence[0] = np.random.randint(0, 2)
        
        for i in range(1, sequence_length):
            if np.random.rand() < p_same_as_prev:
                # Next bit is same as previous
                sequence[i] = sequence[i-1]
            else:
                # Next bit is different from previous
                sequence[i] = 1 - sequence[i-1]
        data.append(sequence)
        
    return np.array(data)

if __name__ == '__main__':
    # MVP Parameters for data generation
    MVP_SEQUENCE_LENGTH = 32
    MVP_CORRELATION_STRENGTH_PARAM = 8 # Represents a conceptual correlation length of 8
    MVP_NUM_SEQUENCES_TRAIN = 1000
    MVP_NUM_SEQUENCES_VAL = 100
    MVP_NUM_SEQUENCES_TEST = 100

    print(f"Generating {MVP_NUM_SEQUENCES_TRAIN} training sequences...")
    train_data = generate_exponential_correlated_bitstring(
        MVP_SEQUENCE_LENGTH, MVP_CORRELATION_STRENGTH_PARAM, MVP_NUM_SEQUENCES_TRAIN
    )
    print(f"Shape of train_data: {train_data.shape}")
    print("Example training sequence:", train_data[0])

    print(f"\nGenerating {MVP_NUM_SEQUENCES_VAL} validation sequences...")
    val_data = generate_exponential_correlated_bitstring(
        MVP_SEQUENCE_LENGTH, MVP_CORRELATION_STRENGTH_PARAM, MVP_NUM_SEQUENCES_VAL
    )
    print(f"Shape of val_data: {val_data.shape}")

    print(f"\nGenerating {MVP_NUM_SEQUENCES_TEST} test sequences...")
    test_data = generate_exponential_correlated_bitstring(
        MVP_SEQUENCE_LENGTH, MVP_CORRELATION_STRENGTH_PARAM, MVP_NUM_SEQUENCES_TEST
    )
    print(f"Shape of test_data: {test_data.shape}")
