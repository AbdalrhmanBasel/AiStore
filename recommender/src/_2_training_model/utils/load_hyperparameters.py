




def load_best_hyperparameters(file_path):
    """
    Load the best hyperparameters from a text file.

    Args:
        file_path (str): Path to the file containing the best hyperparameters.

    Returns:
        dict: Dictionary of hyperparameters.
    """
    hyperparameters = {}
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    # Convert values to appropriate types
                    if key in ["HIDDEN_DIM", "BATCH_SIZE", "NUM_LAYERS"]:
                        hyperparameters[key] = int(value)
                    elif key in ["LEARNING_RATE", "DROPOUT_RATE"]:
                        hyperparameters[key] = float(value)
    except Exception as e:
        print(f"[ERROR] Failed to load hyperparameters: {e}")
        raise

    return hyperparameters
