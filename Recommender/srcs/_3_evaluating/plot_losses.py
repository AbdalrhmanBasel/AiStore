import matplotlib.pyplot as plt

def plot_losses(training_losses, validation_losses):
    """
    Function to plot the training and validation losses.
    
    Parameters:
    - training_losses: List of training losses at each epoch.
    - validation_losses: List of validation losses at each epoch.
    """
    epochs = range(1, len(training_losses) + 1)

    plt.plot(epochs, training_losses, label='Training Loss')
    plt.plot(epochs, validation_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')
    plt.show()
