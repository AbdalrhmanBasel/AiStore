import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.manifold import TSNE
import torch
import numpy as np



# --- Plot Training and Validation Loss ---
def plot_losses(training_losses, validation_losses):
    """
    Plot training and validation losses interactively.
    
    Parameters:
    - training_losses: List of training losses for each epoch.
    - validation_losses: List of validation losses for each epoch.
    """
    # Set interactive mode to true for non-blocking plotting
    plt.ion()

    # Clear the figure and plot
    plt.clf()

    # Plot the losses
    epochs = range(1, len(training_losses) + 1)
    plt.plot(epochs, training_losses, label='Training Loss')
    plt.plot(epochs, validation_losses, label='Validation Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')

    # Pause for a brief moment to update the plot
    plt.pause(0.1)

# --- Plot Precision-Recall Curve ---
def plot_precision_recall_curve(y_true, y_scores):
    """
    Plot the Precision-Recall curve.
    
    Parameters:
    - y_true: True labels.
    - y_scores: Predicted scores or probabilities.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    
    plt.figure()
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

# --- Plot ROC Curve ---
def plot_roc_curve(y_true, y_scores):
    """
    Plot the ROC curve.
    
    Parameters:
    - y_true: True labels.
    - y_scores: Predicted scores or probabilities.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

# --- Plot t-SNE Embeddings ---
def plot_tsne_embeddings(embeddings, labels):
    """
    Plot t-SNE of node embeddings.
    
    Parameters:
    - embeddings: Learned node embeddings.
    - labels: True labels of the nodes (for color coding).
    """
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    plt.figure()
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Node Embeddings')
    plt.show()

# --- Plot Weight Distribution ---
def plot_weight_distribution(model):
    """
    Plot the weight distribution of the model's parameters.
    
    Parameters:
    - model: The trained model.
    """
    weights = [param.data.cpu().numpy().flatten() for param in model.parameters()]
    all_weights = np.concatenate(weights)
    
    plt.figure()
    plt.hist(all_weights, bins=50, color='blue', alpha=0.7)
    plt.title('Weight Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.show()

# --- Plot Training Time ---
def plot_training_time(training_times):
    """
    Plot the training time for each epoch.
    
    Parameters:
    - training_times: List of training times for each epoch.
    """
    epochs = range(1, len(training_times) + 1)
    
    plt.figure()
    plt.plot(epochs, training_times, label='Training Time per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Time (seconds)')
    plt.title('Training Time per Epoch')
    plt.legend()
    plt.show()

# --- Plot Gradients ---
def plot_gradients(model):
    """
    Plot the gradients of the model's parameters.
    
    Parameters:
    - model: The trained model.
    """
    gradients = [param.grad.cpu().numpy().flatten() for param in model.parameters()]
    all_gradients = np.concatenate(gradients)
    
    plt.figure()
    plt.hist(all_gradients, bins=50, color='red', alpha=0.7)
    plt.title('Gradient Distribution')
    plt.xlabel('Gradient Value')
    plt.ylabel('Frequency')
    plt.show()

# --- Plot Activation Outputs ---
def plot_activation_outputs(model, input_data):
    """
    Plot activation outputs of the model.
    
    Parameters:
    - model: The trained model.
    - input_data: Sample input data for forward pass.
    """
    activations = []

    def hook_fn(module, input, output):
        activations.append(output)

    # Register a hook to capture activations
    hook = model.layer_name_here.register_forward_hook(hook_fn)
    
    # Forward pass through the model
    with torch.no_grad():
        model(input_data)

    # Unregister the hook
    hook.remove()
    
    # Plot activations
    plt.figure()
    for activation in activations:
        plt.plot(activation.cpu().numpy().flatten(), label='Activation')
    plt.title('Activation Outputs')
    plt.xlabel('Neuron Index')
    plt.ylabel('Activation Value')
    plt.legend()
    plt.show()

# --- Final Plot Function ---
def plot_all_metrics(training_losses, validation_losses, y_true, y_scores, embeddings, labels, model, input_data, training_times):
    """
    Plot all relevant metrics in one go.
    
    Parameters:
    - training_losses: List of training losses for each epoch.
    - validation_losses: List of validation losses for each epoch.
    - y_true: True labels.
    - y_scores: Predicted scores or probabilities.
    - embeddings: Node embeddings.
    - labels: True labels for t-SNE.
    - model: Trained model.
    - input_data: Sample input data.
    - training_times: List of training times for each epoch.
    """
    # Plot loss curves
    plot_losses(training_losses, validation_losses)
    
    # Plot Precision-Recall Curve
    plot_precision_recall_curve(y_true, y_scores)
    
    # Plot ROC Curve
    plot_roc_curve(y_true, y_scores)
    
    # Plot t-SNE of embeddings
    plot_tsne_embeddings(embeddings, labels)
    
    # Plot weight distribution
    plot_weight_distribution(model)
    
    # Plot training time per epoch
    plot_training_time(training_times)
    
    # Plot gradients
    plot_gradients(model)
    
    # Plot activations (make sure to adjust the layer name)
    plot_activation_outputs(model, input_data)
