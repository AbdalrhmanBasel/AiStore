from src._0_data_preprocessing.preprocessing import preprocess_data
from src._0_data_preprocessing.reporting import show_graph_summary
from recommender.src._1_model_selection.GraphSAGEModelV1 import GraphSAGEModelV0
from recommender.src._2_training_model.training import train_model
from src._3_evaluating_model import evaluate_model
from src._4_recommenderation import generate_recommendations
from src._0_data_preprocessing.preprocessing import preprocess_data



def train():
    """
    Function to handle the training of the model.
    Should include steps like loading data, defining a model, training, and saving the model.
    """
    # Example setup
    model = GraphSAGEModelV0(input_dim=128, hidden_dim=64, output_dim=32)
    # dataloader = DataLoader(dataset)
    # loss_fn = nn.MSELoss()

    # # Setup early stopping and checkpoint saving
    # early_stopping_config = {'patience': 5, 'metric': 'loss'}
    # checkpoint_path = "model_checkpoint.pth"

    # # Start training
    # train_model(model, dataloader, loss_fn, optimizer_name="adam", lr=0.001, epochs=100,
    #             device='cuda', early_stopping=early_stopping_config, checkpoint_path=checkpoint_path)




def evaluate():
    """
    Function to evaluate the trained model.
    Should include steps like model evaluation on test data and performance metrics calculation.
    """
    print("Evaluating the model...")
    # Example:
    evaluate_model()
    pass


def recommender():
    """
    Function to generate recommendations.
    Should include the logic to load the trained model and make predictions on new data.
    """
    print("Generating recommendations...")
    # Example:
    generate_recommendations(12)
    pass


def main():
    """
    Main entry point for the system.
    Orchestrates the execution of the full pipeline (preprocessing, training, evaluation, and recommendation).
    """
    print("Starting AI System...")

    # Step 1: Preprocess the data
    preprocess_data()

    # Step 2: Train the model
    train()

    # Step 3: Evaluate the model
    evaluate()

    # Step 4: Generate recommendations
    recommender()


if __name__ == "__main__":
    # Run the main function to start the system
    main()
