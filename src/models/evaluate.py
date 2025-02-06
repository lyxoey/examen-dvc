import pandas as pd
import json
import joblib
import os
import logging
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from ..data.check_structure import check_existing_file, check_existing_folder
from pathlib import Path

# Main function
def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    logger.info("Loading the trained model...")
    model = load_trained_model()

    logger.info("Loading the test data...")
    X_test, y_test = load_test_data()

    logger.info("Evaluating the model...")
    metrics = evaluate_model(model, X_test, y_test)

    logger.info(f"Model evaluation metrics: {metrics}")

# Load the trained model from a joblib file
def load_trained_model():
    """
    Load the trained model from the saved file.
    """
    model_path = "./models/trained_model.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        return model
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")

# Load test data
def load_test_data():
    """
    Load test dataset (X_test and y_test).
    """
    X_test = pd.read_csv("./data/processed_data/X_test_scaled.csv")
    y_test = pd.read_csv("./data/processed_data/y_test.csv")
    y_test = y_test.values.ravel()  
    return X_test, y_test

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using test data and calculate various metrics.
    """
    # Predict with the trained model
    y_pred = model.predict(X_test)

    # Calculate MSE, R^2, and optionally more metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Store metrics in a dictionary
    metrics = {
        "mse": mse,
        "r2": r2,
        "mae": mae,
    }
    
    y_pred = pd.DataFrame(y_pred, columns = ["silica_concentrate"])
    
    # Create folder if necessary 
    if check_existing_folder("./metrics/") :
        os.makedirs("./metrics/")

    #--Saving scores.json file in /metrics folder
    for file, filename in zip([metrics], ['scores']):
        output_filepath = os.path.join("./metrics/", f'{filename}.json')
        if check_existing_file(output_filepath):
            with open(output_filepath, "w", encoding="utf-8") as output_filepath:
                json.dump(file, output_filepath, indent=4, ensure_ascii=False)
            
    for file, filename in zip([y_pred], ['predictions']):
        output_filepath = os.path.join("./data/processed_data/", f'{filename}.csv')
        if check_existing_file(output_filepath):
            file.to_csv(output_filepath, index=False, encoding='utf-8')
            
    return metrics
    
if __name__ == '__main__':
    main()