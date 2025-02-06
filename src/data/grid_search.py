import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import joblib
from check_structure import check_existing_file, check_existing_folder
import os

"""
BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_FOLDER = BASE_DIR / "data" / "processed_data"
OUTPUT_FOLDER = BASE_DIR / "models" 
CONFIG_FOLDER = BASE_DIR / "config"
"""

# Mapping for model names to classes
MODEL_MAPPING = {
    "ridge": Ridge,
    "elasticnet": ElasticNet,
    "lasso": Lasso,
    "randomforestregressor": RandomForestRegressor,
    "gradientboostingregressor": GradientBoostingRegressor
}

def main():
    """ Runs a GridSearchCV with the selected model from config.json and saves the best parameters in models/best_params.skl
    """
    config = load_config()
    param_grid = load_param_grid()
    model_name = config.get("model_name")
    
    logger = logging.getLogger(__name__)
    logger.info(f'finding best parameters using GridSearch with model {model_name}')
    
    #input_filepath_test_x = os.path.join(INPUT_FOLDER, "X_test_scaled.csv")
    input_filepath_test_x = "./data/processed_data/X_test_scaled.csv"
    input_filepath_train_x = "./data/processed_data/X_train_scaled.csv"
    input_filepath_test_y = "./data/processed_data/y_test.csv"
    input_filepath_train_y = "./data/processed_data/y_train.csv"
    output_folderpath = "./models/"
        
    try:
        # Get the model and parameter grid based on user input
        model, param_grid = get_model_and_params(model_name, param_grid)
    except ValueError as e:
        # Handle invalid model names
        print(e)
        return
    
    # Call the main data processing function with the provided file paths
    find_best_params(input_filepath_test_x, input_filepath_train_x, 
                 input_filepath_test_y, input_filepath_train_y,
                 output_folderpath,
                 model, param_grid)
    
def load_config():
    """
    Load the model selection config (config.json).
    """
    with open("./config/config.json") as f:
        config = json.load(f)
    return config

def load_param_grid():
    """
    Load the parameter grid config (param_grid.json).
    """
    with open(os.path.join("./config/param_grid.json")) as f:
        param_grid = json.load(f)
    return param_grid

def find_best_params(input_filepath_test_x, input_filepath_train_x, 
                 input_filepath_test_y, input_filepath_train_y, 
                 output_folderpath,
                 model, param_grid):
 
    #--Importing dataset
    X_train = pd.read_csv(input_filepath_train_x, sep=",")
    X_test = pd.read_csv(input_filepath_test_x, sep=",")
    y_train = pd.read_csv(input_filepath_train_y, sep=",")
    y_test = pd.read_csv(input_filepath_test_y, sep=",")
    
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                               scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score}")
    
    
    # Create folder if necessary 
    if check_existing_folder(output_folderpath) :
        os.makedirs(output_folderpath)

    #--Saving the best params in .pkl file
    for file, filename in zip([best_params], ['best_params']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.pkl')
        if check_existing_file(output_filepath):
            joblib.dump(best_params, output_filepath)
            
def get_model_and_params(model_name, param_grid):
    """
    Returns the model and parameter grid for a given model name.
    """
    model_name = model_name.lower()
    if model_name in param_grid:
        model_config = param_grid[model_name]
        model_class = MODEL_MAPPING.get(model_name)
        
        if not model_class:
            raise ValueError(f"Model '{model_name}' not found in MODEL_MAPPING.")
        
        model = model_class()  # Instantiate the model
        param_grid = model_config['param_grid']  # Extract the parameter grid
        
        return model, param_grid
    else:
        raise ValueError(f"Model '{model_name}' not found in param_grid.")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]


    main()