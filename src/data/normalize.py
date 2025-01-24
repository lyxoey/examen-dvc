import pandas as pd
import numpy as np
from pathlib import Path
import click
import logging
from sklearn.preprocessing import MinMaxScaler
from check_structure import check_existing_file, check_existing_folder
import os

@click.command()
@click.argument('input_filepath', type=click.Path(exists=False), required=0)
@click.argument('output_filepath', type=click.Path(exists=False), required=0)
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in../preprocessed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    #input_filepath = click.prompt('Enter the file path for the input data', type=click.Path(exists=True))
    #input_filepath_users = f"{input_filepath}/raw.csv"
    input_filepath_xtrain = "./data/processed_data/X_train.csv"
    input_filepath_xtest = "./data/processed_data/X_test.csv"
    #output_filepath = click.prompt('Enter the file path for the output preprocessed data (e.g., output/preprocessed_data.csv)', type=click.Path())
    output_filepath = "./data/processed_data/"

    process_data(input_filepath_xtrain, input_filepath_xtest, output_filepath)

def process_data(input_filepath_xtrain, input_filepath_xtest, output_filepath):
    # Import datasets
    X_train = import_dataset(input_filepath_xtrain)
    X_test = import_dataset(input_filepath_xtest)
    

    # Split data into training and testing sets
    X_train_scaled, X_test_scaled = normalize_data(X_train, X_test)


    # Create folder if necessary
    create_folder_if_necessary(output_filepath)

    # Save dataframes to their respective output file paths
    save_dataframes(X_train_scaled, X_test_scaled, output_filepath)

def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)

def normalize_data(X_train, X_test):
    # Normalize data with MinMaxScaler()
    features = ["ave_flot_air_flow","ave_flot_level","iron_feed","starch_flow","amina_flow","ore_pulp_flow","ore_pulp_pH","ore_pulp_density"]

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns = features)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns = features)
    return X_train_scaled, X_test_scaled


def create_folder_if_necessary(output_folderpath):
    # Create folder if necessary
    if check_existing_folder(output_folderpath):
        os.makedirs(output_folderpath)

def save_dataframes(X_train_scaled, X_test_scaled, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train_scaled, X_test_scaled], ['X_train_scaled', 'X_test_scaled']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        if check_existing_file(output_filepath):
            file.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()