import logging
from sklearn.model_selection import train_test_split
from logging_utils import LoggerUtility  # Import the LoggerUtility class

class DataSplitter:
    def __init__(self,target_column,logger):
        # Initialize the logger using LoggerUtility
        self.logger = logger
        self.target_column = target_column

    def split_data(self, preprocessed_data):
        #Splitting the datset
        try:
            self.logger.log_info("Splitting data started in split_data function in DataSplitter class in test_train_split.py")
            self.logger.log_info("Splitting data into training and testing sets...")

            # Separate features (X) and target (y)
            X = preprocessed_data.drop(self.target_column, axis=1)
            y = preprocessed_data[self.target_column]

            # Split into train and test sets (with stratified sampling to preserve class distribution)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, stratify=y, random_state=42
            )

            self.logger.log_info("Splitting data successfully completed in split_data function in DataSplitter class in test_train_split.py")
            return X_train, X_test, y_train, y_test

        except Exception as e:
            self.logger.log_error(f"Error in split data function: {e}")
            raise RuntimeError(f"Error in split data function: {e}")