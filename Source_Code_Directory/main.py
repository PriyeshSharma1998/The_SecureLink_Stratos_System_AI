import os
import sys
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Get paths from environment variables
SOURCE_CODE_DIR = os.getenv("SOURCE_CODE_DIR")
CONFIG_DIR = os.getenv("CONFIG_DIR")
sys.path.append(SOURCE_CODE_DIR)  # Path to logging_utils.py
sys.path.append(CONFIG_DIR)  # Path to logging_utils.py

import logging
from preprocess import DataPreprocessor
from train import ModelTrainer
from test_train_split import DataSplitter
from evaluate import ModelEvaluation
from saving import SavingTheModel
from finetune import FineTuningTheModel
from modeldeployment import ModelAPI
import Config  # Import Config
from logging_utils import LoggerUtility  

class DataLoader:
    def __init__(self, logger):
        #Initialise the dataloader class
        self.logger = logger
        self.data = None
        self.logger.log_info("dataloader class is initialised in __init__ function in main.py")
        

    def load_data(self, data_path):
        try:
            self.preprocessor = DataPreprocessor(self.logger)
            #Load the data by passing the arguments
            self.logger.log_info("Data loading successfully started in load_data function in dataloader class in main.py")
            self.data = self.preprocessor.load_data(data_path)
            #Data loading successfully completed
            self.logger.log_info("Data loading successfully completed in load_data function in dataloader class in main.py")
        except Exception as e:
            #Error occured while loading the data
            self.logger.log_error("Error occured while loading the data in dataloader class in main.py",e)
            raise


class Preprocessor:
    def __init__(self, logger, data):
        #Initialise the preprocessor class
        self.logger = logger
        self.data = data
        self.preprocessed_data_correlation_analysis = None
        self.preprocessed_data_anova = None
        self.logger.log_info("preprocessor class is initialised in __init__ function in main.py")

    def preprocess_data(self):
        try:
            self.preprocessor = DataPreprocessor(self.logger)
            #preprocessing the data by passing the arguments
            self.logger.log_info("Preprocessing data successfully started in preprocess_data function in preprocessor class in main.py")
            self.preprocessed_data_correlation_analysis, self.preprocessed_data_anova = self.preprocessor.preprocess_data(self.data)
            #Data loading successfully completed
            self.logger.log_info("Data loading successfully completed in preprocess_data function in preprocessor class in main.py")
        except Exception as e:
            #Error occured while preprocessing the data
            self.logger.log_error("Error occured while loading the data in preprocess_data function in preprocessor class in main.py",e)
            raise

class DataSplitterManager:
    def __init__(self, logger, preprocessed_data_correlation_analysis, preprocessed_data_anova, target_column):
        #Initialise the DataSplitterManager class
        self.logger = logger
        self.target_column = target_column
        self.preprocessed_data_correlation_analysis = preprocessed_data_correlation_analysis
        self.preprocessed_data_anova = preprocessed_data_anova
        self.X_train_correlation = None
        self.X_test_correlation = None
        self.y_train_correlation = None
        self.y_test_correlation = None
        self.X_train_anova = None
        self.X_test_anova = None
        self.y_train_anova = None
        self.y_test_anova = None
        self.logger.log_info("DataSplitterManager class is initialised in __init__ function in main.py")

    def split_data(self):
        try:
            self.data_splitter = DataSplitter(self.target_column,self.logger)  # Initialize the DataSplitter class
            
            #Splitting the data for correlation analysis variable by passing the arguments
            self.logger.log_info("Splitting data for correlation analysis variable successfully started in split_data function in DataSplitterManager class in main.py")
            self.X_train_correlation, self.X_test_correlation, self.y_train_correlation, self.y_test_correlation = self.data_splitter.split_data(self.preprocessed_data_correlation_analysis)
            self.logger.log_info("Splitting data for correlation analysis variable successfully completed in split_data function in DataSplitterManager class in main.py")
            
            #Splitting the data for ANOVA variable by passing the arguments
            self.logger.log_info("Splitting data for ANOVA variable successfully started in split_data function in DataSplitterManager class in main.py")
            self.X_train_anova, self.X_test_anova, self.y_train_anova, self.y_test_anova = self.data_splitter.split_data(self.preprocessed_data_anova)
            self.logger.log_info("Splitting data for ANOVA variable successfully completed in split_data function in DataSplitterManager class in main.py")

        except Exception as e:
            self.logger.log_error("Error occured while splitting the data in split_data function in DataSplitterManager class in main.py",e)
            raise RuntimeError("Error occured while splitting the data in split_data function in DataSplitterManager class in main.py",e)



class ModelTrainerManager:
    def __init__(self, logger, evaluator,X_train_correlation,y_train_correlation,X_test_correlation,y_test_correlation,X_train_anova,y_train_anova,X_test_anova,y_test_anova,knn_params,random_forest_params,xgboost_params,logistic_params):
        #Initialise the ModelTrainerManager class
        self.logger = logger
        self.X_train_correlation = X_train_correlation
        self.y_train_correlation = y_train_correlation
        self.X_train_anova = X_train_anova
        self.y_train_anova = y_train_anova
        self.X_test_correlation=X_test_correlation
        self.y_test_correlation=y_test_correlation
        self.X_test_anova=X_test_anova
        self.y_test_anova=y_test_anova
        self.knn_params=knn_params
        self.random_forest_params=random_forest_params
        self.xgboost_params=xgboost_params
        self.logistic_params=logistic_params
        self.X_train_correlation_sm=None
        self.y_train_correlation_sm=None
        self.X_train_anova_sm=None
        self.y_train_anova_sm=None
        self.model = None
        self.evaluator= evaluator
        self.logger.log_info("ModelTrainerManager class is initialised in __init__ function in main.py")

    def train_model(self):
        try:
            self.model = ModelTrainer(self.logger,self.evaluator)
            #Using SMOTE method for oversampling the dataset for both correlation and anova dataset by passing the arguments
            #Generating oversampleted dataset for correlation dataset
            self.logger.log_info("Oversampling the correlation dataset using SMOTE started in train_model function in ModelTrainerManager class in main.py")
            self.X_train_correlation_sm,self.y_train_correlation_sm=self.model.smote_imbalanced( self.X_train_correlation, self.y_train_correlation)
            self.logger.log_info("Oversampling the correlation dataset using SMOTE completed in train_model function in ModelTrainerManager class in main.py")
            
            #Generating oversampleted dataset for anova dataset
            self.logger.log_info("Oversampling the anova dataset using SMOTE started in train_model function in ModelTrainerManager class in main.py")
            self.X_train_anova_sm,self.y_train_anova_sm=self.model.smote_imbalanced( self.X_train_anova, self.y_train_anova)
            self.logger.log_info("Oversampling the anova dataset using SMOTE completed in train_model function in ModelTrainerManager class in main.py")
            
            #Training the model using SMOTE data
            self.logger.log_info("Training the model with smote with correlation started")
            self.model.train_model( self.X_train_correlation_sm,self.y_train_correlation_sm,self.X_test_correlation,self.y_test_correlation,self.knn_params,self.random_forest_params,self.xgboost_params,self.logistic_params)
            self.logger.log_info("Training the model with smote with correlation completed")
            
            self.logger.log_info("Training the model with smote with anova started")
            self.model.train_model( self.X_train_anova_sm,self.y_train_anova_sm,self.X_test_anova,self.y_test_anova,self.knn_params,self.random_forest_params,self.xgboost_params,self.logistic_params)
            self.logger.log_info("Training the model with smote with anova completed")
            
            #Training the model without SMOTE data
            self.logger.log_info("Training the model without smote with correlation started")
            self.model.train_model( self.X_train_correlation,self.y_train_correlation,self.X_test_correlation,self.y_test_correlation,self.knn_params,self.random_forest_params,self.xgboost_params,self.logistic_params)
            self.logger.log_info("Training the model without smote with correlation completed")
            
            self.logger.log_info("Training the model without smote with anova started")
            self.model.train_model( self.X_train_anova,self.y_train_anova,self.X_test_anova,self.y_test_anova,self.knn_params,self.random_forest_params,self.xgboost_params,self.logistic_params)
            self.logger.log_info("Training the model without smote with anova completed")

        except Exception as e:
            self.logger.log_error("Error occured while training the model in train_model function in ModelTrainerManager class in main.py",e)
            raise RuntimeError("Error occured while training the model in train_model function in ModelTrainerManager class in main.py",e)

class FineTuningTheModelManager:
    def __init__(self, logger,evaluator,X_train_anova,y_train_anova,X_test_anova,y_test_anova):
        #Initialise the FineTuningTheModelManager class
        self.logger = logger
        self.evaluator=evaluator
        self.X_train_anova = X_train_anova
        self.y_train_anova = y_train_anova
        self.X_test_anova=X_test_anova
        self.y_test_anova=y_test_anova
        self.fine_tune=None
        self.metric_train_afterfinetune=None
        self.metric_test_afterfinetune=None
        self.best_model=None
        self.best_model_hyperparameters=None
        self.logger.log_info("FineTuningTheModelManager class is initialised in __init__ function in main.py")

    def fine_tune(self):
        try:
            #Finetuning the Model
            self.evaluator = ModelEvaluation(self.logger)   
            self.model = ModelTrainer(self.logger,self.evaluator)
            self.fine_tune=FineTuningTheModel(self.logger,self.model)
            
            #Fine tuning the model started
            self.logger.log_info("Fine tuning the model successfully started in FineTune function in FineTuningTheModelManager class in main.py")
            self.metric_train_afterfinetune,self.metric_test_afterfinetune,self.best_model,self.best_model_hyperparameters = self.fine_tune.grid_search(self.X_train_anova, self.y_train_anova, self.X_test_anova, self.y_test_anova)
            self.logger.log_info("Fine tuning the model successfully completed in FineTune function in FineTuningTheModelManager class in main.py")
            #Fine Tuning the model completed
        except Exception as e:
            #Error occured while finetuning the model
            self.logger.log_error("Error occured while finetuning the model in FineTune function in FineTuningTheModelManager class in main.py",e)
            raise RuntimeError("Error occured while finetuning the model in FineTune function in FineTuningTheModelManager class in main.py",e)


class ModelSaverManager:
    def __init__(self, logger, output_path,output_filename,output_metadatafilename,best_model,best_model_hyperparameters,metric_train_afterfinetune,metric_test_afterfinetune):
         #Initialise the ModelSaverManager class
        self.logger = logger
        self.model_save = None
        self.best_model = best_model
        self.output_path=output_path
        self.output_filename=output_filename
        self.output_metadatafilename=output_metadatafilename
        self.best_model_hyperparameters=best_model_hyperparameters
        self.metric_train_afterfinetune=metric_train_afterfinetune
        self.metric_test_afterfinetune=metric_test_afterfinetune
        self.logger.log_info("ModelSaverManager class in intialised in __init__ function in ModelSaverManager class in main.py")

    def save_model(self, model_path):
        try:
            self.model_save=SavingTheModel(self.logger,self.best_model,self.output_filename,self.output_path,self.output_metadatafilename,self.best_model_hyperparameters,self.metric_train_afterfinetune,self.metric_test_afterfinetune)
            
            #Saving the model started
            self.logger_log_info("Saving the model is started in save_model function in ModelSaverManager class in main.py")
            self.model_save.save_model()
            self.logger_log_info("Saving the model is completed in save_model function in ModelSaverManager class in main.py")
            #Saving the model completed
        except Exception as e:
            #Error occured while saving the model
            self.logger.log_error("Error occured while saving the model in save_model function in ModelSaverManager class in main.py",e)
            raise RuntimeError("Error occured while saving the model in save_model function in ModelSaverManager class in main.py",e)
class ModelDeployment:
    def __init__(self, logger, output_path,output_filename,api_key):
        #Initialise the ModelDeployment class
        self.logger = logger
        self.output_path=output_path
        self.output_filename=output_filename
        self.api_key=api_key
        self.logger.log_info("ModelDeployment class is initialised in __init__ function in ModelDeployment class in main.py")

    def model_deployment(self):
        try:
            #Model Deployment started
            self.logger.log_info("Model Deployment started in model_deployment function in ModelDeployment class in main.py")
            model_api = ModelAPI(self.logger,self.api_key, self.output_filename,self.output_path)
            app = model_api.create_app()
            self.logger.log_info("Model Deployment completed in model_deployment function in ModelDeployment class in main.py")
            #Model Deployment completed
        except Exception as e:
            #Error occured while deploying the model
            self.logger.log_error("Error occured while deploying the model in model_deployment function in ModelDeployment class in main.py",e)
            raise RuntimeError("Error occured while deploying the model in model_deployment function in ModelDeployment class in main.py",e)



class MLPipeline:
    def __init__(self):
        #Initialise the MLPipeline class
        self.logger = LoggerUtility(Config.LOG_PATH)  # self.logger is the output variable from LoggerUtility class
        self.data_loader = DataLoader(self.logger)  # self.data_loader is also output variable which stores DataLoader class
        self.preprocessor = None
        self.splitter = None
        self.trainer = None
        self.saver = None
        self.finetune = None
        self.modeldeployer=None
        self.logger.log_info("MLPipeline class is  initialised in __init__ function in main.py")
        

    def run(self):
        try:
            self.logger.log_info("Calling the functions started in run function in MLPipeline class in main.py")

            #Calling the load_data function
            self.logger.log_info("Calling the load_data function")
            self.data_loader = DataLoader(self.logger)
            self.data_loader.load_data(Config.DATA_PATH)

            #Calling preprocess_data function
            self.logger.log_info("Calling preprocess_data function")
            self.preprocessor = Preprocessor(self.logger, self.data_loader.data)  #Passing self.data to preprocessor class by mentioned class name(self.data_loader) and variable name(data)
            self.preprocessor.preprocess_data()

            #Calling split_data function
            self.logger.log_info("Calling split_data function")
            self.splitter = DataSplitterManager(self.logger, self.preprocessor.preprocessed_data_correlation_analysis, self.preprocessor.preprocessed_data_anova,Config.TARGET_FEATURE_NAME)
            self.splitter.split_data()

            #Calling train_model function
            self.logger.log_info("Calling train_model function")
            self.evaluator = ModelEvaluation(self.logger)
            self.trainer = ModelTrainerManager(self.logger, self.evaluator,self.splitter.X_train_correlation, self.splitter.y_train_correlation,self.splitter.X_test_correlation,self.splitter.y_test_correlation,self.splitter.X_train_anova,self.splitter.y_train_anova,self.splitter.X_test_anova,self.splitter.y_test_anova,Config.KNN_PARAMS,Config.RANDOM_FOREST_PARAMS,Config.XGBOOST_PARAM,Config.LOGISTIC_PARAMS)
            self.trainer.train_model()
            
            #Calling fine_tune function
            self.logger.log_info("Calling fine_tune function")
            self.finetune = FineTuningTheModelManager(self.logger, self.evaluator,self.splitter.X_train_anova,self.splitter.y_train_anova,self.splitter.X_test_anova,self.splitter.y_test_anova)
            self.finetune.fine_tune()

            #Calling save_model function
            self.logger.log_info("Calling save_model function")
            self.saver = ModelSaverManager(self.logger,Config.OUTPUT_PATH,Config.OUTPUT_FILENAME,Config.OUTPUT_METADATAFILENAME,self.finetune.best_model,self.finetune.best_model_hyperparameters,self.finetune.metric_train_afterfinetune,self.finetune.metric_test_afterfinetune)
            self.saver.save_model()

            #Calling model_deployment function
            self.logger.log_info("Calling model_deployment function")
            self.modeldeployer = ModelDeployment(self.logger,Config.OUTPUT_PATH,Config.OUTPUT_FILENAME,Config.API_KEY)
            self.modeldeployer.model_deployment()


        except Exception as e:
            #Error occured while calling the functions
            self.logger.log_error("Error occured while calling the functions in run function in MLPipeline class in main.py",e)
            raise RuntimeError("Error occured while calling the functions in run function in MLPipeline class in main.py",e)


if __name__ == "__main__":
    pipeline = MLPipeline()
    try:
        #Running the pipeline
        pipeline.run()
    except Exception as e:
        raise RuntimeError("Error occured during pipeline execution",e)