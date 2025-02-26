import joblib

class SavingTheModel:
    def __init__(self, logger, model,filename,save_directory,metadata_filename,best_model_hyperparameters,metric_train_afterfinetune,metric_test_afterfinetune):
        #Initialise the SavingTheModel class
        self.logger = logger
        self.model = model  
        self.filename = filename
        self.save_directory = save_directory
        self.metadata_filename = metadata_filename
        self.best_model_hyperparameters=best_model_hyperparameters
        self.metric_train_afterfinetune=metric_train_afterfinetune
        self.metric_test_afterfinetune=metric_test_afterfinetune
        self.logger.log_info("SavingTheModel class is initialised in __init__ function in saving.py")
    
    def save_model(self):
        try:
            #Saving the model in pkl file 
            self.logger.log_info("Saving the model in pkl file is successfully started in save_model function in SavingTheModel class in saving.py")
            filepath = self.save_directory/self.filename.pkl
            joblib.dump(self.model, filepath)
            self.logger.log_info("Saving the model in pkl file is successfully completed  in save_model function in SavingTheModel class in saving.py")
        except Exception as e:
            # Error occured while saving the model in pkl file
            self.logger.log_info("Error occured while saving the model in pkl file in save_model function in SavingTheModel class in saving.py",e)
            raise RuntimeError("Error occured while saving the model in pkl file in save_model function in SavingTheModel class in saving.py",e)
        
    def save_metadata(self):
        try:
            # Saving the metadata of the model in txt file
            self.logger.log_info("Saving the metadata of the model in txt file is successfully started in save_metadata function in SavingTheModel class in saving.py")
            metadata_content = f"Model: {self.model}\n"
            metadata_content += f"Best Hyperparameters:\n{self.best_model_hyperparameters}\n"
            metadata_content += f"Training Metrics After Fine-tuning:\n{self.metric_train_afterfinetune}\n"
            metadata_content += f"Test Metrics After Fine-tuning:\n{self.metric_test_afterfinetune}\n"

            # Define file path for metadata
            filepath = self.save_directory/self.metadata_filename.txt

            # Write the content to the .txt file
            with open(filepath, 'w') as file:
                file.write(metadata_content)

            self.logger.log_info("Saving thr metadata of the model in txt file is successfully completed in save_metadata function in SavingTheModel class in saving.py")
        except Exception as e:
            #Error occured while saving the metadata of the model in txt file
            self.logger.log_error("Error occured while saving the metadata of the model in txt file in save_metadata function in SavingTheModel class in saving.py",e)
            raise RuntimeError("Error occured while saving the metadata of the model in txt file in save_metadata function in SavingTheModel class in saving.py",e)
        