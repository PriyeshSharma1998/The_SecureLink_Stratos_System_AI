from fastapi import FastAPI, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import joblib
import numpy as np


class ModelAPI:
    def __init__(self, logger,api_key, filename,save_directory):
        #Initialise the ModelAPI class
        self.logger=logger
        self.filename=filename
        self.save_directory=save_directory
        self.api_key = api_key
        self.model = joblib.load(self.save_directory/self.filename.pkl)  # Load the model
        self.api_key_header = APIKeyHeader(name="X-API-KEY")  # Define API key security
        self.logger.log_info("ModelAPI class is initialised in __init__ function in modeldeployment.py")

    def verify_api_key(self, api_key: str = Depends(APIKeyHeader(name="X-API-KEY"))):
        try:
            #Verifying the api key
            self.logger.log_info("Verifying the api key is started in verify_api_key function in ModelAPI class in modeldeployment.py")
            if api_key != self.api_key:
                self.logger.log_error("status_code is 403 and detail is Invalid API key")
                raise HTTPException("status_code is 403 and detail is Invalid API key")
            self.logger.log_info("Verifying the api key is completed in verify_api_key function in ModelAPI class in modeldeployment.py")
        except Exception as e:
            # Error occured while verifying the api key
            self.logger.log_error("Error occured while verifying the api key in verify_api_key function in ModelAPI class in modeldeployment.py",e)
            raise RuntimeError("Error occured while verifying the api key in verify_api_key function in ModelAPI class in modeldeployment.py",e)
        
    def predict(self, input_data: dict):
        try:
            # Converting input data to a NumPy array
            self.logger.log_info("Converting input data to numpy array is started in predict function in ModelAPI class in modeldeployment.py")
            data = np.array([list(input_data.values())])
            prediction = self.model.predict(data)
            self.logger.log_info("Converting input data to numpy array is completed in predict function in ModelAPI class in modeldeployment.py")
            return int(prediction[0])
        except Exception as e:
            # Error occured while converting the input data to numpy array
            self.logger.log_error("Error occured while converting the input data to numpy array in predict function in ModelAPI class in modeldeployment.py",e)
            raise RuntimeError("Error occured while converting the input data to numpy array in predict function in ModelAPI class in modeldeployment.py",e)
        

    def create_app(self):
        try:
            # Create the FastAPI app
            self.logger.log_info("Creating the FastAPI is started in creat_app function in ModelAPI class in modeldeployment.py")
            app = FastAPI()
            
            # Define the input data model
            self.logger.log_info("Defining the input data model")
            class ModelInput(BaseModel):
                Time: float
                V1: float
                V2: float
                V3: float
                V4: float
                V5: float
                V6: float
                V7: float
                V8: float
                V9: float
                V10: float
                V11: float
                V12: float
                V13: float
                V14: float
                V15: float
                V16: float
                V17: float
                V18: float
                V19: float
                V20: float
                V21: float
                V22: float
                V23: float
                V24: float
                V25: float
                V26: float
                V27: float
                V28: float
                V29: float
                Amount: float
                Class: float

            # Define the prediction endpoint
            self.logger.log_info("Defining the prediction endpoint")
            @app.post("/predict")
            
            def predict(input_data: ModelInput, api_key: str = Depends(self.verify_api_key)):
                #Predicting the end point
                input_dict = input_data.dict()  # Convert Pydantic model to dictionary
                prediction = self.predict(input_dict)  # Get prediction from the model
                return {"prediction": prediction}
            self.logger.log_info("Creating the FastAPI is completed in creat_app function in ModelAPI class in modeldeployment.py")
            return app

        except Exception as e:
            # Error occuring while predicting the end point
            self.logger.log_Error("Error occured while predicting the end point in creat_app function in ModelAPI class in modeldeployment.py",e)
            raise RuntimeError("Error occured while predicting the end point in creat_app function in ModelAPI class in modeldeployment.py",e)