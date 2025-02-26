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
                Feature1: float
                Feature2: float
                Feature3: float
                Feature4: float
                Feature5: float
                Feature6: float
                Feature7: float
                Feature8: float
                Feature9: float
                Feature10: float
                Feature11: float
                Feature12: float
                Feature13: float
                Feature14: float
                Feature15: float
                Feature16: float
                Feature17: float
                Feature18: float
                Feature19: float
                Feature20: float
                Feature21: float
                Feature22: float
                Feature23: float
                Feature24: float
                Feature25: float
                Feature26: float
                Feature27: float
                Feature28: float
                Feature29: float
                Feature30: float
                Feature31: float
                TargetFeature: float

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