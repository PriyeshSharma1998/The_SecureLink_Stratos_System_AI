from fastapi import FastAPI, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import joblib
import numpy as np


class ModelAPI:
    def __init__(self, api_key: str, model_file: str):
        self.api_key = api_key
        self.model = joblib.load(model_file)  # Load the model
        self.api_key_header = APIKeyHeader(name="X-API-KEY")  # Define API key security

    def verify_api_key(self, api_key: str = Depends(APIKeyHeader(name="X-API-KEY"))):
        """Verify the API key."""
        if api_key != self.api_key:
            raise HTTPException(status_code=403, detail="Invalid API Key")

    def predict(self, input_data: dict):
        """Predict using the loaded model."""
        # Convert input data to a NumPy array
        data = np.array([list(input_data.values())])
        prediction = self.model.predict(data)
        return int(prediction[0])

    def create_app(self):
        """Create and return the FastAPI app."""
        app = FastAPI()

        # Define the input schema with feature names
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

        @app.post("/predict")
        def predict(input_data: ModelInput, api_key: str = Depends(self.verify_api_key)):
            """Prediction endpoint."""
            input_dict = input_data.dict()  # Convert Pydantic model to dictionary
            prediction = self.predict(input_dict)  # Get prediction from the model
            return {"prediction": prediction}

        return app
