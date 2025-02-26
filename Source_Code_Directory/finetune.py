import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, make_scorer,precision_recall_curve, recall_score, precision_score, accuracy_score, roc_curve, roc_auc_score, f1_score, matthews_corrcoef, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

class FineTuningTheModel:
    def __init__(self, logger, model):
        #Initialise the FineTuningTheModel class
        self.logger = logger
        self.model = model  # In this case, the model is always SGDClassifier
        self.logger.log_info("FineTuningTheModel class is initialised in __init__ function in finetune.py")

    def grid_search(self, X_train, y_train, X_test, y_test):
        #Performing the grid search hyperparameter tuning on SGDClassifier
        try:
            self.logger.log_info("Grid Search Hyperparameter Tuning is started in grid_search function in FineTuningTheModel class in finetune.py")
            # Parameter grid for SGDClassifier hyperparameter tuning
            param_grid = {
                'loss': ['hinge', 'log', 'squared_hinge'],
                'penalty': ['l2', 'l1', 'elasticnet'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'optimal', 'invscaling'],
                'max_iter': [1000, 2000],
                'tol': [1e-3, 1e-4],
                'early_stopping': [True, False],
            }

            # Custom scoring functions using make_scorer
            scoring = {
                'AUC': make_scorer(roc_auc_score, needs_proba=True),
                'F1': make_scorer(f1_score),    
                'MCC': make_scorer(matthews_corrcoef),
                'Accuracy': make_scorer(accuracy_score),
                'Precision': make_scorer(precision_score),
                'Recall': make_scorer(recall_score),
            }

            # Initialize the SGDClassifier
            self.logger.log_info("Setting up the SGDClassifier for the GridSearchCV hyperparameter tuning")
            model = SGDClassifier(random_state=42, loss='log', max_iter=1000, tol=1e-3, early_stopping=True, n_iter_no_change=5)

            # Grid search for hyperparameter tuning
            self.logger.log_info("Performing the GridSearchCV")
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring=scoring, refit='AUC', n_jobs=-1, verbose=1)
            
            self.logger.log_info("Fitting the grid search by passing X_train and y_train")
            grid_search.fit(X_train, y_train)

            # Output the best parameters and AUC score
            print(f"Best Parameters: {grid_search.best_params_}")
            print(f"Best AUC: {grid_search.best_score_}")

            best_model = grid_search.best_estimator_  # The best model after tuning
            
            self.logger.log_info("Calling the result function in evaluate.py to perform SGDClassifier training and to print train and test metrics")
            self.logger.log_info("Grid Search Hyperparameter Tuning is completed in grid_search function in FineTuningTheModel class in finetune.py")
            # Return the result of the evaluation with the best model
            return self.model.result(best_model, grid_search.best_params_, X_train, y_train, X_test, y_test)

        except Exception as e:
            # Error occured while fine tuning the model
            self.logger.log_error("Error occured while fine tuning the model in grid_search function in FineTuningTheModel class",e)
            raise RuntimeError("Error occured while fine tuning the model in grid_search function in FineTuningTheModel class",e)