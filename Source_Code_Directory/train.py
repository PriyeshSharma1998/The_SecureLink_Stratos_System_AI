import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, precision_recall_curve, recall_score, precision_score, accuracy_score, roc_curve, roc_auc_score, f1_score, matthews_corrcoef, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression,SGDClassifier
from xgboost import XGBClassifier

class ModelTrainer:
    def __init__(self, logger,evaluator):
        #Initialise the ModelTrainer class
        self.logger = logger
        self.data = None
        self.evaluator = evaluator  # Initialize evaluation class
        self.logger.log_info("ModelTrainer class is initialised in __init__ function in train.py")

    def smote_imbalanced(self, X_train, y_train):
        try:
            #SMOTE imbalanced resampling started
            self.logger.log_info("SMOTE imbalanced resampling is started in smote_imbalanced function in ModelTrainer class in train.py")
            smote = SMOTE(sampling_strategy='auto', random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            X_resampled_df = pd.DataFrame(X_resampled, columns=X_train.columns)
            y_resampled_df = pd.DataFrame(y_resampled, columns=["Class"])
            resampled_df = pd.concat([X_resampled_df, y_resampled_df], axis=1)
            #SMOTE imbalanced resampling completed
            self.logger.log_info("SMOTE imbalanced resampling is completed in smote_imbalanced function in ModelTrainer class in train.py")
            
            # Checking count frequency after SMOTE and visualize using countplot
            print("Checking count frequency after SMOTE and visualize using countplot started")
            self.logger.log_info("Checking count frequency after SMOTE and visualize using countplot started")
            ax = sns.countplot(x='Class', data=resampled_df)
            plt.title('Number of Fraud Cases') 
            plt.xlabel('Class')
            plt.ylabel('Frequency') 
            ax.set_xticklabels(['No Fraud', 'Fraud'])
            plt.show()
            print("Checking count frequency after SMOTE and visualize using countplot completed")
            self.logger.log_info("Checking count frequency after SMOTE and visualize using countplot completed")
           
            return X_resampled, y_resampled
        except Exception as e:
            #Error occured while resampling the data using SMOTE
            self.logger.log_error("Error occured while resampling the data using SMOTE in smote_imbalanced function in ModelTrainer class in train.py",e)
            raise RuntimeError("Error occured while resampling the data using SMOTE in smote_imbalanced function in ModelTrainer class in train.py",e)

    def result(self, model, model_params, X_train, y_train, X_test, y_test):
        try:
            #Training the model started
            self.logger.log_info("Training the model started is started in result function in ModelTrainer class in train.py")
            classifier = model(**model_params)
            classifier.fit(X_train, y_train)
            
            #predict y test values and probabilities
            #This calculates binary labels after sigmoid which 0 or 1
            self.logger.log_info("predicting the test data started")
            y_test_pred = classifier.predict(X_test)  
            #Lets select the probability of class 1 i.e., second column
            #This calculates probabilities of each class since we selected [:,1] so we are fetching second column
            y_test_pred_prob = classifier.predict_proba(X_test)[:, 1]
            self.logger.log_info("predicting the test data completed")
            
            #predict y train values and probabilities
            self.logger.log_info("predicting the train data started")
            y_train_pred = classifier.predict(X_train)
            #Lets select the probability of class 1 i.e., second column
            y_train_pred_prob = classifier.predict_proba(X_train)[:, 1]
            self.logger.log_info("predicting the train data completed")

            
            # Below function metric callings are for test data
            self.logger.log_info("Calling the confusion_matrix function by passing y_test and y_test_pred")
            cnf_matrix_test = confusion_matrix(y_test, y_test_pred)
            #Lets plot the curve for confusion matrix for the test set
            self.logger.log_info("Calling the plot_confusion_matrix function by passing cnf_matrix_test")
            self.evaluator.plot_confusion_matrix(cnf_matrix_test)  # Calling method via class instance
            #Lets see accuracy for test data
            self.logger.log_info("Calling the accuracy_score function by passing y_test and y_test_pred")
            accuracy_test = accuracy_score(y_test, y_test_pred)
            #Lets see best_threshold_value_test etc
            self.logger.log_info("Calling the get_best_threshold function by passing y_test and y_test_pred_prob")
            best_threshold_test, highest_f1_scores_test = self.evaluator.get_best_threshold(y_test, y_test_pred_prob)  # Calling method via class instance
            #Lets plot the precision recall curve for test data
            self.logger.log_info("Calling the plot_precision_recall_curve function by passing y_test and y_test_pred_prob")
            self.evaluator.plot_precision_recall_curve(y_test, y_test_pred_prob)  # Calling method via class instance
            #Lets calculate fpr,tpr,auc for the test set
            self.logger.log_info("Calling the plot_roc_curve function by passing y_test and y_test_pred_prob")
            fpr_test, tpr_test, auc_test = self.evaluator.plot_roc_curve(y_test, y_test_pred_prob)  # Calling method via class instance
            #Lets calculate mcc for the test set
            self.logger.log_info("Calling the calculate_mcc_with_threshold function by passing y_test, y_test_pred_prob, best_threshold_test")
            mcc_test = self.evaluator.calculate_mcc_with_threshold(y_test, y_test_pred_prob, best_threshold_test)  # Calling method via class instance
            #Lets print all the metrics for the test set
            self.logger.log_info("Calling the print_metrics function by passing auc_test, mcc_test, highest_f1_scores_test, cnf_matrix_test, best_threshold_test, model, accuracy_test, y_test, y_test_pred")
            print("Below are the test set metrics for the model",model)
            metrics_test = self.evaluator.print_metrics(auc_test, mcc_test, highest_f1_scores_test, cnf_matrix_test, best_threshold_test, model, accuracy_test, y_test, y_test_pred)  # Calling method via class instance
            
            
            # Below function metric callings are for train data
            self.logger.log_info("Calling the confusion_matrix function by passing y_train, y_train_pred")
            cnf_matrix_train = confusion_matrix(y_train, y_train_pred)
            #Lets plot the curve for confusion matrix for the train set
            self.logger.log_info("Calling the plot_confusion_matrix function by passing cnf_matrix_train")
            self.evaluator.plot_confusion_matrix(cnf_matrix_train)  # Calling method via class instance
            #Lets see accuracy for train data
            self.logger.log_info("Calling the accuracy_score function by passing y_train, y_train_pred")
            accuracy_train = accuracy_score(y_train, y_train_pred)
            #Lets see best_threshold_value_train etc
            self.logger.log_info("Calling the get_best_threshold function by passing y_train,  y_train_pred_prob")
            best_threshold_train, highest_f1_scores_train = self.evaluator.get_best_threshold(y_train,  y_train_pred_prob)
            #Lets plot the precision recall curve for train data
            self.logger.log_info("Calling the plot_precision_recall_curve function by passing y_train, y_train_pred_prob")
            self.evaluator.plot_precision_recall_curve(y_train, y_train_pred_prob) 
            #Lets calculate fpr,tpr,auc for the train set
            self.logger.log_info("Calling the plot_roc_curve function by passing y_train, y_train_pred_prob")
            fpr_train, tpr_train, auc_train = self.evaluator.plot_roc_curve(y_train, y_train_pred_prob)  # Calling method via class instance
            #Lets calculate mcc for the train set
            self.logger.log_info("Calling the calculate_mcc_with_threshold function by passing y_train, y_train_pred_prob, best_threshold_train")
            mcc_train = self.evaluator.calculate_mcc_with_threshold(y_train, y_train_pred_prob, best_threshold_train)  # Calling method via class instance
            #Lets print all the metrics for the train set
            self.logger.log_info("Calling the print_metrics function by passing auc_train, mcc_train,highest_f1_scores_train,cnf_matrix_train, best_threshold_train, model, accuracy_train, y_train, y_train_pred")
            print("Below are the train set metrics for the model",model)
            metrics_train=self.evaluator.print_metrics(auc_train, mcc_train,highest_f1_scores_train,cnf_matrix_train, best_threshold_train, model, accuracy_train, y_train, y_train_pred)  # Calling method via class instance

            self.logger.log_info("SMOTE imbalanced resampling is completed n result function in ModelTrainer class in train.py")
            return metrics_train,metrics_test,model,model_params
        except Exception as e:
            #Error occured while training the model
            self.logger.log_error("Error occured while training the model in result function in ModelTrainer class in train.py",e)
            raise RuntimeError("Error occured while training the model in result function in ModelTrainer class in train.py",e)
        
    def train_model(self, X_train, y_train, X_test, y_test,knn_params,random_forest_params,xgboost_params,logistic_params):
        try:
            # Train with KNeighborsClassifier
            self.logger.log_info("Calling result function by passing KNeighborsClassifier,knn_params, X_train, y_train, X_test, y_test")
            self.result(KNeighborsClassifier, knn_params, X_train, y_train, X_test, y_test)

            # Train with RandomForestClassifier
            self.logger.log_info("Calling result function by passing RandomForestClassifier, rf_params, X_train, y_train, X_test, y_test")
            self.result(RandomForestClassifier, random_forest_params,X_train, y_train, X_test, y_test)

            # Train with XGBClassifier
            self.logger.log_info("Calling result function by passing XGBClassifier, xgb_params, X_train, y_train, X_test, y_test")
            self.result(XGBClassifier, xgboost_params,X_train, y_train, X_test, y_test)

            # Train with LogisticRegression
            self.logger.log_info("Calling result function by passing SGDClassifier, lr_params, X_train, y_train, X_test, y_test")
            self.result(SGDClassifier, logistic_params,X_train, y_train, X_test, y_test)

        except Exception as e:
            #Error occured while training the model
            self.logger.log_error("Error occured while training the model in train_model in ModelTrainer class in train.py",e)
            raise RuntimeError("Error occured while training the model in train_model in ModelTrainer class in train.py",e)




