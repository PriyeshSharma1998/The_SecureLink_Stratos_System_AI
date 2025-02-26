import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score,recall_score,precision_recall_curve, roc_curve, roc_auc_score, f1_score, matthews_corrcoef, accuracy_score

class ModelEvaluation:
    def __init__(self, logger):
        #Initialise the ModelEvaluation class
        self.logger = logger
        self.logger.log_info("ModelEvaluation class is initialised in __init__ function in evaluate.py")

    def plot_confusion_matrix(self, cnf_matrix):
        try:
            #Plotting the confusion matrix using heatmpa
            print("Plotting the confusion matrix using heatmap started")
            self.logger.log_info("Plotting the confusion matrix using heatmap started")
            sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
            plt.ylabel('Actual Label')
            plt.xlabel('Predicted Label')
            plt.title('Confusion Matrix')
            plt.show()
            print("Plotting the confusion matrix using heatmap completed")
            self.logger.log_info("Plotting the confusion matrix using heatmap completed")
        except Exception as e:
            #Error occured while plotting the confusion matrix
            self.logger.log_error("Error occured while plotting the confusion matrix plot in plot_confusion_matrix function in  ModelEvalution class in evaluate.py",e)
            raise RuntimeError("Error occured while plotting the confusion matrix plot in plot_confusion_matrix function in  ModelEvalution class in evaluate.py",e)


    def get_best_threshold(self, y, y_pred_prob):
        try:
            print("Estimating the best_threshold and highest_f1_score is started")
            self.logger.log_info("Estimating the best_threshold and highest_f1_score is started")
            precision, recall, thresholds = precision_recall_curve(y, y_pred_prob)
            f1_scores_calculation = 2 * (precision * recall) / (precision + recall)
            # Find the index of the highest F1 score
            best_index = np.argmax(f1_scores_calculation)
            # Extract the best threshold and highest F1 score
            best_threshold = thresholds[best_index]
            highest_f1_score = f1_scores_calculation[best_index]
            print("Estimating the best_threshold and highest_f1_score is completed")
            self.logger.log_info("Estimating the best_threshold and highest_f1_score is completed")
            return best_threshold, highest_f1_score
        except Exception as e:
            #Error occured while calculating best_threshold, highest_f1_score
            self.logger.log_error("Error occured while calculating best_threshold, highest_f1_score in get_best_threshold function in  ModelEvalution class in evaluate.py",e)
            raise RuntimeError("Error occured while calculating best_threshold, highest_f1_score in get_best_threshold function in  ModelEvalution class in evaluate.py",e)

        

    def plot_precision_recall_curve(self, y, y_pred_prob):
        try:
            #Plotting the plot_precision_recall_curve using heatmap started
            print("Plotting the plot_precision_recall_curve using heatmap started")
            self.logger.log_info("Plotting the plot_precision_recall_curve using heatmap started")
            precision, recall, _ = precision_recall_curve(y, y_pred_prob)
            plt.plot(precision, recall)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.show()
            print("Plotting the plot_precision_recall_curve using heatmap completed")
            self.logger.log_info("Plotting the plot_precision_recall_curve using heatmap completed")
        except Exception as e:
            #Error occured while plotting the plot_precision_recall_curve
            self.logger.log_error("Error occured while plotting the plot_precision_recall_curve in plot_precision_recall_curve function in  ModelEvalution class in evaluate.py",e)
            raise RuntimeError("Error occured while plotting the plot_precision_recall_curve in plot_precision_recall_curve function in  ModelEvalution class in evaluate.py",e)
        

    def plot_roc_curve(self, y, y_pred_prob):
        try:
            #Calculating auc and plotting the roc curve
            fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
            auc = roc_auc_score(y, y_pred_prob)
            self.logger.log_info("auc fpr tpr values are",auc,fpr,tpr)
            
            print("Plotting the plot_roc_curve using started")
            self.logger.log_info("Plotting the plot_roc_curve using started")
            plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc:.2f})')
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.xlabel('False Positive Rate (FPR)')
            plt.ylabel('True Positive Rate (TPR)')
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.show()
            print("Plotting the plot_roc_curve using completed")
            self.logger.log_info("Plotting the plot_roc_curve using completed")
            return fpr, tpr, auc
        except Exception as e:
            #Error occured while calculating auc and plotting the roc curve
            self.logger.log_error("Error occured while calculating auc and plotting the roc curve in plot_roc_curve function in  ModelEvalution class in evaluate.py",e)
            raise RuntimeError("Error occured while calculating auc and plotting the roc curve in plot_roc_curve function in  ModelEvalution class in evaluate.py",e)
        


    def calculate_mcc(self, y, y_pred):
        try:
            return matthews_corrcoef(y, y_pred)
        except Exception as e:
            #Error occured while calculating mcc
            self.logger.log_error("Error occured while calculating mcc in calculaye_mcc function in  ModelEvalution class in evaluate.py",e)
            raise RuntimeError("Error occured while calculating mcc in calculaye_mcc function in  ModelEvalution class in evaluate.py",e)
        


    def calculate_mcc_with_threshold(self, y, y_pred_prob, threshold):
        try:
            print("Calculating mcc")
            self.logger.log_info("Calculating mcc")
            y_pred_thresholded = (y_pred_prob >= threshold).astype(int)
            mcc = matthews_corrcoef(y, y_pred_thresholded)
            self.logger.log_info("mcc has been calculated and the value is", mcc)
            return mcc
        except Exception as e:
            #Error occured while calculating mcc
            self.logger.log_error("Error occured while calculating mcc in calculaye_mcc function in  ModelEvalution class in evaluate.py",e)
            raise RuntimeError("Error occured while calculating mcc in calculaye_mcc function_with_threshold in in  ModelEvalution class in evaluate.py",e)
    
    def calculate_sensitivity_specificity(self, cnf_matrix):
        try:
            self.logger.log_info("Calculating sensitivity and specificity")
            tn, fp, fn, tp = cnf_matrix.ravel()
            sensitivity = tp / (tp + fn)  # True Positive Rate (Recall)
            specificity = tn / (tn + fp)  # True Negative Rate
            self.logger.log_info("sensitivity and specificity have been calculated and the values are",sensitivity,specificity)
            return sensitivity, specificity
        except Exception as e:
            #Error occured while calculating sensitivity and specificity
            self.logger.log_error("Error occured while calculating sensitivity and specificity in calculate_sensitivity_specificity function  in  ModelEvalution class in evaluate.py",e)
            raise RuntimeError("Error occured while calculating sensitivity and specificity in calculate_sensitivity_specificity function  in  ModelEvalution class in evaluate.py",e)

    def print_metrics(self, auc, mcc, highest_f1_scores_train, cnf_matrix, best_threshold, model, accuracy, y, y_pred):
        try:
            self.logger.log_info("Calling the calculate_sensitivity_specificity by passing cnf_matrix")
            sensitivity, specificity = self.calculate_sensitivity_specificity(cnf_matrix)
            metrics = {
                'Model': str(model),
                'AUC': auc,
                'MCC': mcc,
                'Best Threshold': best_threshold,
                'F1 Scores': highest_f1_scores_train,
                'Confusion Matrix': cnf_matrix,
                'Accuracy': accuracy,
                'Sensitivity': sensitivity,
                'Specificity': specificity,
                  }
            self.logger.log_info("Metrics for the model",model)
            for key, value in metrics.items():
                self.logger.log_info(key,":",value)
            return metrics
        except Exception as e:
            #Error occured while printing metrics
            self.logger.log_error("Error occured while printing metrics in print_metrics function in  ModelEvalution class in evaluate.py",e)
            raise RuntimeError("Error occured while printing metrics in print_metrics function in  ModelEvalution class in evaluate.py",e)