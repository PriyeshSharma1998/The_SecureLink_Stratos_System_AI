import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif
import logging

class DataPreprocessor:
    def __init__(self,logger):
        #Initialise the datapreprocessor class
        self.logger = logger
        self.data=None
        self.logger.log_info("Datapreprocessor class is initialised in __init__ function in preprocessor.py")

    def load_data(self, file_path):
        #Lets load the data from the csv file
        try:
            #Data load started from csv file
            self.logger.log_info("data load is started in load_data function in DataPreprocessor class in preprocessor.py")
            data = pd.read_csv(file_path)
            #Data load completed from csv file
            self.logger.log_info("data load is completed in load_data function in DataPreprocessor class in preprocessor.py")
            return data
        except Exception as e:
            #Error occured while loading the data
            self.logger.log_error("Error occured while loading the data in load_data function in DataPreprocessor class in preprocessor.py",e)
            raise RuntimeError("Error occured while loading the data in load_data function in DataPreprocessor class in preprocessor.py",e)

    def preprocess_data(self,data):
        #Lets preprocess the data which was loaded
        try:
            #preprocessing the data is started in preprocess_data function in DataPreprocessor class in preprocessor.pypreprocessing the data is started in preprocess_data function in DataPreprocessor class in preprocessor.py
            print("preprocessing the data is started in preprocess_data function in DataPreprocessor class in preprocessor.py")
            self.logger.log_info("preprocessing the data is started in preprocess_data function in DataPreprocessor class in preprocessor.py")

            # Printing the top 5 rows of the dataset
            self.logger.log_info("Capturing top 5 rows in the dataset")
            print("Top 5 rows", data.head())
            
            # Printing the bottom 5 rows of the dataset
            self.logger.log_info("Capturing bottom 5 rows in the dataset")
            print("Bottom 5 rows", data.tail())
            
            # Printing the info of the dataset
            self.logger.log_info("Capturing info of the dataset is")
            print("Data Info", data.info())
            
            # Printing the shape of the dataset
            self.logger.log_info("Shape of the dataset is",data.shape)
            print("Data Shape", data.shape)
            
            # Printing the columns of the dataset
            self.logger.log_info("Colunmns of the dataset are",data.columns)
            print("Columns in Data", data.columns)

            # Printing the describe of the dataset
            self.logger.log_info("Describe of the dataset is",data.describe().T)
            print("Data Description:", data.describe().T)

            # Checking missing values and visualize using heatmap
            print("Checking missing values and visualize using heatmap started")
            self.logger.log_info("Checking missing values and visualize using heatmap started")
            plt.figure(figsize=(12,7))
            sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
            plt.title("Missing Values Presence Across Dataset")
            plt.xlabel('Features')
            plt.ylabel('Density')
            plt.show()
            print("Checking missing values and visualize using heatmap completed")
            self.logger.log_info("Checking missing values and visualize using heatmap completed")
            
            # Checking missing count
            missing_values = data.isnull().sum()
            print("Missing values", missing_values)
            self.logger.log_info("Missing values are", missing_values)
            if missing_values.any():
                # There are missing values in the dataset
                print("There are missing values in the dataset")
                self.logger.log_warning("There are missing values in the dataset.")
            else:
                #There are no missing values in the dataset
                print("There are no missing values in the dataset")
                self.logger.log_info("There are no missing values in the dataset.")

            # Checking for outliers using IQR
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = (data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))
            self.logger.log_info("Outliers are", missing_values)

            # Checking outlier and visualize using heatmap
            print("Checking outliers and visualize using heatmap started")
            self.logger.log_info("Checking outliers and visualize using heatmap started")
            plt.figure(figsize=(12, 6))
            sns.heatmap(outliers, cmap="coolwarm", cbar=False)
            plt.title("Outlier Presence Across Dataset")
            plt.xlabel('Features')
            plt.ylabel('Density')
            plt.show()
            print("Checking outliers and visualize using heatmap completed")
            self.logger.log_info("Checking outliers and visualize using heatmap completed")

            # Visualize class feature frequency
            value_counts = data['Class'].value_counts()
            print("Class Feature Value Counts", value_counts)
            self.logger.log_info("Class Feature Value Counts",value_counts)
            
            # Count plot for 'Class' feature
            print("Checking frequency of Class feature and visualize using countplot started")
            self.logger.log_info("Checking frequency of Class feature and visualize using countplot started")
            ax = sns.countplot(x='Class', data=data)
            plt.title('Number of Fraud Cases') 
            plt.xlabel('Class')
            plt.ylabel('Frequency') 
            ax.set_xticklabels(['No Fraud', 'Fraud'])
            plt.show()
            print("Checking frequency of Class feature and visualize using countplot completed")
            self.logger.log_info("Checking frequency of Class feature and visualize using countplot completed")
            
            # Pie chart for class feature
            print("Checking frequency of Class feature and visualize using piechart started")
            self.logger.log_info("Checking frequency of Class feature and visualize using piechart started")
            data['Class'].value_counts().plot.pie(autopct='%1.1f%%', labels=['No Fraud', 'Fraud'])
            plt.title('Class Distribution')
            plt.show()
            print("Checking frequency of Class feature and visualize using piechart completed")
            self.logger.log_info("Checking frequency of Class feature and visualize using piechart completed")

            # Checking the features using correlation analysis for feature selection using heatmap
            print("Checking the features using correlation analysis using heatmap started")
            self.logger.log_info("Checking the features using correlation analysis using heatmap started")
            correlation_matrix = data.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
            plt.title('Correlation Matrix')
            plt.show()
            print("Checking the features using correlation analysis using heatmap completed")
            self.logger.log_info("Checking the features using correlation analysis using heatmap completed")

            #It is unable to understand the above analysis, so checking each feature against target feature
            #Checking the features using correlation analysis using heatmap against target feature
            print("Checking the features using correlation analysis using heatmap against target feature started")
            self.logger.log_info("Checking the features using correlation analysis using heatmap against target feature started")
            target_corr = correlation_matrix[['Class']]
            plt.figure(figsize=(8, 6))
            sns.heatmap(target_corr, annot=True, cmap='coolwarm', cbar=True, vmin=-1, vmax=1)
            plt.title('Correlation with Target')
            plt.show()
            print("Checking the features using correlation analysis using heatmap against target feature completed")
            self.logger.log_info("Checking the features using correlation analysis using heatmap against target feature completed")

            #Checking the features using ANOVA F-test for feature selection since it is between categorical vs numerical features
            #Creating the dataset by excluding the 'class' feature
            features = data.drop(columns=['Class'])
            #Creating the target feature using dataset
            target = data['Class']
            #Fetching the F_values and p_values using f_classif
            F_values, p_values = f_classif(features, target)
            #Storing the anova results in anova_results which is dataframe
            anova_results = pd.DataFrame({"Feature": features.columns, "F-value": F_values, "p-value": p_values})
            #Storing the anova results in anova_results_sorted by sorting
            anova_results_sorted = anova_results.sort_values(by="F-value", ascending=False)
            #Checking the features using ANOVA F-test for feature selection

            #Checking the features using ANOVA F-test using heatmap against target feature started
            print("Checking the features using ANOVA F-test using heatmap against target feature started")
            self.logger.log_info("Checking the features using ANOVA F-test using heatmap against target feature started")
            plt.figure(figsize=(8, 12))
            sns.heatmap(anova_results_sorted[['F-value', 'p-value']], annot=True, cmap="viridis", fmt=".2f", linewidths=0.4, linecolor='black', cbar=True, yticklabels=anova_results_sorted["Feature"])
            plt.title('ANOVA F-test Results', fontsize=14)
            plt.tight_layout()
            plt.show()
            #Checking the features using ANOVA F-test using heatmap against target feature completed
            print("Checking the features using ANOVA F-test using heatmap against target feature completed")
            self.logger.log_info("Checking the features using ANOVA F-test using heatmap against target feature completed")

            
            # Storing two datasets in two variables, one is anova dataset and other one is correlation analysis dataset
            
            
            #Storing the highly correlated feastures in preprocessed_data_correlation_analysis
            preprocessed_data_correlation_analysis = data[['V3','V4','V7','V10','V11','V12','V14','V16','V17','Class']].copy(deep = True)
            print("Storing the highly correlated feastures in preprocessed_data_correlation_analysis",preprocessed_data_correlation_analysis.columns)
            self.logger.log_info("Storing the highly correlated feastures in preprocessed_data_correlation_analysis",preprocessed_data_correlation_analysis.columns)
            
            #Storing the top 20 features in preprocessed_data_correlation_anova
            preprocessed_data_anova = data.copy(deep = True)
            print("Storing the top 20 features in preprocessed_data_anova",preprocessed_data_anova.columns)
            self.logger.log_info("Storing the top 20 features in preprocessed_data_anova",preprocessed_data_anova.columns)
            columns=list(anova_results_sorted.index[20:])
            #Converting indices to column names
            indices_to_drop = list(anova_results_sorted.index[20:])
            #Creating the logic for removing the columns
            column_names_to_drop=[]
            for i in indices_to_drop:
                column_names_to_drop.append(preprocessed_data_anova.columns[i])
            print("columns to be removed",columns)
            self.logger.log_info("Columns to be removed",columns)
            preprocessed_data_anova.drop(columns = column_names_to_drop,inplace = True)
            print("preprocessed_data_anova after dropping columns",preprocessed_data_anova)
            self.logger.log_info("preprocessed_data_anova after dropping columns",preprocessed_data_anova)

            self.logger.log_info("preprocessing the data is completed in preprocess_data function in DataPreprocessor class in preprocessor.py")
            return preprocessed_data_correlation_analysis, preprocessed_data_anova

        except Exception as e:
            self.logger.log_error("Error occured while preprocessing the data in preprocess_data function in DataPreprocessor class in preprocessor.py",e)
            raise RuntimeError("Error occured while preprocessing the data in preprocess_data function in DataPreprocessor class in preprocessor.py",e)