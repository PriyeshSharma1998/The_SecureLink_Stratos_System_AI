Project Title:SecureLink Stratos System

Detecting fraudulent credit card transactions using advanced machine learning techniques.


Table of Contents
1) Overview
2) Key Features
3) Dataset Details
4) Approach
5) Model Evaluation Summary
6) Key Results
7) Installation
8) Usage
9) Future Improvements
10) Contact


1) Overview:

This project aims to detect fraudulent credit card transactions using machine learning techniques. The dataset was preprocessed to handle missing values, detect outliers, and perform feature selection through correlation analysis and ANOVA. Several machine learning models were trained, and hyperparameter tuning was applied to optimize performance. The final model, Logistic Regression with an SGDClassifier, was fine-tuned using GridSearchCV and deployed for real-time usage.

2) Key Features:
    a) Data Preprocessing:
        Missing values: Visualized using heatmaps (no missing values found).
        Outliers: Visualized using IQR heatmaps; outliers were retained due to their relevance in detecting fraudulent transactions.
        Feature Selection:
            Correlation Analysis: Removed highly correlated features.
            ANOVA: Assessed feature significance.
        Data Split: The dataset was split into training (80%) and testing (20%) sets.
    b) Model Building:
        Models Evaluated:
           RandomForestClassifier
           XGBClassifier
           LogisticRegression
           KNNClassifier
        Final Model: Logistic Regression with SGDClassifier selected after extensive evaluation
    c) Hyperparameter Optimization:
       Used GridSearchCV to fine-tune the final model.
    d) Evaluation Metrics:
       Evaluated using:
          Precision
          Recall
          F1-Score
          ROC-AUC
          MCC
          Sensitivity
          Specificity
      Threshold Selection: Applied a custom function to determine the optimal threshold based on the precision-recall curve, optimizing the F1-Score.
    

3) Dataset Details:
    a) Source: Internal company dataset
    b) Type: Structured/Tabular
    c) Size: 284,808 rows, 31 features
    d) Features: Time-related features, anonymized variables (V1 to V28), transaction amount, and fraud label (Class)
    e) Class Imbalance: Approximately 90:10 (Non-fraud vs. Fraud)
    f) Imbalance Handling: Techniques like SMOTE and evaluation metrics like Sensitivity, AUC, MCC, and F1-Score were used to address the imbalance.

4) Approach:
    a) Data Preprocessing: 
          Missing Values: Heatmaps were used to check for missing values (none found).
          Outliers: Visualized using IQR in heatmaps but retained as they may signify fraudulent transactions.
          Feature Selection:
                Correlation Analysis: Identified and removed highly correlated features.
                ANOVA: Used to assess the significance of features.
          Data Split: The dataset was split into training (80%) and testing (20%) sets.
    b) Model Selection:
    Models were evaluated based on performance and suitability for handling class imbalance:
      RandomForestClassifier
      XGBClassifier
      LogisticRegression
      KNNClassifier
      Final Model: Logistic Regression with SGDClassifier chosen after extensive evaluation.
    c) Hyperparameter Optimization:
      GridSearchCV was used for hyperparameter tuning to enhance the model's performance.
    d) Evaluation:
      Precision, Recall, F1-Score, ROC-AUC, MCC, Sensitivity, and Specificity were used to evaluate models.
      Threshold Optimization: A custom function optimized the threshold based on the precision-recall curve, resulting in a significant F1-Score improvement.

5) Model Evaluation Summary:
  

| Model                                                      | Metric            | Test Set       | Train Set      |
|------------------------------------------------------------|-------------------|----------------|----------------|
| K-Nearest Neighbors (KNN)(without smote with correlation)  | AUC               | 0.9226         | 0.9999         |
|                                                            | MCC               | 0.8621         | 0.8779         |
|                                                            | Best Threshold    | 0.6            | 0.6            |
|                                                            | F1-Score          | 0.8596         | 0.8748         |
|                                                            | Accuracy          | 99.96%         | 99.96%         |
|                                                            | Sensitivity       | 0.7967         | 0.8049         |
|                                                            | Specificity       | 0.9999         | 0.9999         |
| Random Forest(with smote with anova)                       | AUC               | 0.9532         | 1.0            |
|                                                            | MCC               | 0.8341         | 1.0            |
|                                                            | Best Threshold    | 0.75           | 0.75           |
|                                                            | F1-Score          | 0.8304         | 1.0            |
|                                                            | Accuracy          | 99.94%         | 100%           |
|                                                            | Sensitivity       | 0.8130         | 1.0            |
|                                                            | Specificity       | 0.9997         | 1.0            |
| XGBoost(with smote with anova)                             | AUC               | 0.9793         | 1.0            |
|                                                            | MCC               | 0.8317         | 1.0            |
|                                                            | Best Threshold    | 0.9775         | 0.862          |
|                                                            | F1-Score          | 0.8297         | 1.0            |
|                                                            | Accuracy          | 99.89%         | 99.99%         |
|                                                            | Sensitivity       | 0.8537         | 1.0            |
|                                                            | Specificity       | 0.9992         | 0.9999         |
| SGD Classifier(with smote with anova)                      | AUC               | 0.9708         | 0.9900         |
|                                                            | MCC               | 0.8127         | 0.8921         |
|                                                            | Best Threshold    | 1.0            | 0.277          |
|                                                            | F1-Score          | 0.8120         | 0.9463         |
|                                                            | Accuracy          | 97.36%         | 94.31%         |
|                                                            | Sensitivity       | 0.8943         | 0.9122         |
|                                                            | Specificity       | 0.9737         | 0.9739         |

Observations and Recommendations:
  a) K-Nearest Neighbors (KNN):
    Best for balanced performance, suitable for scenarios where sensitivity is not a major priority.
    Sensitivity is slightly lower compared to other models, which is a consideration for fraud detection.
  b) Stochastic Gradient Descent (SGD):
    Best for higher sensitivity, crucial in fraud detection to catch more positive fraud cases.
    There is some overfitting as evidenced by the train-test discrepancy, but it still offers the highest sensitivity.
  c) Random Forest:
    Strong performance across all metrics, with slight overfitting observed (perfect train results).
    Performs well, especially in terms of specificity and accuracy.
  d) XGBoost:
    Performs exceptionally well with high accuracy and sensitivity.
    Minimal overfitting, as the train-test results are comparable.
Final Model Choice:
  a) Stochastic Gradient Descent (SGD): Recommended for fraud detection due to higher sensitivity despite some overfitting concerns.
  b) K-Nearest Neighbors (KNN): Can also be used if a more balanced performance is preferred with less emphasis on sensitivity.


6) Key Results:
    
    a) Baseline Models (With SMOTE + ANOVA)
    Test Set Metrics (SGDClassifier):
    
    | Metric         | Value       |
    |----------------|-------------|
    | AUC            | 0.9708      |
    | MCC            | 0.8127      |
    | Best Threshold | 0.9999999860|
    | F1-Score       | 0.812       |
    | Accuracy       | 97.36%      |
    | Sensitivity    | 89.43%      |
    | Specificity    | 97.37%      |

    Train Set Metrics (SGDClassifier):

    | Metric         | Value       |
    |----------------|-------------|
    | AUC            | 0.9900      |
    | MCC            | 0.8921      |
    | Best Threshold | 0.2770      |
    | F1-Score       | 0.946       |
    | Accuracy       | 94.31%      |
    | Sensitivity    | 91.22%      |
    | Specificity    | 97.39%      |
 
   b) Optimized Model (Logistic Regression with SGDClassifier):
    F1-Score: 0.814 (Improvement through GridSearchCV and threshold optimization)
   c) Threshold Adjustment and F1-Score Optimization:
     The threshold determines the cutoff probability for classifying a prediction as positive (fraud) or negative (non-fraud).
     Increasing the threshold reduces the probability of predicting a fraud (class 1), which increases Precision but decreases Recall.
     Decreasing the threshold increases the probability of predicting fraud, which improves Recall but reduces Precision.
     By adjusting the threshold and calculating both Precision and Recall, we can find the optimal threshold that maximizes the F1-Score, ensuring the best balance between both metrics.
    Best Threshold: 0.9999999850, improving F1-Score to 0.814.
   d) Final Model Performance:
    
    Test Set Metrics:
    
    | Metric         | Value       |
    |----------------|-------------|
    | AUC            | 0.974       |
    | MCC            | 0.818       |
    | Best Threshold | 0.9999999850|
    | F1-Score       | 0.814       |
    | Accuracy       | 97.42%      |
    | Sensitivity    | 89.70%      |
    | Specificity    | 97.38%      |
    
    Confusion Matrix (Test Set):
    
    [[69240  1839]  
     [   12   111]]

     Train Set Metrics:

     | Metric         | Value       |
     |----------------|-------------|
     | AUC            | 0.991       |
     | MCC            | 0.894       |
     | Best Threshold | 0.2770      |
     | F1-Score       | 0.948       |
     | Accuracy       | 94.35%      |
     | Sensitivity    | 91.30%      |
     | Specificity    | 97.39%      |

     Confusion Matrix (Train Set):

     [[207650   5556]  
      [ 18710 194526]]
    
    e) Model Deployment:
      The final model was deployed using the FastAPI web framework, which enables real-time prediction of fraudulent credit card transactions.
      The API allows for input of transaction data, processes it through the trained model, and returns a prediction of whether the transaction is fraudulent or not.
      The deployment is operational and can be accessed through a RESTful interface.


7) Installation:
Steps to Install and Run the Project:
  a) Clone the Repository:

     git clone https://github.com/PriyeshSharma1998/The_SecureLink_Stratos_System_AI.git
     cd The_SecureLink_Stratos_System_AI
  
  b) Install Dependencies:
     
     pip install -r requirements.txt
  
  c) Run the Project:
     
     python main.py

8) Usage:
  This project trains models using various algorithms like KNN,XGBoost,RandomForest,SGDClassifier and evaluates them using metrics like accuracy, confusion matrix, etc.
  The project also includes functionality to handle imbalanced datasets with SMOTE (Synthetic Minority Over-sampling Technique).
a) Train Models: 
   - The model will be trained based on your script settings. You can modify the script to select the model and whether to use SMOTE for balancing data.
b) Evaluate Models: 
   - The evaluation of models is done in the script itself and prints metrics like accuracy, confusion matrix, etc.


9) Future Improvements:
    a) Implement advanced hyperparameter tuning using Bayesian Optimization (e.g., Optuna).
    b) Implement neural networks model and check the metrics
    c) Expand the dataset to include more diverse samples, which can help improve the model’s generalization.

10) Contact:
If you have any questions or suggestions, feel free to reach out.

Name: Priyesh Kumar Sharma Tumbur  
Email: priyeshsharma812@gmail.com  
LinkedIn: [linkedin.com/in/priyesh-sharma-a11048270](https://linkedin.com/in/priyesh-sharma-a11048270)  
GitHub: [github.com/PriyeshSharma1998](https://github.com/PriyeshSharma1998)

"# The_SecureLink_Stratos_System_ML_Project" 
