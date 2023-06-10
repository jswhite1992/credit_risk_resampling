# credit_risk_resampling

## Required Libraries
This project requires Python along with the following libraries installed:
* numpy
* pandas
* pathlib
* sklearn
* imbalanced-learn

## Overview of the Analysis
The purpose of this analysis is to build a machine learning model that can accurately predict the creditworthiness of borrowers. We're using a dataset from a peer-to-peer lending services company to assess credit risk, which is an inherently imbalanced classification problem, as healthy loans significantly outnumber risky loans.

We first use a Logistic Regression Model on the original data. We then handle the imbalanced class issue by oversampling the data using the RandomOverSampler module from the imbalanced-learn library. 

## Results

### Logistic Regression Model with Original Data
* Balanced Accuracy Score: 0.95
* Precision: 0.99 (healthy loan), 0.85 (high-risk loan)
* Recall: 0.99 (healthy loan), 0.91 (high-risk loan)

### Logistic Regression Model with Oversampled Data
* Balanced Accuracy Score: 0.99
* Precision: 0.99 (healthy loan), 0.84 (high-risk loan)
* Recall: 0.99 (healthy loan), 0.99 (high-risk loan)

## Summary
In comparison, the logistic regression model with oversampled data has a better balanced accuracy score, precision, and recall. Specifically, the recall of high-risk loans (which is a critical metric in credit risk model) significantly improved with oversampled data, making it a more reliable model for predicting credit risk. 

Despite slightly lower precision for high-risk loans (class 1) in the oversampled model, the model's ability to catch almost all the high-risk loans with a recall of 0.99 makes it preferable. This is because, in the credit risk context, false negatives (predicting a loan as healthy when it is actually high-risk) are generally more costly than false positives (predicting a loan as high-risk when it is actually healthy).

Therefore, for the task of credit risk classification, I recommend using the logistic regression model with oversampled data.
