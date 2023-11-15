# Credit Risk Classification

## Overview
In this challenge, the goal is to build a model that evaluates loan risk and borrower creditworthiness. Utilizing data science and machine learning techniques, I'll create a precise binary classification model to categorize borrowers into two distinct groups.

## Features
## Data Preprocessing
- I initiate the data preprocessing by importing the "lending_data.csv" dataset from the "Resources" folder and converting it into a Pandas DataFrame.
- The labels (y) are created from the "loan_status" column, where 0 signifies a healthy loan, and 1 indicates a high risk of default.
- The features (X) are constructed from the remaining dataset columns.
- I optimize model development by splitting the data into training and testing sets using the train_test_split function from scikit-learn.
## Model Development
- I employ the scikit-learn's LogisticRegression class to construct a logistic regression model.
- The model is honed through training, utilizing the features (X_train) and corresponding labels (y_train).
- Subsequently, predictions on the testing data (X_test) are generated to evaluate the model's predictive capabilities.
- In consideration of potential data imbalance, I explore implementing a resampling technique.
## Model Evaluation
- I assess the model's performance by employing essential evaluation metrics, including accuracy, precision, recall, and F1-score.

## Results

### Machine learning model 1
Accuracy: Our model demonstrates a strong accuracy level of 0.99, correctly categorizing around 99% of all loans in our dataset. Nevertheless, it's crucial to be cautious about relying solely on high accuracy when dealing with imbalanced datasets, necessitating consideration of other evaluation metrics.

Precision: Our model exhibits a precision of approximately 0.85. This signifies that when our model identifies a loan as "high-risk," it is accurate approximately 85% of the time. In simpler terms, the model maintains a relatively low false positive rate, rendering it fairly dependable when predicting a loan's risk.

Recall (Sensitivity): Our model's recall stands at around 0.88. This implies that our model is proficient at capturing 89% of the actual high-risk loans within the dataset. It boasts a relatively low false negative rate, indicating that it doesn't overlook many high-risk loans.

### Machine learning model 2
Accuracy: Our secondary model boasts an exceptionally high accuracy of 0.994, signifying that it accurately classifies approximately 99.4% of all loans within our dataset. This exceptional accuracy rate underscores the model's outstanding overall performance.

Precision: The precision of our secondary model stands at around 0.994. This implies that when our model identifies a loan as "high-risk," it is accurate approximately 99.4% of the time. The model maintains an exceedingly low false positive rate, which reinforces its high reliability in flagging loans as risky.

Recall (Sensitivity): The recall of our secondary model is approximately 0.994. This demonstrates that our model effectively captures 99.4% of the actual high-risk loans in the dataset. It achieves an extremely low false negative rate, indicating its rare instances of missing high-risk loans.

## Summary
The secondary model surpasses the primary model in terms of accuracy, precision, and recall. It attains nearly flawless scores across these three metrics, highlighting its exceptional capability in distinguishing between "healthy" and "high-risk" loans. When we compare the results with those of the primary model, it's evident that this improved performance is a direct outcome of achieving data balance.

While the initial model did exhibit a respectable level of accuracy and remains effective in identifying risky loans, the secondary model is close to perfect precision and recall values for class 1 establish its exceptional reliability, making it a top choice for minimizing false negatives, or instances where it misses identifying high-risk loans.
