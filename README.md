# Predicting Credit Card Approvals

This project builds a machine learning model to predict whether a credit card application will be approved or denied. It uses the Credit Card Approval dataset from the UCI Machine Learning Repository. The workflow includes data preprocessing, exploratory data analysis, feature selection, and model training using Logistic Regression with hyperparameter tuning.

## Project Overview

Commercial banks receive a lot of applications for credit cards. Many of them get rejected for many reasons, like high loan balances, low income levels, or too many inquiries on an individual's credit report, for example. Manually analyzing these applications is mundane, error-prone, and time-consuming (and time is money!). Luckily, this task can be automated with the power of machine learning and pretty much every commercial bank does so nowadays. In this project, we will build an automatic credit card approval predictor using machine learning techniques, just like the real banks do.

## Key Features

*   **Data Preprocessing:**
    *   Handling missing values (imputing with mean for numeric and mode for categorical variables).
    *   Encoding categorical variables using one-hot encoding.
    *   Feature scaling using `StandardScaler`.
*   **Model Training:**
    *   Implementation of a Logistic Regression classifier.
*   **Model Optimization:**
    *   Hyperparameter tuning using `GridSearchCV` to find the best `tol` and `max_iter` parameters.
*   **Evaluation:**
    *   Performance evaluation using accuracy score and confusion matrix.

## Dependencies

To run this project, you need the following Python libraries:

*   pandas
*   numpy
*   scikit-learn

You can install them using pip:

```bash
pip install pandas numpy scikit-learn
```

## Dataset

The project uses the `cc_approvals.data` file, which contains the credit card application data. The dataset has been anonymized to protect user privacy.

## Usage

1.  Clone the repository:
    ```bash
    git clone https://github.com/RudraJad2005/Predicting-Credit-Card-Approvals.git
    ```
2.  Navigate to the project directory:
    ```bash
    cd Predicting-Credit-Card-Approvals
    ```
3.  Ensure the dataset file `cc_approvals.data` is present in the directory.
4.  Run the main script:
    ```bash
    python main.py
    ```

## Results

The script outputs the confusion matrix for the training set, the best hyperparameters found during grid search, and the final accuracy of the optimized Logistic Regression model on the test set.
