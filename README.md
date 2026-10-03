# CreditSense

### Machine Learning-Based Loan Approval Prediction

CreditSense is a machine learning project that predicts loan approval outcomes based on applicant financial, demographic, and loan-related attributes.

The project follows a complete supervised machine learning workflow, including data preprocessing, exploratory data analysis, feature encoding, correlation analysis, feature scaling, model training, evaluation, and feature engineering.

## Project Overview

The objective of this project is to build and evaluate classification models capable of predicting whether a loan application will be approved.

The dataset contains applicant information such as:

- Applicant and coapplicant income
- Employment status
- Age
- Marital status
- Number of dependents
- Credit score
- Existing loans
- Debt-to-income ratio
- Savings
- Collateral value
- Loan amount and term
- Loan purpose
- Property area
- Education level
- Gender
- Employer category

## Machine Learning Workflow

The project follows these major steps:

1. **Data Loading & Preprocessing**
   - Removed irrelevant features
   - Handled missing numerical values using mean imputation
   - Handled missing categorical values using mode imputation

2. **Exploratory Data Analysis**
   - Analyzed distributions and categorical variables
   - Examined relationships between features and loan approval
   - Performed correlation analysis

3. **Feature Encoding**
   - Applied Label Encoding to categorical target/ordinal features
   - Applied One-Hot Encoding to categorical variables

4. **Feature Scaling**
   - Applied `StandardScaler` to normalize input features
   - Used an 80/20 train-test split

5. **Model Development**
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Gaussian Naive Bayes

6. **Model Evaluation**
   - Evaluated models using:
     - Precision
     - Recall
     - F1-score
     - Accuracy

7. **Feature Engineering**
   - Created squared features for:
     - DTI Ratio
     - Credit Score

## Models

### Logistic Regression

A linear classification model used to estimate the probability of loan approval based on the applicant's features.

### K-Nearest Neighbors

A distance-based classification algorithm configured with `k = 5` to classify loan applications based on neighboring observations.

### Gaussian Naive Bayes

A probabilistic classification algorithm based on Bayes' theorem with a Gaussian distribution assumption for continuous features.

## Model Performance

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 86% |
| K-Nearest Neighbors | 76% |
| Gaussian Naive Bayes | 86% |

The models were evaluated on a test set of 200 samples using classification metrics.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Project Structure

```text
CreditSense/
│
├── main_file.ipynb
├── loan_approval_data.csv
└── README.md
