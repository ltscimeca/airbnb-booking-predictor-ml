# Airbnb Booking Predictor

A machine learning model that predicts whether a host on Airbnb qualifies as a **Superhost**, based on listing activity, host behavior, and platform engagement data.

# ☆ Overview

This project applies a full machine learning workflow to predict Superhost status on Airbnb.
Using the CSV data, we cleaned, encoded, and engineered features, trained a logistic regression model, evaluated classification performance, and exported the model for reuse.

# ✮ Objectives

- Predict 'host_is_superhost' using logistic regression
- Apply structured data cleaning, encoding, and feature scaling
- Train and evaluate a classification model on labeled Airbnb data
- Save the trained model for future predictions

# ⊹ Tools & Technologies

- Python, NumPy, Pandas
- scikit-learn (LogisticRegression, StandardScaler, train_test_split, classification_report, roc_auc_score, joblib)

# ✮ Model Performance

Achieved strong classification performance:
- **Train Accuracy:** 87%
- **Test Accuracy:** 88%
- **ROC AUC Score:** 0.89
Highlights:
- Dropped irrelevant or high-missing columns
- One-hot encoded categorical variables
- Scaled numerical features with 'StandardScaler'
- Used 'class_weight='balanced'' to address label imbalance
- Evaluated with precision, recall, F1-score, and ROC curve

# ☆ File Structure

| File                                      | Purpose                                                 |
|-------------------------------------------|---------------------------------------------------------|
| ModelSelectionForLogisticRegression.ipynb | Full Jupyter notebook with code, output, and commentary |
| airbnbData_train.csv                      | Airbnb dataset used for model training                  |
| best_model.pkl                            | Trained and exported logistic regression model          |

# ✮ Key Takeaways

Even simple linear models, such as linear regression, can perform well when paired with thoughtful feature engineering and preprocessing.
This project highlights the importance of model interpretability in domains where transparency is valued, like host evaluations.

# ⊹ Future Work

- Apply 'GridSearchCV' for logistic regression hyperparameter tuning
- Compare performance with ensemble methods
- Visualize coefficients and explore SHAP for deeper insight
- Build a simple UI for hosts to check their Superhost probability

# ☆ Program Info

This project was completed as part of the **Break Through Tech AI Program** (Summer 2025).
