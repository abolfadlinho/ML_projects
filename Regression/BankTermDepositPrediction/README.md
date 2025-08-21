# Bank Term Deposit Prediction

This project predicts whether a bank customer will subscribe to a term deposit using various machine learning algorithms. The workflow includes data loading, exploratory data analysis (EDA), feature engineering, preprocessing, model training, and evaluation.

## Workflow Overview

1. **Data Loading & EDA**

   - Loads train and test datasets.
   - Inspects data samples, info, and distributions for features like age, job, marital status, education, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome, and target column.
   - Handles missing/unknown values and outliers (e.g., replaces 'unknown' with 'others', drops outliers in balance, drops low-value features).
   - Visualizes distributions and relationships using seaborn and matplotlib.

2. **Feature Engineering & Preprocessing**

   - Drops irrelevant or low-value columns (e.g., 'default', 'previous', 'poutcome').
   - Encodes categorical features using LabelEncoder.
   - Splits data into features (X) and target (y).
   - Scales features if needed.

3. **Model Training & Evaluation**
   - Trains and evaluates multiple classifiers:
     - Logistic Regression
     - SVC
     - K-Nearest Neighbors
     - Decision Tree
     - Random Forest
     - XGBoost
     - CatBoost
     - LightGBM
   - Prints train/test scores, confusion matrices, and classification reports for each model.

## Key Code Snippet: Model Training Example

```python
# Logistic Regression model
lr = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)
lr.fit(X_train, y_train)
print(lr.score(X_train, y_train))
print(lr.score(X_test, y_test))
y_pred = lr.predict(X_test)
print(classification_report(y_test, y_pred))
```

## Requirements

- Python
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib
- xgboost
- catboost
- lightgbm

## Usage

1. Install dependencies:
   ```powershell
   pip install pandas numpy scikit-learn seaborn matplotlib xgboost catboost lightgbm
   ```
2. Run the notebook `bank-term-deposit-prediction.ipynb` for step-by-step execution.

## References

- [Bank Marketing Dataset](https://www.kaggle.com/datasets)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)

---

For details, see the notebook: `bank-term-deposit-prediction.ipynb`.
