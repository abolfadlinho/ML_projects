# Twitter Sentiment Analysis

This project demonstrates sentiment classification on Twitter data using machine learning and natural language processing techniques. The workflow includes data loading, preprocessing, model training, evaluation, and prediction.

## Workflow Overview

1. **Data Loading**

   - Reads the dataset `twitter_training.csv` with columns: `id`, `country`, `Label`, `Text`.

2. **Exploratory Data Analysis (EDA)**

   - Prints shape, info, and sample rows.
   - Shows label distribution and sample tweets.

3. **Preprocessing**

   - Removes missing values.
   - Uses spaCy (`en_core_web_sm`) for stopword removal and lemmatization.
   - Adds a new column `Preprocessed Text`.

4. **Label Encoding**

   - Encodes sentiment labels using `LabelEncoder`.

5. **Train-Test Split**

   - Splits data into training and test sets (80/20 split, stratified).

6. **Model Training & Evaluation**

   - Trains two models:
     - Naive Bayes (`MultinomialNB`)
     - Random Forest (`RandomForestClassifier`)
   - Uses `TfidfVectorizer` for feature extraction.
   - Evaluates models using accuracy and classification report.

7. **Prediction on Validation Data**
   - Loads `twitter_validation.csv`.
   - Preprocesses a sample tweet and predicts its sentiment.
   - Maps prediction to sentiment classes: Irrelevant, Natural, Negative, Positive.

## Requirements

- Python
- pandas
- numpy
- scikit-learn
- spaCy (`en_core_web_sm` model)

## Usage

1. Install dependencies:
   ```powershell
   pip install pandas numpy scikit-learn spacy
   python -m spacy download en_core_web_sm
   ```
2. Run the notebook `twitter-sentiment-analysis.ipynb` for step-by-step execution.

## Sentiment Classes

- **Irrelevant** : 0
- **Natural** : 1
- **Negative** : 2
- **Positive** : 3

## References

- [Kaggle Twitter Entity Sentiment Analysis Dataset](https://www.kaggle.com/datasets)
- [spaCy Documentation](https://spacy.io/)
- [scikit-learn Documentation](https://scikit-learn.org/)

---

For details, see the notebook: `twitter-sentiment-analysis.ipynb`.
