# ML_projects

## 1. 🎬 Movie Recommender using Decision Tree and Optuna

This project builds a movie recommender system using the [Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) from Kaggle. It trains a Decision Tree Classifier to predict whether a user would "like" a movie based on its popularity and genres.

### 📌 Features

- Binary classification: `like = 1 if vote_average ≥ 7, else 0`
- One-hot encoding for genres
- Uses `popularity` and `genres` as features
- Hyperparameter tuning with **Optuna**
- Model evaluation using accuracy, precision, recall, and F1-score
- Top movie recommendations based on predicted "likes"
- Visualizes optimized decision tree

### 📁 Dataset

Downloaded via `kagglehub`:
- File used: `movies_metadata.csv`

### 📦 Dependencies

pip install optuna kagglehub

### 🧠 Model Training

- Train-test split on preprocessed features
- Custom evaluation metric prioritizes class 1 (likes)
- Decision Tree optimized with Optuna for best generalization

### 📊 Evaluation Sample

Accuracy: 0.73

Classification Report:
              precision    recall  f1-score   support
           0       0.83       0.82      0.83      7168
           1       0.36       0.38      0.37      1893

### 🎯 Movie Recommendations

Predicts which movies are likely to be liked (`predicted_like = 1`) and recommends the top 20 based on vote average and popularity.

### 🌳 Decision Tree Visualization

Plots a limited-depth decision tree to interpret model behavior and dominant features.

### 📌 Notes

This is a baseline model. You can improve it by:
- Balancing the dataset using SMOTE or class weights
- Trying ensemble models (e.g., Random Forest)
- Incorporating additional content features (overview, cast, crew, tags)

## 2. Gold Price Predictor  

### Overview  
A deep learning model that predicts gold prices using historical data with a Convolutional Neural Network (CNN).  

### Features  
- Data preprocessing & normalization  
- CNN architecture with Conv1D layers  
- Early stopping during training  
- Evaluation metrics (MAE, RMSE)  
- 30-day future price forecasting  

### Requirements  
- Python 3.x  
- TensorFlow  
- scikit-learn  
- pandas, numpy, matplotlib  
- kagglehub  

### Usage  
1. Load and preprocess historical gold price data  
2. Split into training/test sets  
3. Train CNN model  
4. Evaluate performance  
5. Generate future predictions  

### Results  
- Price prediction vs actual visualization  
- Training/validation loss curves  
- 30-day forecast with dates  

Dataset: [Kaggle Gold Price Prediction Dataset](https://www.kaggle.com/datasets/sid321axn/gold-price-prediction-dataset)  
