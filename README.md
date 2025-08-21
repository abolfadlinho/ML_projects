# ML_projects: Comprehensive Machine Learning Repository

Welcome to ML_projects, my curated collection of machine learning and deep learning projects spanning classification, clustering, regression, reinforcement learning, time-series forecasting, NLP, and more. This repository is designed for learners, practitioners, and researchers seeking hands-on examples, best practices, and reproducible workflows across the ML spectrum.

**My Kaggle Profile:** [Ahmed Abolfadl on Kaggle](https://www.kaggle.com/ahmedamrabolfadl)

---

## Repository Structure

- **Classification/**
  - EyeDiseases: Deep learning for eye disease image classification (EfficientNetB3).
  - PlantVillageDisease: Plant disease detection using transfer learning.
  - README.md: Concepts, algorithms, and workflow for classification tasks.
- **Clustering/**
  - CustomersSegmentation: Mall customer segmentation with KMeans and EDA.
  - IrisClustering: Automated clustering optimization with Optuna (KMeans, DBSCAN, GMM).
  - README.md: Clustering theory, algorithms, and evaluation.
- **Deep-Learning/**
  - MathSymbolsCNN: CNN for mathematical symbol image classification.
  - README.md: Deep learning architectures, concepts, and applications.
- **NLP/**
  - TwitterSentimentAnalysis: Sentiment classification on Twitter data (spaCy, scikit-learn).
  - README.md: NLP concepts, workflows, and libraries.
- **Regression/**
  - BankTermDepositPrediction: Predicting bank term deposit subscriptions with multiple ML models.
  - README.md: Regression theory, algorithms, and metrics.
- **Reinforcement-Learning/**
  - README.md: RL concepts, algorithms, and applications.
- **Time-Series/**
  - TeslaStockForecasting: LSTM-based forecasting of Tesla stock prices.
  - README.md: Time-series forecasting concepts and methods.

---

## Project Highlights

### Classification

- **EyeDiseases**: EfficientNetB3-based image classification, stratified data splits, custom callbacks, and reproducible results.
- **PlantVillageDisease**: Transfer learning for plant disease detection, advanced augmentation, and model evaluation.

### Clustering

- **CustomersSegmentation**: Univariate, bivariate, and multivariate clustering with KMeans, EDA, and cluster analysis.
- **IrisClustering**: Optuna-powered search for best clustering algorithm and parameters, maximizing silhouette score.

### Deep Learning

- **MathSymbolsCNN**: Custom CNN for symbol recognition, with full training, evaluation, and result saving.

### NLP

- **TwitterSentimentAnalysis**: End-to-end sentiment analysis pipeline, including preprocessing, feature extraction, and model comparison.

### Regression

- **BankTermDepositPrediction**: Multi-model approach (Logistic Regression, SVC, KNN, Decision Tree, Random Forest, XGBoost, CatBoost, LightGBM), feature engineering, and robust evaluation.

### Time-Series

- **TeslaStockForecasting**: LSTM neural network for stock price prediction, sequence generation, and future forecasting.

---

## How to Use This Repository

1. **Explore Subdomains**: Each subfolder contains a README.md with domain-specific theory, workflow, and references.
2. **Run Notebooks**: Follow the step-by-step Jupyter notebooks for hands-on experimentation.
3. **Install Dependencies**: Use the provided requirements in each project README or run:
   ```powershell
   pip install pandas numpy scikit-learn seaborn matplotlib tensorflow keras xgboost catboost lightgbm spacy
   python -m spacy download en_core_web_sm
   ```
4. **Reproduce Results**: All projects include code for data loading, preprocessing, model training, evaluation, and saving outputs.
5. **Learn & Extend**: Use the workflows as templates for your own ML projects or research.

---

## Technologies & Libraries

- **Python**: Core language for all projects.
- **pandas, numpy, scikit-learn**: Data manipulation and classical ML.
- **seaborn, matplotlib**: Data visualization.
- **tensorflow, keras, EfficientNet, LSTM**: Deep learning and neural networks.
- **xgboost, catboost, lightgbm**: Gradient boosting for tabular data.
- **spaCy, NLTK, Hugging Face Transformers**: NLP and text processing.
- **Optuna**: Hyperparameter optimization.
- **Stable Baselines3, RLlib**: Reinforcement learning (see RL README).

---

## Best Practices & Reproducibility

- **Stratified Data Splits**: Ensures balanced training, validation, and test sets.
- **Data Augmentation**: Improves generalization for image tasks.
- **Custom Callbacks**: Early stopping, learning rate scheduling, and user interaction.
- **Comprehensive Evaluation**: Confusion matrices, classification reports, cluster analysis, and forecasting metrics.
- **Result Saving**: Models, weights, and key outputs are saved for reproducibility.

---

## References & Further Reading

- Each subdomain README.md includes links to foundational papers, datasets, and documentation.
- Explore the notebooks for code examples, explanations, and practical tips.

---

## License

This repository is open-source and available under the MIT License.
