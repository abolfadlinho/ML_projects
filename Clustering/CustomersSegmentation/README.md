# Customers Segmentation (Mall Customers)

This project demonstrates customer segmentation using clustering techniques on the Mall Customers dataset. The workflow includes data loading, exploratory data analysis (EDA), clustering (univariate, bivariate, multivariate), and saving results.

## Workflow Overview

1. **Data Loading**

   - Reads the dataset `Mall_Customers.csv`.
   - Displays initial rows and column names for inspection.

2. **Exploratory Data Analysis (EDA)**

   - Univariate analysis: Distribution plots for Age, Annual Income, and Spending Score.
   - Bivariate analysis: Scatter plots, KDE plots, boxplots, and pairplots by Gender.
   - Correlation analysis: Heatmap and correlation matrix.

3. **Clustering**

   - **Univariate Clustering**: KMeans on Annual Income, cluster assignment, and inertia scores for elbow method.
   - **Bivariate Clustering**: KMeans on Annual Income and Spending Score, cluster assignment, cluster centers, and visualization.
   - **Multivariate Clustering**: One-hot encoding for Gender, scaling features, KMeans clustering, and inertia scores for elbow method.

4. **Analysis & Results**
   - Grouping and mean statistics by clusters.
   - Crosstab analysis for cluster vs. gender.
   - Saves clustered data to `Clustering.csv`.

## Key Code Snippet: Display DataFrame Columns

```python
dff.columns
```

## Requirements

- Python
- pandas
- seaborn
- matplotlib
- scikit-learn

## Usage

1. Install dependencies:
   ```powershell
   pip install pandas seaborn matplotlib scikit-learn
   ```
2. Run the notebook `customers-segmentation.ipynb` for step-by-step execution.

## References

- [Mall Customers Dataset](https://www.kaggle.com/datasets)
- [scikit-learn Documentation](https://scikit-learn.org/)

---

For details, see the notebook: `customers-segmentation.ipynb`.
