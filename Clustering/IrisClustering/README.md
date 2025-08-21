# IrisClustering: Optuna-based Clustering Optimization

This project demonstrates unsupervised clustering of the Iris dataset using automated hyperparameter optimization with Optuna. Multiple clustering algorithms are explored and tuned, including KMeans, DBSCAN, and Gaussian Mixture Models (GMM).

## Workflow

- **Dataset:** Uses the classic Iris dataset (sepal/petal measurements for three species).
- **Algorithms:** KMeans, DBSCAN, and GMM are considered.
- **Optimization:** Optuna is used to search for the best algorithm and hyperparameters by maximizing the silhouette score.
- **Evaluation:** The best clustering configuration is selected and retrained; final clustering quality is reported using the silhouette score.

## Key Features

- Automated selection between clustering algorithms and their parameters.
- Objective function ensures meaningful clusters (avoids trivial solutions).
- Final model and clustering quality are reported for reproducibility.

## Usage

- Run the notebook to install dependencies and execute the optimization workflow.
- The best algorithm and parameters are printed, along with the final silhouette score.

## Requirements

- Python 3.x
- scikit-learn
- optuna
- numpy

## Reference

- See `iris-clustering-optuna.ipynb` for full code and experiment details.
