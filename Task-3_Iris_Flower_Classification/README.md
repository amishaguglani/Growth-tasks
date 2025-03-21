# Iris Flower Classification

This project demonstrates the classification of Iris flower species using machine learning algorithms. The dataset used is the famous Iris dataset, which contains measurements of sepal length, sepal width, petal length, and petal width for three species of Iris flowers: Setosa, Versicolor, and Virginica.

## Features
- Data preprocessing, including handling duplicates and encoding categorical data.
- Exploratory Data Analysis (EDA) with visualizations such as scatter plots, pie charts, and heatmaps.
- Implementation of two machine learning models:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
- Model evaluation using accuracy scores.

## How It Works
1. The dataset is loaded and cleaned by removing duplicates and encoding the target variable (`species`).
2. EDA is performed to understand the relationships between features and the distribution of the target variable.
3. The dataset is split into training and testing sets.
4. Two machine learning models (Logistic Regression and KNN) are trained on the training set.
5. The models are evaluated on the test set, and their accuracy is calculated.

## Visualizations
- Scatter plots for sepal and petal dimensions.
- Correlation heatmap to show relationships between features.
- Pie chart to display the distribution of species.

## How to Use
1. Clone the repository from GitHub:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd Growth link
   ```
3. Open the Jupyter Notebook file `Iris_Flower_Classification.ipynb` in your preferred environment (e.g., Jupyter Notebook, VS Code).
4. Run the cells sequentially to execute the code.

## Requirements
- Python 3.x
- Libraries: `numpy`, `pandas`, `seaborn`, `matplotlib`, `scikit-learn`

Install the required libraries using:
```bash
pip install -r requirements.txt
```

## Dataset
The dataset file `IRIS.csv` should be placed in the same directory as the notebook. If not available, download it from [Kaggle](https://www.kaggle.com/).



