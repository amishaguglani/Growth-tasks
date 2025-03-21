# Movie Rating Prediction

This project demonstrates a machine learning pipeline for predicting movie ratings based on various features such as duration, votes, genre, director, and actors. The dataset is preprocessed, analyzed, and used to train multiple regression models to predict the target variable, `Rating`.

## Features of the Project

1. **Data Preprocessing**:
   - Handles missing values by removing or imputing them.
   - Encodes categorical variables (e.g., actors, directors, genres) using mean encoding.
   - Converts textual and numerical data into a usable format for machine learning.

2. **Exploratory Data Analysis (EDA)**:
   - Visualizes distributions and relationships between features using plots (e.g., scatterplots, violin plots, heatmaps).
   - Analyzes trends such as average ratings per year and top-rated genres/directors/actors.

3. **Model Training**:
   - Implements multiple regression models:
     - Linear Regression
     - Ridge Regression
     - Decision Tree Regressor
     - Random Forest Regressor
   - Uses hyperparameter tuning with GridSearchCV for optimal performance.

4. **Model Evaluation**:
   - Evaluates models using metrics such as R², MAE, MSE, and RMSE.
   - Visualizes residuals and actual vs. predicted values.

5. **Cross-Validation**:
   - Performs k-fold cross-validation to ensure model robustness.

## How to Use the Model

### Prerequisites
- Python 3.7 or higher
- Required libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

Install the dependencies using:
```bash
pip install -r requirements.txt
```

### Steps to Run the Project
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Place the dataset (`tested.csv`) in the project directory.

3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Movie_Rating_Prediction.ipynb
   ```

4. Run the cells sequentially to:
   - Preprocess the data
   - Perform EDA
   - Train and evaluate models
   - Visualize results

### Using the Model
- After training, the best-performing model can be used to predict movie ratings for new data.
- Replace the `X_test` dataset with your own data in the same format and use the `.predict()` method of the trained model.

### Example
```python
# Example: Predict ratings for new data
new_data = pd.DataFrame({
    'Duration': [120],
    'Votes': [50000],
    'Year': [2022],
    'actor1_encoded': [7.5],
    'actor2_encoded': [7.0],
    'actor3_encoded': [6.8],
    'director_encoded': [8.0],
    'genre_encoded': [7.2]
})
predicted_rating = grid_search_RF.predict(new_data)
print("Predicted Rating:", predicted_rating)
```

## Results
- The project compares the performance of different models and selects the best one based on evaluation metrics.
- Visualizations provide insights into the data and model predictions.

