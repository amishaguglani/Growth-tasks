# Titanic Survival Prediction

This project predicts the survival of passengers aboard the Titanic using a machine learning model. The dataset used for this analysis is preprocessed to handle missing values, encode categorical variables, and train a Naive Bayes classifier.

## How the Prediction is Done

1. **Data Loading**:
   - The dataset is loaded from a CSV file (`tested.csv`) into a pandas DataFrame.

2. **Data Exploration**:
   - The dataset is inspected for missing values, duplicate rows, and column data types.
   - Visualizations are created to understand the distribution of data and missing values.

3. **Data Cleaning**:
   - Columns with excessive missing values (e.g., `Cabin`) are dropped.
   - Missing values in columns like `Age` and `Embarked` are filled with the mean and a default value, respectively.

4. **Feature Engineering**:
   - Irrelevant columns (`Name`, `Ticket`, `PassengerId`) are removed.
   - Categorical variables (`Sex`, `Embarked`) are encoded into numerical values using `LabelEncoder`.

5. **Splitting the Dataset**:
   - The dataset is split into features (`x`) and the target variable (`y`).
   - The data is further divided into training and testing sets using `train_test_split`.

6. **Model Training**:
   - A Naive Bayes classifier (`GaussianNB`) is trained on the training data.

7. **Model Evaluation**:
   - Predictions are made on the test set.
   - The accuracy of the model is calculated using `accuracy_score`.

8. **Testing the Model**:
   - The model is tested with a sample input to predict whether a passenger would survive or not.

## How to Use the Project

1. **Clone the Repository**:
   - Clone this repository to your local machine using:
     ```bash
     git clone <repository-url>
     ```

2. **Install Dependencies**:
   - Ensure you have Python installed.
   - Install the required libraries using:
     ```bash
     pip install pandas numpy matplotlib seaborn scikit-learn missingno
     ```

3. **Run the Notebook**:
   - Open the Jupyter Notebook file `Titanic_0Survival_Prediction.ipynb`.
   - Execute the cells step-by-step to preprocess the data, train the model, and make predictions.

4. **Test the Model**:
   - Modify the sample input in the testing section to test the model with different passenger details:
     ```python
     testPrediction = NB.predict([[0, 3, 1, 22.0, 1, 6, 2]])
     ```

5. **Understand the Output**:
   - If the output is `1`, the passenger is predicted to have survived.
   - If the output is `0`, the passenger is predicted to have not survived.

## Example Usage

```python
# Example input for prediction
testPrediction = NB.predict([[0, 3, 1, 22.0, 1, 6, 2]])

# Output interpretation
if testPrediction == 1:
    print("Survived")
else:
    print("Dead💀")
```

## Project Structure

- `Titanic_0Survival_Prediction.ipynb`: Jupyter Notebook containing the code for data preprocessing, model training, and prediction.
- `tested.csv`: Dataset used for training and testing the model.
- `README.md`: Documentation for understanding and using the project.

## Dependencies

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- missingno

