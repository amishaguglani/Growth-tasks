# Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using machine learning techniques. The dataset used is highly imbalanced, with a majority of transactions being legitimate and a small fraction being fraudulent. The goal is to build a robust model that can accurately classify transactions as fraud or non-fraud.

## Steps Involved

1. **Import Libraries**:
   - Essential libraries like NumPy, pandas, Matplotlib, Seaborn, Plotly, and scikit-learn are used for data manipulation, visualization, and modeling.

2. **Load and Explore Data**:
   - The dataset is loaded, and basic information such as null values and statistical summaries are explored.

3. **Data Visualization**:
   - Various plots (count plots, histograms, scatter plots, box plots) are created to understand the data distribution and relationships.

4. **Data Preprocessing**:
   - The dataset is balanced using undersampling and SMOTE (Synthetic Minority Oversampling Technique).
   - Features are normalized, and unnecessary columns are dropped.

5. **Model Creation and Training**:
   - The data is split into training and testing sets.
   - Multiple models (Logistic Regression, Random Forest, Gradient Boosting) are trained and evaluated.

6. **Model Evaluation**:
   - Models are evaluated using metrics like accuracy, classification report, and ROC-AUC score.
   - Confusion matrices are visualized for all models.

## How to Run the Notebook

1. **Prerequisites**:
   - Install Python (version 3.7 or higher).
   - Install the required libraries using the following command:
     ```bash
     pip install numpy pandas matplotlib seaborn plotly scikit-learn imbalanced-learn
     ```

2. **Dataset**:
   -https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
   - Ensure the dataset file `creditcard.csv` is placed in the same directory as the notebook.

4. **Run the Notebook**:
   - Open the `Card_Fraud_Detection.ipynb` file in Jupyter Notebook or any compatible IDE.
   - Execute the cells sequentially to reproduce the results.

## Results

- The notebook demonstrates the performance of different models in detecting fraudulent transactions.
- Visualizations and metrics provide insights into the effectiveness of each model.

## Note

- The dataset is highly imbalanced, so handling class imbalance is crucial for building an effective model.
- The project uses both undersampling and SMOTE to address this issue.

Feel free to explore and modify the notebook to experiment with different techniques and models!
