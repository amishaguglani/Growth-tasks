# Sales Prediction using Machine Learning

This project focuses on forecasting product sales based on historical data using machine learning models. The goal is to help businesses optimize their marketing strategies and make data-driven decisions.

## Key Steps in the Project
1. **Data Loading and Preprocessing**: 
   - Loaded the dataset and handled missing values, duplicates, and outliers.
   - Applied feature scaling to normalize the data.

2. **Exploratory Data Analysis (EDA)**:
   - Visualized distributions and relationships between features.
   - Identified key patterns and trends in the data.

3. **Feature Engineering**:
   - Rounded numerical values for better interpretability.
   - Applied transformations to enhance model performance.

4. **Model Training and Evaluation**:
   - Trained models including K-Nearest Neighbors (KNN) and Random Forest Regressor.
   - Evaluated models using metrics like Mean Absolute Error (MAE) and Mean Squared Error (MSE).

5. **Conclusion and Recommendations**:
   - Provided actionable insights based on the analysis and model results.

---

## Insights from the Analysis

### 1. Feature Importance:
The Random Forest model highlights the most significant factors impacting sales:
- **Advertising Spend**: Increasing investment in advertising can lead to higher sales.
- **Promotions/Discounts**: Running targeted promotional campaigns boosts sales.

### 2. Correlation Analysis:
- **Positive Correlations**: Factors like advertising spend positively impact sales.
- **Negative Correlations**: Issues like customer complaints negatively affect sales, suggesting areas for improvement.

### 3. Outlier Detection:
- Outliers may indicate anomalies or specific events (e.g., successful campaigns or supply chain issues). Understanding these can help replicate successes or address problems.

### 4. Forecasting:
- The model predicts future sales based on historical data and planned strategies.
- Businesses can simulate scenarios (e.g., increased advertising budgets) to forecast potential sales growth.

### 5. Customer Segmentation:
- Analysis can reveal which customer groups contribute the most to sales.
- Businesses can focus on these segments with tailored marketing strategies to maximize ROI.

---

## Recommendations
- **Increase Investment**: Focus on impactful factors like advertising and promotions.
- **Address Negative Factors**: Improve customer satisfaction by addressing complaints.
- **Forecasting**: Use the model to simulate scenarios and optimize strategies.
- **Regular Updates**: Continuously update the model with new data for accurate predictions.

---

## How to Use the Model
1. Load the dataset and preprocess it as described in the notebook.
2. Train the model using the provided code.
3. Use the trained model to predict sales for new data points.
4. Analyze the results and apply the insights to optimize business strategies.

This project provides a robust framework for sales prediction and strategic planning, enabling businesses to make informed decisions and drive growth.
