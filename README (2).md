# Formula 1 DNF Classification - Exploratory Data Analysis (EDA)

This project performs an initial Exploratory Data Analysis (EDA) on a historical Formula 1 dataset to understand factors influencing whether a driver finishes a race (`target_finish`).

## Dataset

The dataset was obtained from Kaggle (`pranay13257/f1-dnf-classification`) using the `kagglehub` library.

## EDA Process and Findings

The EDA involved examining the structure, missing values, and distributions of various features, along with visualizing relationships with the target variable (`target_finish`).

**Key Observations:**

*   **Data Loading and Cleaning:** The dataset was loaded and initial cleaning addressed missing values and data type conversions (e.g., converting '\N' to numerical and the 'date' column to datetime). The `fastestLapTime` column was dropped due to excessive missing values.
*   **Feature Distributions and Relationships:** Visualizations were generated to explore features such as year, round, grid position, points, laps completed, driver nationality, and constructor.

**Insights for Machine Learning Modeling:**

Based on the EDA, the following insights are crucial for guiding the choice of machine learning models:

1.  **Grid Position:** Starting grid position (`grid`) shows a strong correlation with finishing the race, with higher grid numbers (further back) associated with increased DNF rates. This feature is likely a key predictor.
2.  **Performance Metrics:** Features like `points` and `laps` clearly differentiate finishers from DNFs. While informative, careful handling is needed due to missing values and their outcome-dependent nature. The significant missing data in other performance metrics requires a strategy (imputation or robust models).
3.  **Categorical Influence:** Driver nationality (`nationality_x`) and constructor (`constructorRef`) show varying DNF rates, indicating their importance as categorical features in a predictive model.
4.  **Numerical Feature Correlations:** The correlation heatmap revealed relationships between numerical features such as a negative correlation between race completion time (`milliseconds`) and `year`, suggesting races have become shorter over time.

These insights suggest that a classification model considering a mix of numerical and categorical features, and capable of handling missing data, would be suitable for predicting race completion.

## Code

The analysis was performed using Python with pandas, matplotlib, and seaborn. Key steps included data loading, cleaning (handling missing values and data types), and visualization.

*(Note: Code snippets and visualizations are included in the accompanying notebook.)*
