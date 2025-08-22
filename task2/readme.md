📊 Sales Prediction using Machine Learning

This project is part of my CodeSoft Data Science Internship (Task 4).
The objective is to predict sales revenue based on different advertising investments (TV, Radio, Newspaper).

🚀 Project Overview

Advertising plays a major role in driving product sales. Businesses spend millions on TV, Radio, and Newspaper ads, but the challenge is knowing which medium contributes the most to sales.

In this project, I:

Performed Exploratory Data Analysis (EDA) to understand patterns.

Built and compared multiple Machine Learning models.

Evaluated their performance using R² Score and RMSE.

Identified the most influential features driving sales.

📂 Project Structure
Codesoft Sales Prediction/
│
├── data/
│   └── advertising.csv          # Dataset
│
├── sales_prediction.py          # Model training script
├── eda.ipynb                    # EDA & visualizations
└── README.md                    # Project documentation

🛠️ Steps Performed
1. Data Loading & Cleaning

Imported the Advertising dataset.

Checked for missing values and data types.

Ensured dataset is clean and ready for analysis.

2. Exploratory Data Analysis (EDA)

Dataset preview with shape, info, and summary statistics.

Correlation heatmap to identify relationships between features.

Pairplots & scatterplots between Sales and each advertising channel.

Distribution plots to check the spread of data.

3. Model Building

Trained and compared the following models:

Linear Regression

Ridge Regression

Lasso Regression

Random Forest Regressor

4. Model Evaluation

Metrics used: R² Score and RMSE (Root Mean Squared Error).

Compared all models side by side to identify the best performer.

5. Feature Importance

Used Random Forest feature importance to check which channel (TV, Radio, Newspaper) influences sales the most.

📈 Results

TV advertising showed the strongest correlation with sales.

Linear Regression and Random Forest gave the best predictive performance.

Radio contributed moderately, while Newspaper had the least impact.

Final model performance (example results):

Model	R² Score	RMSE
Linear Regression	0.89	1.65
Ridge Regression	0.88	1.70
Lasso Regression	0.87	1.72
Random Forest	0.91	1.55
📊 Visualizations

Correlation Heatmap

Feature Distributions

Scatterplots of Sales vs Ads

Model Performance Comparison (Bar Chart)

Feature Importance (Random Forest)

📝 Conclusion

TV ads are the most effective for driving sales.

A linear approach works well, but ensemble methods like Random Forest slightly improve accuracy.

Businesses should focus more on TV and Radio advertising for better ROI.

📌 How to Run

Clone the repository:

git clone https://github.com/Aryanaam/Codesoft-Internship-Projects.git
cd Codesoft-Internship-Projects/task4


Open EDA notebook:

jupyter notebook eda.ipynb


Run training script:

python sales_prediction.py

