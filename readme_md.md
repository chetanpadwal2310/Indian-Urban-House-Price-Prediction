# 🏠 Indian Urban House Price Prediction

An end-to-end Machine Learning regression system designed to predict residential property prices in urban India. This project features automated preprocessing pipelines, model selection across multiple regression algorithms, and an interactive web application built with Gradio.

---

## 🚀 Features
- **Data Pipeline:** Automated missing value imputation, scaling (`StandardScaler`), and categorical encoding (`OneHotEncoder`) wrapped in Scikit-Learn `Pipeline` and `ColumnTransformer`.
- **Model Evaluation:** Trains and compares multiple models (Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, and AdaBoost) to automatically select the one with the highest $R^2$ score.
- **Interactive Web App:** Deploys the best-performing model using an automatically generated **Gradio** interface.

---

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib, Seaborn, Gradio, Joblib
- **Platform:** Google Colab / Kaggle API

---

## 📂 Project Structure
```text
├── indian_house_prices_dataset.csv       # Dataset
├── house_price_regression_pipeline.pkl   # Saved best model pipeline
└── README.md                             # Project Documentation
```

---

## 🏃‍♂️ How to Run Locally

1. Clone the repository and install the dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn gradio joblib
   ```

2. Run the Jupyter Notebook or load the saved pipeline into your inference script:
   ```python
   import joblib

   model = joblib.load("house_price_regression_pipeline.pkl")
   ```

3. Launch the Gradio web UI:
   ```python
   demo.launch()
   ```

---

## 📊 Model Performance
The pipeline automatically evaluates all models and selects the top performer based on metrics including **$R^2$ Score**, **RMSE**, **MSE**, and **MAE**.