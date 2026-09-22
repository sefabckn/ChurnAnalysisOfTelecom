
## Telecom Customer Churn Intelligence

### 🔍 Project Overview

This project analyzes and predicts customer churn using a reusable Python pipeline and a Streamlit dashboard. It uses the Telco Customer Churn dataset to identify churn drivers, compare customer segments, and score individual retention risk.

### 📁 Project Structure

* 📊 `app.py`: Streamlit dashboard for segment analysis and customer scoring.
* 🧠 `src/churn_analysis/pipeline.py`: Shared data preparation and random forest model pipeline.
* 🧪 `tests/test_pipeline.py`: Automated checks for cleaning and scoring.
* 📓 `churnanalysis.ipynb`: Exploratory analysis and original research notes.
* 📊 Data source: [Telco Customer Churn Dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

### 🎯 Objectives

* Understand patterns in customer behavior that lead to churn.
* Explore statistical features using EDA.
* Visualize customer characteristics and trends.
* Build classification models to predict churn.
* Recommend strategic actions based on insights.

---

### 🧠 Key Questions Addressed

* What types of customers are more likely to churn?
* Does contract type or monthly charges influence churn?
* Can we predict churn with high accuracy using machine learning?

---

### 📊 Exploratory Data Analysis (EDA)

* **Distributions:** Tenure, MonthlyCharges, TotalCharges
* **Central Tendencies:** Mean, Median, Mode
* **Visualizations:** Histograms, Boxplots, Scatterplots, Heatmaps
* **Categorical Insights:** Churn by Contract, Payment Method, Internet Service, etc.

📷 *Example Visuals:*

![Tenure Histogram](Visuals/histOfTenure.png)
![Dist Contract Type](Visuals/distofcontracttype.png)
![HeatMap of Correlation Matris](Visuals/heatmap.png)

---

### 🤖 Machine Learning Models

#### ✅ Classification

* **Target:** `Churn (Yes/No)`
* **Models:**

  * Random Forest Classifier (best performing)
  * Logistic Regression
  * XGBoost (optional for future)
* **Evaluation Metrics:**

  * Accuracy: \~78.7%
  * Precision: 65.0%
  * Recall: 42.5%
  * F1 Score: 51.4%
  * Confusion Matrix & Classification Report

#### 📈 Regression (Optional Insight)

* **Target:** `MonthlyCharges` (continuous variable)
* **Model:** Random Forest Regressor
* **Evaluation:** R² \~0.9976, RMSE \~1.46

---

### 💡 Business Insights

* **High churn risk:** Customers with month-to-month contracts and high monthly charges.
* **Loyal customers:** Tend to have longer tenure, use bundled services.
* **Actionable strategies:**

  * Promote long-term contracts
  * Improve onboarding experience
  * Offer discounts to high-value customers

---

### Requirements

```bash
Python 3.10+
See `requirements.txt` for the complete dependency list.
```

Install via:

```bash
pip install -r requirements.txt
```

---

### Run the dashboard

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start Streamlit:

```bash
streamlit run app.py
```

3. Upload `Telco-Customer-Churn.csv` in the sidebar, or place it at the project root. The CSV must include `Churn`, `tenure`, `MonthlyCharges`, and `TotalCharges`.

### Run with Docker

Build and start the dashboard container:

```bash
docker build -t telecom-churn-dashboard .
docker run --rm -p 8501:8501 telecom-churn-dashboard
```

Open `http://localhost:8501` and upload the dataset in the sidebar. To use a local CSV automatically, mount it into the container:

```bash
docker run --rm -p 8501:8501 -v "${PWD}/Telco-Customer-Churn.csv:/app/Telco-Customer-Churn.csv:ro" telecom-churn-dashboard
```

### Run tests

```bash
pytest
```

---

### 📚 References

* [Telco Churn Dataset – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* Scikit-learn Documentation
* Python Data Science Libraries

---

### 🙌 Author

**Sefa Bockun**
Data Analyst | MBA in Data Science
📍 Nowy Sącz, Poland
📫 [LinkedIn](https://www.linkedin.com/in/sefabockun)

