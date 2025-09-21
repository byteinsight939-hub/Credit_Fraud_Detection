# 💳 Credit Card Fraud Detection Dashboard

A dynamic data visualization dashboard built using **Dash** and **Plotly**, designed to explore and analyze credit card transaction data for patterns of fraud. This project offers an interactive way to understand how fraud occurs across different features and time periods using real-world anonymized data.

---

## 📊 About the Project

This dashboard analyzes over 280,000 transactions, highlighting fraudulent behavior patterns and allowing users to:

- Filter transactions by **hour of the day**
- Explore feature-wise distributions
- Visualize PCA-based feature reduction
- Discover hidden trends through a **correlation heatmap**

This project demonstrates the power of **visual analytics** in fraud detection scenarios.

---

## 📁 Dataset Information

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Records**: 284,807 transactions
- **Features**:
  - V1–V28: Anonymized features after PCA transformation
  - Time: Seconds elapsed since the first transaction
  - Amount: Transaction amount
  - Class: `1` = Fraud, `0` = Normal

---

## 🚀 Key Features of the Dashboard

| Feature                     | Description                                               |
|----------------------------|-----------------------------------------------------------|
| ⏱ Hour Filter              | Filter transactions between specific hours of the day     |
| 📉 Feature Histogram        | Compare distributions of selected features by class       |
| 🔍 PCA Scatter Plot         | View 2D reduction of transaction features                 |
| 🌡 Correlation Heatmap      | Analyze inter-feature relationships                      |

---

## 🧠 How it Works

- **Preprocessing**:
  - Extracted `Hour` from `Time` feature.
  - Filtered data dynamically based on user inputs.
- **Visualization Tools**:
  - Built with [Dash](https://dash.plotly.com/) and [Plotly](https://plotly.com/python/).
  - Real-time callbacks for seamless interactivity.

---

## 🛠 Installation

```bash
pip install pandas numpy scikit-learn dash plotly
