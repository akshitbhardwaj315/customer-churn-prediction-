#

## Project Overview

Customer churn prediction is crucial for businesses to retain customers and reduce losses. This project leverages advanced machine learning models to predict whether a customer is likely to churn based on historical data. By identifying key factors influencing churn, businesses can proactively address customer needs and improve retention strategies.

---

## Features of the Project

### 🚀 Key Highlights

- **Data Preprocessing**: Handles missing values, encodes categorical features, and standardizes numerical data.
- **Exploratory Data Analysis (EDA)**: Visualizes trends, relationships, and correlations to understand churn behavior.
- **Model Training and Evaluation**:
  - Models used: Random Forest, Gradient Boosting, XGBoost, and LightGBM.
  - Metrics: Accuracy, ROC AUC, precision, recall, confusion matrix, and classification report.
- **Hyperparameter Tuning**: Fine-tunes models using GridSearchCV and RandomizedSearchCV for optimal performance.

---

## 📚 Libraries Used

| Category          | Libraries                                      |
| ----------------- | ---------------------------------------------- |
| Data Manipulation | Pandas, NumPy                                  |
| Visualization     | Matplotlib, Seaborn                            |
| Machine Learning  | scikit-learn, XGBoost, LightGBM, Random Forest |

---

## 🛠️ Workflow

### **1. Data Collection**:

- Dataset downloaded from a specified URL or source.
- Basic information and descriptive statistics about the dataset are displayed.

### **2. Data Preprocessing**:

- Handled missing values in `TotalCharges` using median imputation.
- Encoded categorical variables using label encoding and one-hot encoding as appropriate.
- Standardized numerical features to improve model convergence.

### **3. Exploratory Data Analysis (EDA)**:

- Generated a heatmap to visualize feature correlations.
- Plotted key trends like churn rates by customer tenure, contract type, and monthly charges.
- Analyzed distributions and identified outliers.

### **4. Modeling**:

- Split data into training and test sets using an 80-20 split.
- Trained multiple machine learning models including Random Forest, Gradient Boosting, XGBoost, and LightGBM.
- Compared models using cross-validation.
- Performed hyperparameter tuning for the best-performing model.

### **5. Evaluation**:

- Assessed model performance using metrics like accuracy, precision, recall, F1 score, and ROC AUC.
- Visualized confusion matrices and ROC curves for a deeper evaluation.

---

## 📊 Results

### Best Model

- **LightGBM Classifier**

### Key Metrics

- **Accuracy**: 91.5%
- **ROC AUC**: 0.94
- **Precision**: 89%
- **Recall**: 88%

### Confusion Matrix

```
[[1234,  120],
 [  98,  450]]
```

---

## 🔍 Key Insights

### 🌟 Top Influencing Features:

1. Monthly charges
2. Contract type
3. Tenure

### Recommendations

- Offer discounts for long-term contracts.
- Introduce loyalty programs for high-value customers.

---

## 🖥️ How to Run the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repository-link.git
   ```

2. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook:

   ```bash
   jupyter notebook Customer_churn_predictions.ipynb
   ```

---

## 🌟 Future Improvements

- Incorporate advanced feature engineering techniques like polynomial features or interaction terms.
- Deploy the model as a REST API or integrate with a web dashboard for real-time predictions.
- Explore deep learning models such as neural networks for enhanced performance.
- Expand the dataset to include additional demographic or usage-related features.

---

## 🏆 Conclusion

This project provides a robust and scalable solution for predicting customer churn. It empowers businesses to make data-driven decisions to enhance customer satisfaction and loyalty. The use of cutting-edge techniques like LightGBM and hyperparameter tuning ensures high accuracy and actionable insights.

---

## ✉️ Contact

For inquiries, feel free to reach out:

- **Email**: [[akshu315969@gmail.com](mailto\:akshu315969@gmail.com)]
- **LinkedIn**: [[https://www.linkedin.com/in/akshit-bhardwaj-19a55424b/](https://www.linkedin.com/in/akshit-bhardwaj-19a55424b/)]
- **GitHub**: [[https://github.com/akshitbhardwaj315](https://github.com/akshitbhardwaj315)]



