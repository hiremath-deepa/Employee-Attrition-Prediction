# 📊 Employee Attrition Prediction using Python and RapidMiner

## 📌 About
This project applies advanced data mining techniques to predict employee attrition using the IBM HR Analytics dataset. Leveraging both Python (ML/DL models) and RapidMiner Auto Model, the analysis follows the CRISP-DM framework to build, evaluate, and compare predictive models for employee turnover, aiming to help organizations retain key talent and reduce operational costs.

---

## 🎯 Objectives
- Identify key factors influencing employee attrition.
- Develop predictive models using machine learning and deep learning techniques.
- Compare the performance of models developed in Python vs. RapidMiner.
- Recommend actionable insights for HR departments to reduce attrition.

---

## 🧪 Dataset
**Source:** [IBM HR Analytics Dataset (Kaggle)](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)  
- **Instances:** 1,470  
- **Features:** 35 (demographics, job role, satisfaction metrics, etc.)  
- **Target Variable:** `Attrition` (Yes/No)  

---

## ⚙️ Tools & Technologies
- **Python** (Pandas, Scikit-learn, Matplotlib, Seaborn)
- **RapidMiner Auto Model**
- **SMOTE** for handling class imbalance
- **CRISP-DM Framework** for structured analysis

---

## 🧹 Data Preparation
- Label Encoding & Feature Scaling
- Class imbalance addressed using **SMOTE**
- Correlation analysis performed
- No missing values detected

---

## 🤖 Models Implemented

### Python Models:
- Naive Bayes
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- Neural Network (MLP)

### RapidMiner Models:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosted Trees
- Deep Learning
- Naive Bayes
- Support Vector Machine

---

## 📊 Model Performance

| Model                   | Python Accuracy | RapidMiner Accuracy |
|------------------------|------------------|----------------------|
| Naive Bayes            | 0.72             | 0.72                 |
| Logistic Regression    | 0.81             | 0.77                 |
| Decision Tree          | 0.80             | 0.68                 |
| Random Forest          | 0.90             | 0.67                 |
| Gradient Boosting      | 0.88             | 0.84                 |
| SVM                    | 0.88             | 0.64                 |
| Neural Network (MLP)   | 0.91             | 0.81                 |

> **Gradient Boosting and Neural Networks** outperformed other models in both tools, with high accuracy and reliability.

---

## 🔍 Key Insights
- **Top predictors**: OverTime, MonthlyIncome, JobSatisfaction
- **Models**: Neural Networks (Python) and Gradient Boosting (RapidMiner) showed the best predictive performance.
- **RapidMiner** provided visual interpretability and ease of use for benchmarking Python models.

---

## 🚀 Deployment
- The best deep learning model was saved as `best_model.pkl` using Python's `pickle`.
- Future employee data can be passed to this model for real-time attrition prediction.

---

## 💡 Real-World Applications
- **Retention Strategies**: Focus on employees with high overtime and low satisfaction
- **HR Automation**: Integrate attrition prediction in HR systems
- **Monitoring**: Retrain models periodically with updated data

---

## 📚 Learnings
- Gained hands-on experience with SMOTE, ensemble methods, and model tuning.
- Learned to use RapidMiner Auto Model for quick, explainable ML benchmarking.
- Understood the CRISP-DM framework for systematic data mining projects.

---

## 🧾 References
- IBM HR Analytics Dataset (Kaggle)
- RapidMiner Documentation & Academy
- Hastie et al. (2009). *The Elements of Statistical Learning*
- Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*
- Chollet, F. (2017). *Deep Learning with Python*

