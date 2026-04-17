# 🏥 Patient Diagnosis Classification System

This project implements a robust **Decision Tree** classification pipeline to predict patient diagnosis based on clinical features like Blood Pressure, Cholesterol, and BMI. The system is designed to be production-ready with automated reporting and model serialization.

## 🚀 Overview
The project automates the machine learning lifecycle:
- **Synthetic Data Generation:** Can generate data if the dataset is missing.
- **Automated Pipeline:** Handles both categorical (Gender, Smoking) and numerical data seamlessly.
- **Rigorous Evaluation:** Uses both Holdout and K-Fold Cross-Validation.
- **Model Persistence:** Saves the trained model for future use.

## 🛠️ Tech Stack
- **Language:** Python 🐍
- **Core Libraries:**
  - `Scikit-learn`: For the `Pipeline`, `ColumnTransformer`, and `DecisionTreeClassifier`.
  - `Pandas` & `NumPy`: For data engineering.
  - `Matplotlib`: For visualizing the decision tree structure and feature importance.
  - `Openpyxl`: For handling Excel datasets.

## 📊 Project Workflow
1. **Data Handling:** Loads or generates patient records.
2. **Preprocessing:** Uses `OneHotEncoder` for categories and maintains numerical scales.
3. **Training:** Fits a Decision Tree model with optimized parameters.
4. **Validation:** Computes Accuracy, Precision, Recall, and F1-score, plus a Confusion Matrix.
5. **Output:** Generates a `model.pkl` file, a JSON report, and a visual tree diagram.

## ⚙️ Installation & Usage
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YourUsername/Patient-Health-Classifier.git](https://github.com/YourUsername/Patient-Health-Classifier.git)

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt

3. **Run the script:**
   ```bash
   python train_decision_tree_patients.py

## 📝 Key Features
. **Visual Insights:** Generates a full visualization of the Decision Tree logic and feature importance charts.
. **Detailed Reporting:** Saves comprehensive metrics in a report.json for tracking performance.
. **Real-world Readiness:** Uses Pipeline to prevent data leakage and simplify deployment.
