🚢 Titanic Survival Predictor: End-to-End ML Pipeline & Dashboard
This repository showcases a comprehensive, production-ready Data Science and Machine Learning workflow applied to the classic Titanic dataset. The project heavily emphasizes advanced data preprocessing, robust feature engineering, and ensemble modeling to deliver high-accuracy predictions via an interactive web dashboard.

🧠 Key Data Science & ML Highlights
1. Advanced Data Preprocessing (Data Cleaning)
Intelligent Imputation: Instead of using simple mean or median strategies which can distort data variance, this pipeline implements KNNImputer to handle missing values in the Age column, preserving the underlying distribution based on passenger similarities.

Feature Scaling: Applied StandardScaler to distance-sensitive numerical features (Age and Fare) to ensure optimal convergence for linear and distance-based algorithms.

2. Domain-Specific Feature Engineering
FamilySize: Combined SibSp (siblings/spouses) and Parch (parents/children) plus the passenger themselves to capture family dynamics aboard.

IsAlone: Derived a binary indicator from FamilySize to model the strong socio-behavioral survival contrast between solo travelers and families.

3. Ensemble Modeling & Optimization
Built a robust VotingClassifier (Hard Voting) combining three diverse algorithms to reduce variance and improve generalization:

XGBoost: High-performance gradient boosted decision trees.

Random Forest: Tree-based bagging ensemble to reduce overfitting.

Logistic Regression: A reliable linear baseline.

4. Dashboard Deployment
Developed an interactive, real-time prediction dashboard using Streamlit.

The dashboard loads the pre-trained ensemble model and the scaler artifact, allowing users to tweak passenger parameters and instantly see survival probabilities.

💻 Tech Stack
Language: Python

Data Manipulation & Preprocessing: Pandas, NumPy, Scikit-Learn (KNNImputer, StandardScaler)

Machine Learning: Scikit-Learn (VotingClassifier, RandomForestClassifier, LogisticRegression), XGBoost

Deployment & Frontend: Streamlit, Joblib

🚀 How to Run the Project Locally
Follow these steps to clone the repository and run the Streamlit dashboard on your Mac/PC:

Clone the Repository:
git clone https://github.com/amirhosseinhajizadeh2007/Titanic-survival-predictor.git
cd Titanic-survival-predictor

Set Up a Virtual Environment & Install Dependencies:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Launch the Dashboard:
streamlit run app.py

📈 Project Workflow & Architecture
EDA & Visualizations: Analyzing passenger distributions, correlations, and target impacts.

Pipeline Processing: Transforming raw user inputs through the calibrated StandardScaler.

Inference: Passing scaled vectors to the trained VotingClassifier for the final Survived (1) or Not Survived (0) output.

Author: AmirHossein HajiZadeh
