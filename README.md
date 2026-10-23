# Credit Scoring Model

## Project Overview
This project was developed as part of my CodeAlpha Machine Learning Internship. It uses machine learning to classify credit applicants into good or bad credit risk categories based on their financial and personal information.

A Flask web application allows users to enter applicant details and view the model's prediction.

**Note:** This is an educational project using the German Credit Dataset. It does not calculate an official CIBIL score and should not be used to make real lending decisions.

## Objectives
- Preprocess and analyze credit-related data.
- Train and compare multiple machine learning models.
- Evaluate model performance using classification metrics.
- Save the best-performing model.
- Build a simple web application for predictions.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Flask
- HTML and CSS
- OpenML German Credit Dataset

## Dataset
The project uses the German Credit Dataset (`credit-g`) from OpenML. It includes features such as credit history, loan duration, credit amount, savings, employment, age, and housing.

## Machine Learning Models
The following models were trained and evaluated:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 72.00% | 81.82% | 77.14% | 79.41% | 79.83% |
| Logistic Regression | 67.50% | 83.78% | 66.43% | 74.10% | 76.00% |
| Decision Tree | 63.00% | 81.13% | 61.43% | 69.92% | 63.63% |

The Random Forest model achieved the highest ROC-AUC among the evaluated models.

## Features
- Data preprocessing and categorical encoding.
- Training and comparison of multiple classifiers.
- Evaluation using accuracy, precision, recall, F1-score, and ROC-AUC.
- Saving the trained model using Joblib.
- Web-based predictions using Flask.

## Running the Project

### 1. Clone the repository
```bash
git clone https://github.com/choudarysurakshitha/Credit-Scoring-Model.git
cd Credit-Scoring-Model
```

### 2. Create and activate a virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
python -m pip install -r requirements.txt
```

### 4. Start the web application
```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

### Retrain the models
```bash
python -m src.train
```

## Project Structure
```text
Credit_Scoring/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── train.py
│   └── predict.py
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── models/
└── reports/
```

## Limitations
- The model uses the German Credit Dataset, not Indian CIBIL data.
- Predictions depend on the training data and model performance.
- The project is intended for educational and demonstration purposes only.

## Future Enhancements
- Hyperparameter tuning and further model evaluation.
- More visualizations and explainable predictions.
- Improved validation and web interface features.

## Conclusion
This project demonstrates an end-to-end machine learning workflow, including data preprocessing, model training, evaluation, and deployment through a Flask web application.

**Developed as part of the CodeAlpha Machine Learning Internship.**