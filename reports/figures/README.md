# Credit Scoring Model

## Project Overview
The Credit Scoring Model is a Machine Learning project developed as part of my **CodeAlpha Machine Learning Internship**. It predicts whether an applicant belongs to a good or bad credit category based on financial and personal information.

The project compares multiple machine learning algorithms and provides a simple web interface built with Flask, HTML, and CSS to enter applicant details and view a prediction.

**Note:** This is an educational credit classification project, not an official CIBIL score calculator or a system for making real lending decisions.

## Objectives
- Understand and preprocess credit-related data.
- Train and compare multiple Machine Learning models.
- Evaluate model performance using standard classification metrics.
- Save the best-performing model for future predictions.
- Build a user-friendly web application for predictions.

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
This project uses the **German Credit Dataset (`credit-g`)** available through OpenML. It contains applicant information such as credit history, loan duration, credit amount, employment, savings, age, and housing.

The target categories represent good and bad credit risk. This dataset is not Indian CIBIL data.

## Machine Learning Algorithms
The following algorithms were trained and compared:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

The models were evaluated using accuracy, precision, recall, F1-score, and ROC-AUC.

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 72.00% | 81.82% | 77.14% | 79.41% | 79.83% |
| Logistic Regression | 67.50% | 83.78% | 66.43% | 74.10% | 76.00% |
| Decision Tree | 63.00% | 81.13% | 61.43% | 69.92% | 63.63% |

**Best-performing model:** Random Forest, based on ROC-AUC.

## Features
- Data preprocessing and categorical feature encoding.
- Training and comparison of multiple ML models.
- Model evaluation using classification metrics.
- Saving the trained model using Joblib.
- Flask-based web interface for entering applicant details.
- Prediction of good or bad credit category.
- Display of the model's estimated probability for the good credit category, when available.

## Project Structure

```text
Credit_Scoring/
├── app.py
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── train.py
│   └── predict.py
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── models/
│   └── credit_scoring_model.joblib
├── data/
│   └── processed/
└── reports/
    └── figures/
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Credit_Scoring
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual GitHub repository URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Running the Application

Start the Flask application:

```bash
python app.py
```

Open the following address in your browser:

```text
http://127.0.0.1:5000
```

Enter the applicant details in the form and submit them to view the predicted credit category.

## Training the Model

To train the models again, run:

```bash
python -m src.train
```

The trained model and evaluation outputs are saved in the project's model, data, and reports folders.

## Limitations
- The model uses the German Credit Dataset and does not calculate an official CIBIL score.
- Predictions depend on the quality and limitations of the training dataset.
- The model is intended for learning and demonstration purposes, not real-world financial decisions.

## Future Enhancements
- Test the model on additional datasets.
- Improve model performance through hyperparameter tuning.
- Add more visualizations and explainable predictions.
- Improve the web interface and input validation.

## Conclusion
This project demonstrates the application of Machine Learning to credit risk classification. It covers data preprocessing, model training, performance evaluation, model saving, and deployment through a Flask web application.

**Developed as part of the CodeAlpha Machine Learning Internship.**