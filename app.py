from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# ----------------------------
# Sample Training Data
# ----------------------------

data = {
    "Age": [22,25,28,30,35,40,45,50,23,27,32,38,42,48,55],
    "Income": [20000,30000,35000,50000,60000,70000,80000,90000,25000,40000,55000,65000,75000,85000,95000],
    "Loan": [40000,15000,18000,20000,25000,30000,35000,40000,45000,15000,20000,25000,30000,35000,40000],
    "History": [0,1,1,1,1,1,1,1,0,1,1,1,1,1,1],
    "Approved": [0,1,1,1,1,1,1,1,0,1,1,1,1,1,1]
}

df = pd.DataFrame(data)

X = df[["Age","Income","Loan","History"]]
y = df["Approved"]

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X,y)

# ----------------------------
# Routes
# ----------------------------

@app.route("/", methods=["GET","POST"])
def home():

    result = None
    risk = ""
    score = 0

    if request.method == "POST":

        age = int(request.form["age"])
        income = float(request.form["income"])
        loan = float(request.form["loan"])
        history = int(request.form["history"])

        prediction = model.predict([[age,income,loan,history]])[0]

        probability = model.predict_proba([[age,income,loan,history]])[0]

        score = round(max(probability)*100)

        if prediction == 1:

            result = "Loan Approved"

            if score >= 85:
                risk = "Low Risk 🟢"
            elif score >= 70:
                risk = "Medium Risk 🟡"
            else:
                risk = "High Risk 🔴"

        else:

            result = "Loan Rejected"

            if score >= 70:
                risk = "Medium Risk 🟡"
            else:
                risk = "High Risk 🔴"

    return render_template(
        "index.html",
        result=result,
        score=score,
        risk=risk
    )

if __name__ == "__main__":
    app.run(debug=True)