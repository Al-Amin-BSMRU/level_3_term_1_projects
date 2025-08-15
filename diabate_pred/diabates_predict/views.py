from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset and train model once
df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

def home(request):
    return render(request, "home.html")

def predict(request):
    result = None
    if request.method == "POST":
        # Get form values
        val1 = float(request.POST.get("pregnancies"))
        val2 = float(request.POST.get("glucose"))
        val3 = float(request.POST.get("blood_pressure"))
        val4 = float(request.POST.get("skin_thickness"))
        val5 = float(request.POST.get("insulin"))
        val6 = float(request.POST.get("bmi"))
        val7 = float(request.POST.get("diabetes_pedigree"))
        val8 = float(request.POST.get("age"))

        # Predict
        prediction = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"

    return render(request, "predict.html", {"result": result})
