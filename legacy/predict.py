import joblib
import pandas as pd


MODEL_PATH = "models/final_model.pkl"


# Load saved model
model = joblib.load(MODEL_PATH)

print("\n========================================")
print("       EMPLOYEE ATTRITION PREDICTION")
print("========================================")


# Employee details
employee = {
    "Age": 35,
    "BusinessTravel": "Travel_Rarely",
    "DailyRate": 800,
    "Department": "Sales",
    "DistanceFromHome": 5,
    "Education": 3,
    "EducationField": "Life Sciences",
    "EmployeeCount": 1,
    "EmployeeNumber": 9999,
    "EnvironmentSatisfaction": 3,
    "Gender": "Male",
    "HourlyRate": 70,
    "JobInvolvement": 3,
    "JobLevel": 2,
    "JobRole": "Sales Executive",
    "JobSatisfaction": 3,
    "MaritalStatus": "Single",
    "MonthlyIncome": 5000,
    "MonthlyRate": 15000,
    "NumCompaniesWorked": 2,
    "Over18": "Y",
    "OverTime": "Yes",
    "PercentSalaryHike": 15,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 3,
    "StandardHours": 80,
    "StockOptionLevel": 0,
    "TotalWorkingYears": 8,
    "TrainingTimesLastYear": 3,
    "WorkLifeBalance": 3,
    "YearsAtCompany": 5,
    "YearsInCurrentRole": 3,
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 2
}


# Convert to DataFrame
input_data = pd.DataFrame([employee])


# Make prediction
prediction = model.predict(input_data)[0]


print("\nEmployee Details:")
print(input_data.to_string(index=False))


print("\n========================================")
print("             PREDICTION")
print("========================================")


if prediction == 1:
    print("Prediction: YES - Employee may leave")
else:
    print("Prediction: NO - Employee may stay")


# Probability if available
if hasattr(model, "predict_proba"):

    probabilities = model.predict_proba(
        input_data
    )[0]

    print(
        f"Probability of Staying: "
        f"{probabilities[0] * 100:.2f}%"
    )

    print(
        f"Probability of Leaving: "
        f"{probabilities[1] * 100:.2f}%"
    )


print("\n========================================")