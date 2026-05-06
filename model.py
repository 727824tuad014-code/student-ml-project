import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# load dataset
df = pd.read_csv("student_data.csv")

X = df[['hours', 'attendance', 'sleep']]
y = df['marks']

model = LinearRegression()
model.fit(X, y)

# save model
joblib.dump(model, "model.pkl")

def predict_performance(hours, attendance, sleep):
    model = joblib.load("model.pkl")
    return model.predict([[hours, attendance, sleep]])[0]