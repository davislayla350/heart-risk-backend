import joblib
import numpy as np

# load the model
model = joblib.load("model/heart_disease_model.pkl")

# sample input – should be HIGH risk
sample = np.array([[68, 1, 3, 149, 300, 1, 2, 100, 1, 3.5, 0]])

# make prediction
prediction = model.predict(sample)

# output result
print("Prediction:", "High Risk" if prediction[0] == 1 else "Low Risk")
