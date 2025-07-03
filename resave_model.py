import joblib

# Load your original model
model = joblib.load("model/heart_disease_model.pkl")

# Re-save it with compression so it's safe and portable
joblib.dump(model, "model/safe_heart_model.pkl", compress=3)

print("✅ Model resaved as safe_heart_model.pkl")
