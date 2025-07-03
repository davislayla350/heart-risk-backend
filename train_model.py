# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("data/heart.csv")

# Convert categorical cols to num using Label Encoding
categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Split features and target variable
# Features
X = df.drop('HeartDisease', axis=1) 
# Target label
y = df['HeartDisease']               

# 80% train 20% test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# init and train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Eval model performance
print("Accuracy on test data:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save trained model to a file
joblib.dump(model, 'model.pkl')

print("Model training complete and saved as 'model.pkl'.")
