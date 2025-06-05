import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import joblib
import os

# Load the dataset
df = pd.read_csv("Household energy bill data.csv")

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("(", "").str.replace(")", "")

# Define features and target
features = ['Monthly_Income', 'No_of_AC', 'No_of_Fans', 'No_of_Rooms']
target = 'Electricity_Bill_Rs'

# Drop rows with missing values
df = df.dropna(subset=features + [target])

# Split data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = SVR(kernel='linear')
model.fit(X_train, y_train)

# Save the model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/svm_model.joblib")

print("✅ Model trained and saved successfully as models/svm_model.joblib")
