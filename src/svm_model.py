from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class SVMModel:
    def __init__(self):
        self.model = SVR()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        print("[INFO] SVM model trained successfully.")

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print(f"[INFO] Evaluation Results:\nMSE: {mse:.2f}, R²: {r2:.2f}")
        return predictions

    def save_model(self, path="svm_model.pkl"):
        joblib.dump(self.model, path)
        print(f"[INFO] Model saved to {path}")
