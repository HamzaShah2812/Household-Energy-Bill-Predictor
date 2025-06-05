from sklearn.model_selection import train_test_split
import pandas as pd

class DataSplitter:
    def __init__(self, df: pd.DataFrame, target_column: str, test_size=0.2, random_state=42):
        self.df = df
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state

    def split(self):
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        print(f"[INFO] Data split into train and test sets.")
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test
