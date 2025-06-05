import pandas as pd

class DataPreprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def handle_missing_values(self):
        """Fill or drop missing values (basic strategy)"""
        # For now, we'll fill numeric columns with median
        self.df.fillna(self.df.median(numeric_only=True), inplace=True)
        print("[INFO] Missing values filled with column medians.")

    def remove_duplicates(self):
        """Remove duplicate rows"""
        initial_shape = self.df.shape
        self.df.drop_duplicates(inplace=True)
        print(f"[INFO] Removed {initial_shape[0] - self.df.shape[0]} duplicate rows.")

    def preprocess(self):
        """Run all preprocessing steps"""
        self.handle_missing_values()
        self.remove_duplicates()
        return self.df
