import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """Load the CSV file into a DataFrame"""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"[INFO] Data loaded successfully from: {self.file_path}")
            return self.df
        except FileNotFoundError:
            print(f"[ERROR] File not found at: {self.file_path}")
        except Exception as e:
            print(f"[ERROR] An error occurred: {e}")
