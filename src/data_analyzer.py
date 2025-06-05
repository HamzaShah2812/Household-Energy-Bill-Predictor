import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class DataAnalyzer:
    def __init__(self, df: pd.DataFrame, output_dir="graphs"):
        self.df = df
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_histograms(self):
        """Plots histograms for all numeric columns and saves them"""
        self.df.hist(bins=20, figsize=(14, 10))
        plt.suptitle("Distribution of Numerical Columns", fontsize=16)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "histograms.png")
        plt.savefig(save_path)
        plt.close()
        print(f"[INFO] Histograms saved to {save_path}")

    def plot_heatmap(self):
        """Plots a heatmap of correlations"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.df.corr(numeric_only=True), annot=True, cmap='coolwarm')
        plt.title("Correlation Heatmap")
        save_path = os.path.join(self.output_dir, "heatmap.png")
        plt.savefig(save_path)
        plt.close()
        print(f"[INFO] Heatmap saved to {save_path}")

    def run_all(self):
        """Runs all analysis and saves graphs"""
        self.plot_histograms()
        self.plot_heatmap()
