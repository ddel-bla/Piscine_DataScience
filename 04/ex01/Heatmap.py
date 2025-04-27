import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr
import seaborn as sb
from sklearn import preprocessing


class Heatmap:
    def __init__(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(cur_dir)
        self.csv_dir = os.path.join(parent_dir, "sources/")
        self.table = "items"
        if not os.path.exists(self.csv_dir):
            os.makedirs(self.csv_dir)
        self.filename_test = "Test_knight.csv"
        self.filename_train = "Train_knight.csv"
        self.filename_prediction = "predictions.txt"
        self.filename_truth = "truth.txt"
        self.strong_features = ["Empowered", "Prescience"]
        self.weak_features = ["Midi-chlorien", "Push"]
        self.filename_test = os.path.join(self.csv_dir, self.filename_test)
        self.filepath_train = os.path.join(self.csv_dir, self.filename_train)
        self.filepath_pred = os.path.join(self.csv_dir, self.filename_prediction)
        self.filepath_truth = os.path.join(self.csv_dir, self.filename_truth)

    def normalize_df(self, df):
        x = df.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df_normalized = pd.DataFrame(x_scaled, columns=df.columns)
        return df_normalized

    def compute_correlations(self, df, target="knight"):
        correlations = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                corr, _ = pointbiserialr(df[col], df[target])
                correlations[col] = corr
        return sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    def plot_correlation_heatmap(self, df):
        corr = df.corr()
        sb.heatmap(corr, annot=False, cmap="magma")
        plt.title("Correlation Heatmap")
        plt.show()

    def run(self):
        df_train = pd.read_csv(self.filepath_train, sep=",")

        try:
            df_train["knight"] = df_train["knight"].map({"Sith": 1, "Jedi": 0})
            df_train_norm = self.normalize_df(df_train)
            self.plot_correlation_heatmap(df_train_norm)
        except Exception as e:
            print(f"Error: {e}")


def main():
    a = Heatmap()
    a.run()


if __name__ == "__main__":
    main()
