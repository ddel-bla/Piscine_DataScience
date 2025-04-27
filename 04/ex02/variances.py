import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np

class Variances:
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

    def select_features(self, sorted_variances, total_variance):
        explained = 0
        num_components = 0
        explained_variances = []
        for var in sorted_variances:
            explained += var
            num_components += 1
            explained_variances.append(explained / total_variance)
            print(f"Explained variance: {explained / total_variance:.2%}")
        return num_components, explained_variances

    def plot_selected_variances(self, explained_variances, num_components):
        plt.figure(figsize=(10, 6))
        plt.plot(
            range(1, len(explained_variances) + 1),
            explained_variances,
            marker="o",
            linestyle="--",
        )
        plt.title("Explained Variance by Number of Components")
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.axhline(y=0.9, color="r", linestyle="--", label="90% Explained Variance")
        plt.legend()
        plt.grid()
        plt.show()

    def standardize_df(self, df_train):
        train_columns = df_train.columns.drop("knight")
        knight_train = df_train["knight"]
        scaler = preprocessing.StandardScaler()
        X_train_scaled = scaler.fit_transform(df_train.drop(columns=["knight"]))
        df_train_scaled = pd.DataFrame(X_train_scaled, columns=train_columns)
        df_train_scaled["knight"] = knight_train
        return df_train_scaled

    def pca(self, X, variance_threshold=0.90):
        X_centered = X - np.mean(X, axis=0)
        # 1. Center

        # 2. Covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # 3. Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 4. Sort by decreasing eigenvalues
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # 5. Cumulative explained variance
        explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
        cumulative_variance = np.cumsum(explained_variance_ratio)

        # Find number of components to reach the threshold
        num_components = np.searchsorted(cumulative_variance, variance_threshold) + 1

        # 6. Project data
        projection_matrix = eigenvectors[:, :num_components]
        X_reduced = np.dot(X_centered, projection_matrix)

        return X_reduced, explained_variance_ratio, cumulative_variance, num_components

    def run(self):
        df_train = pd.read_csv(self.filepath_train, sep=",")

        try:
            df_train["knight"] = df_train["knight"].map({"Sith": 1, "Jedi": 0})
            standardized_df_train = self.standardize_df(df_train)
            X = standardized_df_train.drop(columns=["knight"])
            _, explained_variance_ratio, cumulative_variance, num_components = self.pca(X)
            print(f"explained_variance_ratio: \n{explained_variance_ratio}")
            print(f"cumulative_variance: \n{cumulative_variance}")
            self.plot_selected_variances(cumulative_variance, num_components)
            print(
                f"Number of components to explain 90% variance: {num_components} ou of {len(cumulative_variance)}"
            )
        except Exception as e:
            print(f"Error: {e}")


def main():
    a = Variances()
    a.run()


if __name__ == "__main__":
    main()
