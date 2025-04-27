import pandas as pd
import os
from sklearn import preprocessing
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


class FeatureSelection:
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

    def run(self):
        try:
            df_train = pd.read_csv(self.filepath_train, sep=",")
            df_train["knight"] = df_train["knight"].map({"Sith": 1, "Jedi": 0})

            # The independent variable set
            X = df_train.drop(columns=["knight"])
            X = add_constant(X)
            # VIF datafram
            vif_df = pd.DataFrame()
            vif_df["features"] = X.columns

            # Caluclatinf VIF for each feature
            vif_df["VIF"] = [
                variance_inflation_factor(X.values, i) for i in range(len(X.columns))
            ]
            vif_df['Tolerance'] = 1 / vif_df["VIF"]
            print(f"VIF DataFrame: \n{vif_df}")

            # Keep only the features that the VIF is less than 5
            selected_features = vif_df[vif_df["VIF"] < 5]["features"].tolist()
            print(f"Selected features: {selected_features}")
            # Drop line where features == "const"
            vif_df = vif_df[vif_df["features"] != "const"]

            print(vif_df.to_string(index=False))
        except Exception as e:
            print(f"Error: {e}")


def main():
    a = FeatureSelection()
    a.run()

if __name__ == "__main__":
    main()
