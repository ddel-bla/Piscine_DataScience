import pandas as pd
import os
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import sys
from sklearn.neighbors import KNeighborsClassifier
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

class KNN:
  def __init__(self):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(cur_dir)
    self.filename__test_knight = sys.argv[2] if len(sys.argv) > 2 else "Test_knight.csv"
    self.filename_train_knight = sys.argv[1] if len(sys.argv) > 2 else "Train_knight.csv"
    self.csv_dir = os.path.join(parent_dir, 'sources/')
    self.table = 'items'
    if not os.path.exists(self.csv_dir):
        os.makedirs(self.csv_dir)
    self.filepath_train = os.path.join(self.csv_dir, self.filename_train_knight)
    self.filepath_test = os.path.join(self.csv_dir, self.filename__test_knight)
    self.filename_training = "Training_knight.csv"
    self.filename_validation = "Validation_knight.csv"
    self.filename_output = "KNN.txt"
    self.filepath_training = os.path.join(self.csv_dir, self.filename_training)
    self.filepath_validation = os.path.join(self.csv_dir, self.filename_validation)
    self.filepath_output = os.path.join(self.csv_dir, self.filename_output)


  def normalize_df(self, df):
    x = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_normalized = pd.DataFrame(x_scaled, columns=df.columns)
    return df_normalized

  def select_features(self, df):
    X = df.drop(columns=['knight'])
    X = add_constant(X)
    vif_df = pd.DataFrame()
    vif_df['features'] = X.columns
    vif_df['VIF'] = [variance_inflation_factor(X.values, i)
                            for i in range(len(X.columns))]
    vif_df = vif_df[vif_df['features'] != 'const']
    print(f"VIF DataFrame: \n{vif_df}")
    return vif_df[vif_df['VIF'] < 400]['features'].tolist()

  def pca(self, X, variance_threshold=0.90):
        print(f"X head: \n{X.head()}")
        # 1. Center
        X_centered = X - np.mean(X, axis=0)

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
        feature_contributions = np.abs(projection_matrix)
        important_indices = np.argmax(feature_contributions, axis=0)
        selected_features = X.columns[important_indices]
        selected_features = selected_features.tolist()
        print(f"selected_features = {type(selected_features)}")
        

        return selected_features, explained_variance_ratio, cumulative_variance, num_components
  
  def run(self):
    try:
        # Data Preparation
        # Splitting the Dataset
        df_training = pd.read_csv(self.filepath_training, sep=',')
        df_training = self.normalize_df(df_training)
        #print(f"fd training columns = {df_training.columns}")
        selected_features = self.select_features(df_training)
        # selected_features, _, _, _ = self.pca(df_training, variance_threshold=1)
        print(f"----> selected_features = {selected_features}")
        df_training = df_training[selected_features + ['knight']]
        X_train = df_training.drop(columns=['knight'])
        y_train = df_training['knight']

        df_validation = pd.read_csv(self.filepath_validation, sep=',')
        df_validation = self.normalize_df(df_validation)
        df_validation = df_validation[selected_features + ['knight']]
        X_validation = df_validation.drop(columns=['knight'])
        y_validation = df_validation['knight']

        # Training the model
        K = []
        f1_scores = []

        for k in range(2, 21):
           clf = KNeighborsClassifier(n_neighbors=k)
           clf.fit(X_train, y_train)
           y_pred = clf.predict(X_validation)
           f1_score_value = f1_score(y_validation, y_pred, average='weighted')
           K.append(k)

           f1_scores.append(f1_score_value)
          
        # Evaluating the model
        for k, f1 in zip(K, f1_scores):
            print(f"F1 Score for K={k}: {f1}")

        # Plot f1 scores
        plt.plot(K, f1_scores, marker='o')
        plt.title('F1 Score vs K')
        plt.xlabel('K')
        plt.ylabel('F1 Score')
        plt.xticks(K)
        plt.grid()
        plt.show()

        with open(self.filepath_output, 'w') as f:
            for pred in y_pred:
                if pred == 0:
                    f.write('Sith\n')
                else:
                    f.write('Jedi\n')
        
        # X_test = pd.read_csv(self.filepath_test, sep=',')
        # X_test = self.normalize_df(X_test)
        # X_test = X_test[selected_features]
        # print(f"X_test head: \n{X_test.head()}")
        # y_test = clf.predict(X_test)
        # print("Predictions for the test set:")
        # for pred in y_test:
        #     if pred == 0:
        #         print('Sith')
        #     else:
        #         print('Jedi')

    except Exception as e:
      print(f"Error: {e}")

def main():
  a = KNN()
  a.run()

if __name__ == "__main__":
  main()