import pandas as pd
import os
from sklearn import preprocessing
import pandas as pd
import sys
from sklearn.neighbors import KNeighborsClassifier
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier

class Democracy:
  def __init__(self):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(cur_dir)
    self.filename__test_knight = sys.argv[2] if len(sys.argv) > 2 else "Test_knight.csv"
    self.filename_train_knight = sys.argv[1] if len(sys.argv) > 2 else "Train_knight.csv"
    self.csv_dir = os.path.join(parent_dir, 'sources/')
    self.table = 'items'
    if not os.path.exists(self.csv_dir):
        os.makedirs(self.csv_dir)
    self.filename_training = "Training_knight.csv"
    self.filename_validation = "Validation_knight.csv"
    self.filename_output = "Voting.txt"
    self.filepath_training = os.path.join(self.csv_dir, self.filename_training)
    self.filepath_validation = os.path.join(self.csv_dir, self.filename_validation)
    self.filepath_output = os.path.join(self.csv_dir, self.filename_output)
    self.filepath_test = os.path.join(self.csv_dir, self.filename__test_knight)
    self.filepath_train = os.path.join(self.csv_dir, self.filename_train_knight)


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
    print(f"VIF DataFrame: \n{vif_df}")
    return vif_df[vif_df['VIF'] < 50]['features'].tolist()

  def run(self):
    try:
        # Data Preparation
        # Splitting the Dataset
        df_training = pd.read_csv(self.filepath_training, sep=',')
        df_training = self.normalize_df(df_training)
        selected_features = self.select_features(df_training)
        print(f"----> selected_features = {selected_features}")
        df_training = df_training[selected_features + ['knight']]
        X_train = df_training.drop(columns=['knight'])
        y_train = df_training['knight']

        df_validation = pd.read_csv(self.filepath_validation, sep=',')
        df_validation = self.normalize_df(df_validation)
        df_validation = df_validation[selected_features + ['knight']]
        X_validation = df_validation.drop(columns=['knight'])

        # Create different classifiers
        clf1 = KNeighborsClassifier(n_neighbors=6)
        clf2 = RandomForestClassifier(random_state=42)
        clf3 = LogisticRegression(random_state=42)

        labels = ['KNN', 'Random Forest', 'Logistic Regression']
        print('5-fold cross validation:\n')
        for clf, label in zip([clf1, clf2, clf3], labels):
            scores = model_selection.cross_val_score(clf, X_train, y_train, cv=5, scoring='f1_weighted')
            
        # Create a Hard Voting Classifier
        voting_clf_hard = VotingClassifier(
           estimators=[(labels[0], clf1),
                    (labels[1], clf2),
                    (labels[2], clf3)
                    ],
                    voting='hard'
        )
        voting_clf_soft = VotingClassifier(
           estimators=[(labels[0], clf1),
                    (labels[1], clf2),
                    (labels[2], clf3)
                    ],
                    voting='soft'
        )

        labels_new = ['KNN', 'Random Forest', 'Logistic Regression', 'Voting Hard', 'Voting Soft']

        for clf, label in zip([clf1, clf2, clf3, voting_clf_hard, voting_clf_soft], labels_new):
           scores = model_selection.cross_val_score(clf, X_train, y_train, cv=5, scoring='f1_weighted')
           print("f1 score: %0.2f (+/- %0.2f) [%s]"
             % (scores.mean(), scores.std(), label))

        y_pred = voting_clf_hard.fit(X_train, y_train).predict(X_validation)
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
        # y_test = voting_clf_hard.predict(X_test)
        # print("Predictions for the test set:")
        # for pred in y_test:
        #     if pred == 0:
        #         print('Sith')
        #     else:
        #         print('Jedi')
    except Exception as e:
      print(f"Error: {e}")

def main():
  a = Democracy()
  a.run()

if __name__ == "__main__":
  main()