import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

class ConfusionMatrix:
    def __init__(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(cur_dir)
        self.csv_dir = os.path.join(parent_dir, "sources/")
        self.table = "items"
        if not os.path.exists(self.csv_dir):
            os.makedirs(self.csv_dir)
        self.filename_test = "Test_knight.csv"
        self.filename_train = "Train_knight.csv"
        self.filename_prediction = sys.argv[1] if len(sys.argv) > 2 else "predictions.txt"
        self.filename_truth = sys.argv[2] if len(sys.argv) > 2 else "truth.txt"
        self.filename_test = os.path.join(self.csv_dir, self.filename_test)
        self.filepath_train = os.path.join(self.csv_dir, self.filename_train)
        self.filepath_pred = os.path.join(self.csv_dir, self.filename_prediction)
        self.filepath_truth = os.path.join(self.csv_dir, self.filename_truth)

    def normalize_df(self, df):
        return (df - df.min()) / (df.max() - df.min())

    def clean_txt_file(self, filepath):
        with open(filepath, "r") as file:
            content = file.read()
        array = content.split()
        # Replace Jedi with 1 and Sith with 0
        array = [1 if x == "Jedi" else 0 for x in array]
        return array

    def confusion_matrix(self, true, pred):
        if len(true) != len(pred) or len(true) == 0:
            raise ValueError("Length mismatch")
        classes = set(true + pred)
        num_classes = len(classes)
        shape = (num_classes, num_classes)
        mat = np.zeros(shape)
        n = max(len(true), len(pred))
        for i in range(num_classes):
            for j in range(num_classes):
                for k in range(n):
                    if true[k] == i:
                        if pred[k] == j:
                            mat[i][j] = mat[i][j] + 1
        return mat

    def plot_confusion_matrix(self, cm, classes, title="Confusion matrix"):
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()

        # Add labels to the axes
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        # Annotate each cell with the corresponding value
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    format(cm[i, j], ".0f"),
                    ha="center",
                    va="center",
                    color="black",
                )

        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.show()

    def run(self):
        df_train = pd.read_csv(self.filepath_train, sep=",")
        df_train["knight"] = df_train["knight"].map({"Sith": 0, "Jedi": 1})
        array_truth = self.clean_txt_file(self.filepath_truth)
        array_pred = self.clean_txt_file(self.filepath_pred)

        try:
            confusion_matrix = self.confusion_matrix(array_truth, array_pred)
            confusion_matrix = np.flip(confusion_matrix)
            TP = confusion_matrix[0][0]
            TN = confusion_matrix[1][1]
            FP = confusion_matrix[0][1]
            FN = confusion_matrix[1][0]

            accuracy_jedi = (TP + TN) / (TP + TN + FP + FN)
            precision_jedi = TP / (TP + FP)
            precision_sith = TN / (TN + FN)
            recall_jedi = TP / (TP + FN)
            recall_sith = TN / (TN + FP)
            f1_score_jedi = (2 * (precision_jedi * recall_jedi)) / (precision_jedi + recall_jedi)
            f1_score_sith = 2 * (precision_sith * recall_sith) / (precision_sith + recall_sith)
            total_jedi = confusion_matrix[0][0] + confusion_matrix[0][1]
            total_sith = confusion_matrix[1][0] + confusion_matrix[1][1]
            print(f"            precision   recall   f1_score     total")
            print(f"Jedi:       {recall_jedi:.2f}        {precision_jedi:.2f}     {f1_score_jedi:.2f}          {total_jedi:.2f}")
            print(f"Sith:       {recall_sith:.2f}        {precision_sith:.2f}     {f1_score_sith:.2f}          {total_sith:.2f}")
            print(f"accuracy                         {accuracy_jedi:.2f}          {total_jedi+total_sith:.2f}")
            print(f"Confusion matrix: \n{confusion_matrix}")
            self.plot_confusion_matrix(confusion_matrix, ["0", "1"])
        except Exception as e:
            print(f"Error: {e}")


def main():
    a = ConfusionMatrix()
    a.run()


if __name__ == "__main__":
    main()
