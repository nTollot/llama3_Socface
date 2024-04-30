import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from load_data import df, random_elements


def visualize_confusion_matrix(df, random_elements):
    df_true = df.iloc[random_elements]
    df_preds = pd.read_csv("test.csv", dtype=object)
    df_preds.set_index(df_preds.columns[0], inplace=True)
    df_preds.set_index(df_true.index, inplace=True)

    df_true[df_true=='idem'] = np.NaN
    df_preds[df_preds=='idem'] = np.NaN

    class_indices = {col: idx for idx, col in enumerate(df_true.columns)}
    n_classes = len(df_true.columns)
    similarity_matrix = np.zeros((n_classes, n_classes + 1), dtype=np.int32)

    for index, row in df_true.iterrows():
        prediction_row = df_preds.loc[index]
        for gt_col, gt_value in row.dropna().items():
            gt_index = class_indices[gt_col]
            found = False
            for pred_col, pred_value in prediction_row.items():
                if isinstance(pred_value, pd.Series):
                    pred_value = pred_value.iloc[0]
                if pd.notna(pred_value) and str(gt_value) in str(pred_value):
                    pred_index = class_indices[pred_col]
                    similarity_matrix[gt_index, pred_index] += 1
                    found = True
                    break

            if not found:
                # Incrémenter pour prédiction manquante
                similarity_matrix[gt_index, n_classes] += 1

    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_matrix, annot=True, fmt='n', cmap='Blues', xticklabels=list(df_true.columns)+['Non classifié'], yticklabels=df_true.columns)
    plt.xlabel('Colonnes de Prédiction')
    plt.ylabel('Colonnes de Ground truth')
    plt.title('Matrice de Confusion')
    plt.show()

if __name__=='__main__':
    visualize_confusion_matrix(df, random_elements)
