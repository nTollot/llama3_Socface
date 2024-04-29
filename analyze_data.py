import numpy as np
import pandas as pd
import re

def split_and_strip(data: str) -> list:
    # Divise la chaîne de caractère selon : et \n, et supprime les espaces inutiles
    return [item.strip() for item in re.split(r'\n|:', data)]

def update_dataframe_by_index(info_list, df, index):
    # Met à jour le data frame des données en remplissant par NaN les valeurs non précisées
    info_dict = {}
    keys = df.columns.tolist()
    for i, item in enumerate(info_list[:-1]):
        if item in keys:
            if (i+1 < len(info_list) and info_list[i+1] not in keys and not info_list[i+1].startswith("Note")):
                info_dict[item] = info_list[i+1]
            else:
                info_dict[item] = np.NaN
    for key in keys:
        if key in info_dict:
            df.at[index, key] = info_dict[key]
        else:
            df.at[index, key] = np.NaN

def create_df(d, col, index):
    # Met les données sous la forme d'un dataframe
    df = pd.DataFrame(index=index, columns=col)
    for idx, data in enumerate(d):
        update_dataframe_by_index(split_and_strip(data), df, index[idx])
    return df


if __name__=='__main__':
    # Exemple d'utilisation :
    d = [
        "Voici la catégorisation des informations pour le cas suivant :\n\nnom de famille (non chef): Ferazzi\nprénom: Auguste\nprofession: vitrier\nâge: 30\nstatut civil: Garçon\nnationalité: Piémontaise\n\nNotez que les informations se suivent, donc il n'y a",
        "Voici la catégorisation des informations pour le cas suivant :\n\nnom de famille (non chef): Machol\nprénom: Pierre\nprofession: vitrier\nâge: 24\nstatut civil: Garçon\nnationalité: Piémontaise\n\nLes informations sont séparées par des deux-points pour indiquer la classe",
        "Voici la classification des informations pour le cas suivant :\n\nnom de famille (non chef): Desbois\nprénom: Alexandre\nlien familial: prop\nâge: 48\nstatut civil: Homme marié\nnationalité: française\n\nNote : Les éléments 'prop re' sont considérés comme un lien",
    ]
    columns = ['nom de famille (non chef)', 'prénom', 'profession', 'âge', 'statut civil', 'nationalité']
    names = ["Ferazzi", "Machol", "Desbois"]

    df = create_df(d, columns, names)
    print(df)
