import json
import numpy as np
import pandas as pd
import re

def load_data(data_path="entities.json"):
    with open(data_path, 'r') as file:
        data = json.load(file)
    content = [value for sublist in data.values() if sublist !=
               '' for value in sublist.split('\n')]
    return content

def remove_symbols(content, symbols):
    table = str.maketrans('', '', ''.join(symbols))
    return [text.translate(table) for text in content]

def parse_content(symbols, content):
    result = {symbol: [] for symbol in symbols}
    for text in content:
        for symbol in symbols:
            start = text.find(symbol)
            if start != -1:
                end = find_end(text, start, symbols)
                info = text[start + 1:end].strip()
                result[symbol].append(info)
    return {symbol: set(infos) for symbol, infos in result.items()}

def find_end(text, start, symbols):
    end = len(text)
    for other_symbol in symbols:
        if other_symbol != text[start] and other_symbol in text[start + 1:]:
            pos = text.find(other_symbol, start + 1)
            if pos != -1 and pos < end:
                end = pos
    return end

def transform_tagged_string(tagged_str, tags):
    parts = re.split("([ⒶⒷⒸⒺⒻⒽⒾⓀⓁⓂⓄⓅ])", tagged_str)
    transformed = " ".join(
        f"{tags[parts[i]]}: {parts[i+1].strip()}" for i in range(1, len(parts), 2))
    return transformed.strip()

def to_df(content, tags):
    rows, indices = [], []
    for entry in content:
        info, index = extract_info(entry, tags)
        rows.append(info)
        indices.append(index)
    return pd.DataFrame(rows, index=indices)

def extract_info(entry, tags):
    info_dict = {}
    clean_entry = entry
    for symbol, tag in tags.items():
        pattern = f'{symbol}([^Ⓐ-Ⓟ]+)'
        match = re.search(pattern, entry)
        if match:
            info_dict[tag] = match.group(1).strip()
            clean_entry = clean_entry.replace(match.group(0), match.group(1))
        else:
            info_dict[tag] = np.NaN
    return info_dict, ' '.join(clean_entry.split())

symbols_rm = ['Ⓐ', 'Ⓑ', 'Ⓒ', 'Ⓔ', 'Ⓕ', 'Ⓗ', 'Ⓘ', 'Ⓚ', 'Ⓛ', 'Ⓜ', 'Ⓞ', 'Ⓟ']
content = load_data()
content_rm = remove_symbols(content, symbols_rm)
categorized_data = parse_content(symbols_rm, content_rm)

tags = {'Ⓐ': 'âge', 'Ⓑ': 'date de naissance', 'Ⓒ': 'statut civil', 'Ⓔ': 'employeur', 'Ⓕ': 'prénom', 'Ⓗ': 'lien familial', 'Ⓘ': 'division',
        'Ⓚ': 'nationalité', 'Ⓛ': 'observation', 'Ⓜ': 'profession', 'Ⓞ': 'nom de famille (non chef)', 'Ⓟ': 'nom de famille (chef)'}

prompt_sys = "L’objectif est de rassembler et de traiter par reconnaissance automatique de l’écriture toutes les listes nominatives manuscrites des recensements de 1836 à 1936. Il est nécessaire de typer chaque information reconnue par le modèle. Définition des classes/types : âge, date de naissance, status civil, employeur, prénom, lien familial, division, nationalité, observation, profession, nom de famille (non chef), nom de famille.  Exemples : - Brivadier Jean Marie fils 13 française -> nom de famille (non chef): Brivadier prénom: Jean Marie lien familial: fils âge: 13 nationalité: française - Magat Marie s.p 48 Femme mariée -> nom de famille (non chef): Magat prénom: Marie profession: s.p âge: 48 statut civil: Femme mariée - Bourasseau Louis domestique 18 Garçon->  nom de famille (non chef): Bourasseau prénom: Louis lien familial: domestique âge: 18 statut civil: Garçon - Desnoues Florentine idem 12 Fille -> nom de famille (non chef): Desnoues prénom: Florentine profession: idem âge: 12 statut civil: Fille. Quelques exemples d'éléments par classe : âge : 16,  48, 12 1/2 ; date de naissance : 1839, 1915, 1848 ; statut civil : Femme mariée, Homme marié, Veuf ; employeur : de Beauvais, mécanicien - Me Rime et Cie Couverture, Ch fer Etat ; prénom : Josepg, Emery, Blaize, lien familial : idem 2eme, sœur, neveu ; division : Frystas, Paray le Monial, Cerince (Loiret) ; nationalité : Piémontaise, belge, polonais ; observation : occupé dans une autre commune, veuve Charreyre, f Bronne ; profession : religieuse, ep Mathern, ouvrier charron ; nom de famille (non chef) : de Beauvais, Blaize, Emery ; nom de famille (chef) : de Beauvais, Maosonneuve, Coz. Tous les informations fournies doivent être catégorisées, et par des deux-points séparants la classe avec son élément comme dans les exemples, mais uniquement avec les classes utilisées."

df = to_df(content, tags)

prompt_user = "Voici le cas précédent {}, maintenant traite le cas suivant sachant que les informations se suivent : {} :"
np.random.seed(53)
random_elements = np.random.randint(0, len(content), 500)

def generate_data(idx):
    prev_data = transform_tagged_string(content[idx - 1], tags)
    return prompt_sys, prompt_user.format(prev_data, content_rm[idx])
