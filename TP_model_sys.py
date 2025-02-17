import numpy as np
import unicodedata
import re
import pandas as pd
from sympy import Matrix,init_printing
from IPython.display import display

init_printing()

#étape 1
def lire_corpus(fichier):
    try:
        with open(fichier, "r", encoding="utf-8") as f:
            corpus = f.readlines()
    except UnicodeDecodeError:
        with open(fichier, "r", encoding="latin-1") as f:
            corpus = f.readlines()
        
    mots = []
    for ligne in corpus:
        ligne = ligne.lower()
        ligne = unicodedata.normalize('NFD', ligne)
        ligne = ''.join(c for c in ligne if unicodedata.category(c) != 'Mn')
        ligne = re.sub(r'[^a-z\s]', ' ', ligne)
        mots_ligne = ligne.split()
        mots.extend(mots_ligne)
        
    return mots

#etape 2 partie 1
def matrice_emission(nom_fichier):
    df = pd.read_excel(nom_fichier, engine='xlrd')
    B = df.values.tolist()
    B = np.array(B)
    B = B[:, 1:]
    B = np.array(B, dtype=float)
    return B

B = matrice_emission("matrice_emission.xls")
#print(B)

#etape 2 partie 2
def mot_to_indices(mot):
    return [ord(char) - ord('a') for char in mot]

def matrice_transition(nom_fichier):
    mots = lire_corpus(nom_fichier)
    A = np.zeros((26, 26))
    for mot in mots:
        indices = mot_to_indices(mot)
        for i in range(len(indices) - 1):
            A[indices[i], indices[i + 1]] += 1
    A = A / A.sum(axis=1, keepdims=True) #pour normaliser la matrice
    return A

A = matrice_transition("french.txt")

#creation des matrice de transition pour chaque fichier test
fichiers_exemples = ["french.txt", "english.txt", "italian.txt"]
AFR = matrice_transition("french.txt")
AEN = matrice_transition("english.txt")
AIT = matrice_transition("italian.txt")

#creation des lambda
lambdas_FR = [np.ones(26) / 26, AFR, B]
lambdas_EN = [np.ones(26) / 26, AEN, B]
lambdas_IT = [np.ones(26) / 26, AIT, B]


# Etape 3 partie a-1
def forward(O, A, B, PI):
    N = A.shape[0]
    T = len(O)
    alpha = np.zeros((T, N), dtype=float)
    
    O_indices = [ord(char) - ord('a') for char in O]
    
    alpha[0, :] = PI * B[:, O_indices[0]]
    
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = np.sum(alpha[t-1, :] * A[:, j]) * B[j, O_indices[t]]
    
    P = np.sum(alpha[T-1, :])
    return P, alpha

#etape 3 parie a-2
def backward(O, A, B, PI):
    N = A.shape[0]
    T = len(O)
    beta = np.zeros((T, N), dtype=float)
    
    O_indices = [ord(char) - ord('a') for char in O]
    
    beta[T-1, :] = 1
    
    for t in range(T-2, -1, -1):
        for i in range(N):
            beta[t, i] = np.sum(A[i, :] * B[:, O_indices[t+1]] * beta[t+1, :])
    
    P = np.sum(PI * B[:, O_indices[0]] * beta[0, :])
    return P, beta

# Tester la fonction backward
"""pt_backward, betat = backward("probablement", AFR, B, lambdas_FR[0])
print(f"La probabilité d'observation avec backward est {pt_backward}")
"""
#evaluation du model
tab_content = []
#probablement , probably , probabilmente
mots_test = ["probablement", "probably", "probabilmente"]
tab_content = []

for mot in mots_test:
    tab = []
    ptfr, alphatfr = forward(mot, AFR, B, lambdas_FR[0])
    pten, alphaten = forward(mot, AEN, B, lambdas_EN[0])
    ptit, alphatit = forward(mot, AIT, B, lambdas_IT[0])
    
    # Calcul de la somme des probabilités
    total_prob = ptfr + pten + ptit
    
    # Normalisation des probabilités
    norm_ptfr = ptfr / total_prob
    tab.append(norm_ptfr)
    norm_pten = pten / total_prob
    tab.append(norm_pten)
    norm_ptit = ptit / total_prob
    tab.append(norm_ptit)
    
    tab_content.append(tab)

# Création de la matrice de confusion
mat_confusion = np.column_stack((np.array(tab_content[0]), np.array(tab_content[1]), np.array(tab_content[2]))).T
display(Matrix(mat_confusion))

#test de fonction pour determiner dans quelle langue est ecrit chaques fichiers

#fonction de la moyenne des probabilité des mots dans un texte
def moyenne_proba_FR(nom_fichier):
    mots_text = lire_corpus(nom_fichier)
    prob = []
    for mot_ in mots_text:
        pt, alphat = forward(mot_, AFR, B, lambdas_FR[0])
        prob.append(pt)
    if len(prob) == 0:
        return 0,prob  # éviter la division par zéro si la liste est vide
    return sum(prob)/len(prob) ,prob

def moyenne_proba_EN(nom_fichier):
    mots_text = lire_corpus(nom_fichier)
    prob = []
    for mot_ in mots_text:
        pt, alphat = forward(mot_, AEN, B, lambdas_EN[0])
        prob.append(pt)
    if len(prob) == 0:
        return 0,prob  # éviter la division par zéro si la liste est vide
    return sum(prob)/len(prob) , prob

def moyenne_proba_IT(nom_fichier):
    mots_text = lire_corpus(nom_fichier)
    prob = []
    for mot_ in mots_text:
        pt, alphat = forward(mot_, AIT, B, lambdas_IT[0])
        prob.append(pt)
    if len(prob) == 0:
        return 0 ,prob # éviter la division par zéro si la liste est vide
    return sum(prob)/len(prob) , prob

"""1er test"""
"""mots_text = lire_corpus("texte_1.txt")
probfr = moyenne_proba_FR("texte_1.txt")
print(f"\n La probabilité d'observation pour le texte1 en francais est {probfr}")

proben = moyenne_proba_EN("texte_1.txt")
print(f"\n La probabilité d'observation pour le texte1 en anglais est {proben}")

probit = moyenne_proba_IT("texte_1.txt")
print(f"\n La probabilité d'observation pour le texte1 en italien est {probit}")
"""
list_text = ["texte_1.txt", "texte_2.txt", "texte_3.txt"]
i = 0
mat_confusion_fr = []
mat_confusion_en = []
mat_confusion_it = []

for text in list_text:
    probfr, tabfr = moyenne_proba_FR(text)
    proben, taben = moyenne_proba_EN(text)
    probit, tabit = moyenne_proba_IT(text)
    
    # Normalisation des probabilités
    total_prob = probfr + proben + probit
    norm_probfr = probfr / total_prob
    norm_proben = proben / total_prob
    norm_probit = probit / total_prob
    
    # Ajout des probabilités normalisées aux matrices de confusion
    if i == 0:
        mat_confusion_fr = [norm_probfr, norm_proben, norm_probit]
    elif i == 1:
        mat_confusion_en = [norm_probfr, norm_proben, norm_probit]
    elif i == 2:
        mat_confusion_it = [norm_probfr, norm_proben, norm_probit]
    
    # Affichage des résultats
    if norm_probfr > max(norm_proben, norm_probit):
        print(f"le {text} est en francais, avec comme probabilité moyenne {norm_probfr * 100:.2f} %")
    if norm_proben > max(norm_probfr, norm_probit):
        print(f"le {text} est en anglais, avec comme probabilité moyenne {norm_proben * 100:.2f} %")
    if norm_probit > max(norm_probfr, norm_proben):
        print(f"le {text} est en italien, avec comme probabilité moyenne {norm_probit * 100:.2f} %")
    
    i += 1

# Création des matrices de confusion
mat_confusion = np.column_stack((mat_confusion_fr, mat_confusion_en, mat_confusion_it))
print("Matrice de confusion des texte:")
display(Matrix(mat_confusion))