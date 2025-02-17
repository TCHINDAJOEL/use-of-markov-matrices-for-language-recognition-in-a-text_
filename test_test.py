import numpy as np
import unicodedata
import re
import pandas as pd

#étape 1
#fonction pour lire un fichier et retourner une liste de mots
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
#fonction pour lire un fichier excel et retourner une matrice
def matrice_emission(nom_fichier):
    df = pd.read_excel(nom_fichier, engine='xlrd')
    B = df.values.tolist()
    B = np.array(B)
    #on supprime la 1ere colonne car elle contient des lettres
    B = B[:, 1:]
    #on change le type de toute la matrice pour s'assurer qu'on a que des float
    B = np.array(B, dtype=float)
    return B

B = matrice_emission("matrice_emission.xls")

#etape 2 partie 2
#fonction pour transformer un mot en une liste d'indices
def mot_to_indices(mot):
    return [ord(char) - ord('a') for char in mot]

#fonction pour creer une matrice de transition
def matrice_transition(nom_fichier):
    mots = lire_corpus(nom_fichier)
    A = np.zeros((26, 26))
    for mot in mots:
        indices = mot_to_indices(mot)
        for i in range(len(indices) - 1):
            A[indices[i], indices[i + 1]] += 1
    A = A / A.sum(axis=1, keepdims=True) #pour normaliser la matrice
    return A

#creation des matrice de transition pour chaque fichier test
fichiers_exemples = ["french.txt", "english.txt", "italian.txt"]
AFR = matrice_transition("french.txt")
AEN = matrice_transition("english.txt")
AIT = matrice_transition("italian.txt")

#creation des lambda
lambdas_FR = [np.ones(26) / 26, AFR, B]
lambdas_EN = [np.ones(26) / 26, AEN, B]
lambdas_IT = [np.ones(26) / 26, AIT, B]

#etape 3
#fonction pour calculer la probabilité d'une séquence d'observation

def forward(O, A, B, PI):
    # Taille de la matrice de transition (nombre d'états) = 26
    N = A.shape[0]
    # Longueur de la séquence d'observation
    T = len(O)
    
    # Créer une matrice alpha initialisée avec des zéros
    alpha = np.zeros((T, N), dtype=float)
    
    # Transformer les caractères de la séquence d'observation en indices
    O_indices = [ord(char) - ord('a') for char in O]
    
    # Initialiser la première ligne de alpha avec les probabilités initiales et les probabilités d'émission
    for i in range(N):
        alpha[0, i] = PI[i] * B[i, O_indices[0]]
    
    # Remplir alpha en partant du début vers la fin
    for t in range(1, T):  # Pour chaque temps à partir de 1 jusqu'à T-1
        for j in range(N):  # Pour chaque état actuel
            # Calculer alpha pour l'état j au temps t
            somme = 0
            for i in range(N):  # Pour chaque état précédent
                somme += alpha[t - 1, i] * A[i, j]
            alpha[t, j] = somme * B[j, O_indices[t]]
    
    # Calculer la probabilité totale de la séquence d'observation
    P = 0
    for j in range(N):
        P += alpha[T - 1, j]
    
    return P, alpha

#fonction pour calculer la probabilité d'une séquence d'observation

def backward(O, A, B, PI):
    # Taille de la matrice de transition (nombre d'états)
    N = A.shape[0]
    # Longueur de la séquence d'observation
    T = len(O)
    
    # Créer une matrice beta initialisée avec des zéros
    beta = np.zeros((T, N), dtype=float)
    
    # Transformer les caractères de la séquence d'observation en indices
    O_indices = [ord(char) - ord('a') for char in O]
    
    # Initialiser la dernière ligne de beta avec des 1
    beta[T-1, :] = 1
    
    # Remplir beta en partant de la fin vers le début
    for t in range(T - 2, -1, -1):  # On va de T-2 jusqu'à 0
        for i in range(N):  # Pour chaque état
            # Calculer beta pour l'état i au temps t
            somme = 0
            for j in range(N):  # Pour chaque état suivant
                somme += A[i, j] * B[j, O_indices[t + 1]] * beta[t + 1, j]
            beta[t, i] = somme
    
    # Calculer la probabilité totale de la séquence d'observation
    P = 0
    for i in range(N):
        P += PI[i] * B[i, O_indices[0]] * beta[0, i]
    
    return P, beta

# Test de la fonction backward

tab_content = [] #pour stocker les tableaux de resultats
# Test de forward et backward sur les mots "probablement", "probably" et "probabilmente"

mots_test = ["probablement", "probably", "probablimente"]
langues = ["francais","english","italia"]

for mot in mots_test:
    tab = [] #pour stocker les valeurs de probabilités pour chaque mot
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
df_mat_confusion = pd.DataFrame(mat_confusion,index=mots_test,columns=langues)
print("Matrice de confusion des mots\n")
print(df_mat_confusion)

#etape 4
#fonction pour calculer la matrice de confusion pour chaque mot d'un texte

def matrice_confusion_mots_par_texte(nom_fichier, AFR, AEN, AIT, B, lambdas_FR, lambdas_EN, lambdas_IT):
    mots_text = lire_corpus(nom_fichier)
    mat_confusion_mots = []
    for mot in mots_text:
        pt_fr,alphatfr = forward(mot, AFR, B, lambdas_FR[0])
        pt_en,alphaten = forward(mot, AEN, B, lambdas_EN[0])
        pt_it,alphatit = forward(mot, AIT, B, lambdas_IT[0])
        total_prob = pt_fr + pt_en + pt_it
        mat_confusion_mots.append([pt_fr / total_prob, pt_en / total_prob, pt_it / total_prob])
    
    #affichons un instant les 10 premiers mot de chaque modèle pour voir si
    #print(f"\nLes 10 premiers mots du texte '{nom_fichier}' sont : et leurs probabilité dans chaque langue est :\n")
    mat_confusion_mots = np.array(mat_confusion_mots)
    #print(pd.DataFrame(mat_confusion_mots[:10], columns=langues, index=mots_text[:10]))
    return mat_confusion_mots

# Calcul des matrices de confusion pour chaque texte
mat_confusion_texte_1 = matrice_confusion_mots_par_texte("texte_1.txt", AFR, AEN, AIT, B, [np.ones(26) / 26, AFR, B], [np.ones(26) / 26, AEN, B], [np.ones(26) / 26, AIT, B])
mat_confusion_texte_2 = matrice_confusion_mots_par_texte("texte_2.txt", AFR, AEN, AIT, B, [np.ones(26) / 26, AFR, B], [np.ones(26) / 26, AEN, B], [np.ones(26) / 26, AIT, B])
mat_confusion_texte_3 = matrice_confusion_mots_par_texte("texte_3.txt", AFR, AEN, AIT, B, [np.ones(26) / 26, AFR, B], [np.ones(26) / 26, AEN, B], [np.ones(26) / 26, AIT, B])

# Calcul de la moyenne des probabilités et leur normalisation
def moyenne_et_normalisation(mat_confusion):
    #calcul de la moyenne des proba de chaque colonne 
    moyennes = np.mean(mat_confusion, axis=0)
    #normalisation des moyennes
    return moyennes / np.sum(moyennes)

# Calcul des moyennes et normalisation pour chaque texte

print("\n Voici la langue la plus probable pour chaque texte\n")
mat_conf = []
# Fonction pour déterminer la langue dominante d'un texte et afficher le résultat
def afficher_langue_dominante(nom_fichier, mat_confusion):
    norm_probfr, norm_proben, norm_probit = moyenne_et_normalisation(mat_confusion)
    mat_conf.append(norm_probfr)
    mat_conf.append(norm_proben)
    mat_conf.append(norm_probit)
    if norm_probfr > max(norm_proben, norm_probit):
        print(f"\n Le texte '{nom_fichier}' est en français, avec une probabilité moyenne de {norm_probfr * 100:.2f} % \n")
    elif norm_proben > max(norm_probfr, norm_probit):
        print(f"\n Le texte '{nom_fichier}' est en anglais, avec une probabilité moyenne de {norm_proben * 100:.2f} % \n")
    elif norm_probit > max(norm_probfr, norm_proben):
        print(f"\n Le texte '{nom_fichier}' est en italien, avec une probabilité moyenne de {norm_probit * 100:.2f} % \n")

# Utilisation de la fonction pour chaque texte
afficher_langue_dominante("texte_1.txt", mat_confusion_texte_1)
afficher_langue_dominante("texte_2.txt", mat_confusion_texte_2)
afficher_langue_dominante("texte_3.txt", mat_confusion_texte_3)


#test de fonction pour determiner dans quelle langue est ecrit chaques fichiers

mat_conf = np.row_stack(mat_conf)
mat_conf = mat_conf.reshape(3, 3)
texte = ['texte1','texte2','texte3']
mat_conf = pd.DataFrame(mat_conf, index=texte, columns=langues)
print("\nMatrice de confusion des textes\n")
print(mat_conf)