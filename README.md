Active Learning pour la Détection de Spam en Adaptation de Domaine (Email → SMS)

Ce dépôt contient le code et les expérimentations associées à une étude d’active learning dans un contexte d’adaptation de domaine, appliquée à la détection de spam. L’objectif est d’analyser et de comparer différentes stratégies d’active learning lors du transfert d’un modèle entraîné sur des e-mails (domaine source) vers des messages SMS (domaine cible).

📌 Contexte du problème

Tâche : classification binaire (spam / ham)

Domaine source (A) : e-mails

Domaine cible (B) : SMS

Défi principal : fort domain shift entre les e-mails et les SMS

Le modèle est initialement entraîné sur le domaine source, puis progressivement adapté au domaine cible à l’aide de différentes stratégies d’active learning.

📊 Jeux de données

Deux jeux de données publics sont utilisés dans ce projet :

🔹 Domaine A — Dataset Emails

Nom : iamthearafatkhan/email-spam-dataset

Description : ensemble de messages électroniques annotés selon deux classes : spam et ham (non-spam).

Source : Hugging Face Datasets

Lien : https://huggingface.co/datasets/iamthearafatkhan/email-spam-dataset

🔹 Domaine B — Dataset SMS

Nom : ucirvine/sms_spam

Description : ensemble de messages SMS annotés pour la tâche de classification spam / non-spam.

Source : Hugging Face Datasets (UCI Machine Learning Repository)

Lien : https://huggingface.co/datasets/ucirvine/sms_spam

🧠 Modèle utilisé

Représentation des textes : TF-IDF (jusqu’à 10 000 caractéristiques, uni-grammes et bi-grammes)

Classifieur : Multi-Layer Perceptron (MLP)

Couche cachée : 128 neurones avec activation ReLU

Couche de sortie : 2 neurones (spam / ham)

Le modèle est ré-entraîné depuis zéro à chaque itération d’active learning.

🔁 Protocole d’Active Learning

6 itérations d’active learning (5 % → 100 % des SMS annotés)

3 répétitions avec des graines aléatoires différentes

Évaluation réalisée uniquement sur un jeu de test SMS fixe

Métrique principale : F1-score sur la classe spam

🔍 Stratégies comparées

Sélection aléatoire (baseline)

Méthodes basées sur l’incertitude:

Entropy sampling

Méthodes basées sur la diversité:

Outlier sampling

K-Means sampling

Stratégies hybrides:

Entropy → K-Means

K-Means → Entropy

📈 Résultats et analyse

Les résultats sont rapportés sous forme :

de moyennes sur 3 répétitions,

d’intervalles de confiance à 95 %,

de courbes comparatives illustrant l’évolution du F1-score selon les itérations.

Les logs et figures sont générés automatiquement afin d’assurer la reproductibilité des expériences.

⚙️ Dépendances

Principales bibliothèques utilisées :

Python ≥ 3.9

PyTorch

scikit-learn

datasets (Hugging Face)

numpy, pandas, matplotlib

📎 Remarques

Le projet met l’accent sur la comparaison méthodologique des stratégies d’active learning, et non sur l’optimisation maximale des performances.

La sélection aléatoire constitue une baseline particulièrement compétitive dans ce contexte.

Des travaux futurs pourraient explorer des stratégies hybrides combinant sélection aléatoire et sélection informée.
