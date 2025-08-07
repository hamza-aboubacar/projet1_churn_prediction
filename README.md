
# Application de Prédiction de Désabonnement Client

Ce projet présente une application web Flask qui prédit le désabonnement (churn) des clients d'une entreprise de télécommunications. L'application utilise un modèle de Machine Learning entraîné et permet d'effectuer des prédictions soit pour un client unique, soit pour un fichier de données complet.

## 🚀 Fonctionnalités

- **Prédiction Individuelle** : Un formulaire simple pour saisir les informations d'un client et obtenir une prédiction immédiate.
- **Prédiction en Masse** : Possibilité de télécharger un fichier `.csv` ou `.xlsx` pour des prédictions sur plusieurs clients.
- **Gestion des Données Manquantes** : Le modèle est robuste et gère automatiquement les valeurs manquantes dans les fichiers importés.
- **Export des Résultats** : Les prédictions en masse peuvent être téléchargées au format CSV ou Excel.

## 🛠️ Technologies et Dépendances

- **Python 3.x**
- **Framework Web** : Flask
- **Bibliothèques de Data Science** : `pandas`, `numpy`, `scikit-learn`
- **Serveur WSGI** : `gunicorn`
- **Déploiement** : Heroku

## 📦 Installation et Démarrage Local

Pour démarrer une version locale du projet, suivez ces étapes :

1.  **Cloner le dépôt :**
    ```bash
    git clone [https://github.com/votre-nom/votre-repo.git](https://github.com/votre-nom/votre-repo.git)
    cd votre-repo
    ```

2.  **Installer les dépendances :**
    Assurez-vous que `pip` est à jour et que vous êtes dans un environnement virtuel.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Lancer l'application :**
    ```bash
    python app.py
    ```
    L'application sera disponible à l'adresse `http://127.0.0.1:5000`.
    

## 🧠 Le Modèle de Machine Learning

Le modèle est un `RandomForestClassifier` (ou un autre modèle que vous avez choisi) entraîné sur le jeu de données `Telco Customer Churn`. Le pré-traitement des données est géré par un `ColumnTransformer` qui normalise les variables numériques et encode les variables catégorielles.

Le modèle sérialisé est sauvegardé dans le fichier `churn_prediction_model.pkl`.


##  Quelques captures d'images du projet
<img width="1628" height="850" alt="Capture d’écran 2025-08-06 152031" src="https://github.com/user-attachments/assets/ca0489d8-fe5b-486f-a8cb-5ca6ad6f98d9" />

<img width="1315" height="612" alt="Capture d’écran 2025-08-06 152102" src="https://github.com/user-attachments/assets/646741da-bace-402e-978c-2655d8c6ba33" />



## ✍️ Auteur

- **Aboubacar Halidou Hamza** - [GitHub](https://github.com/hamza-aboubacar/projet1_churn_prediction) | [LinkedIn]([https://www.linkedin.com/in/votre-linkedin](https://www.linkedin.com/in/hamza-aboubacar-halidou-536b15226/))

