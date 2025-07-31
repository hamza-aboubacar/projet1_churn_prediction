# -*- coding: utf-8 -*-
"""
Application Web Flask pour la Prédiction de Désabonnement (Churn)

Cette application Flask permet de :
1. Prédire le désabonnement d'un client individuel via un formulaire.
2. Importer un fichier CSV ou Excel, effectuer des prédictions en masse et afficher/télécharger les résultats.
"""

from flask import Flask, request, render_template, redirect, url_for, send_file, session
import joblib
import pandas as pd
import numpy as np
import os
import io # Pour gérer les flux de fichiers en mémoire
import logging # Pour la journalisation

# Configuration de la journalisation
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
# Utiliser une variable d'environnement pour la clé secrète (essentiel pour la production)
# En développement, une clé par défaut est utilisée.
app.secret_key = os.environ.get('SECRET_KEY', 'votre_super_cle_secrete_par_defaut_ici')

# --- Chargement du modèle ---
# Assurez-vous que 'churn_prediction_model.pkl' est dans le même répertoire que app.py
try:
    model_path = os.path.join(os.path.dirname(__file__), 'churn_prediction_model.pkl')
    model = joblib.load(model_path)
    logging.info("Modèle de prédiction chargé avec succès.")

    # Obtenir les noms des colonnes d'entrée attendues directement depuis le préprocesseur entraîné
    # C'est la source la plus fiable pour s'assurer de l'alignement des colonnes.
    expected_input_columns = model.named_steps['preprocessor'].feature_names_in_
    logging.info(f"Colonnes d'entrée attendues par le modèle: {expected_input_columns.tolist()}")

    # Pour faciliter l'accès aux colonnes numériques et catégorielles originales
    numeric_features_original = []
    categorical_features_original = []
    for transformer_name, transformer, features_list in model.named_steps['preprocessor'].transformers_:
        if transformer_name == 'num':
            numeric_features_original.extend(features_list)
        elif transformer_name == 'cat':
            categorical_features_original.extend(features_list)
    logging.info(f"Colonnes numériques originales: {numeric_features_original}")
    logging.info(f"Colonnes catégorielles originales: {categorical_features_original}")

except FileNotFoundError as e:
    logging.error(f"Erreur : Le fichier du modèle 'churn_prediction_model.pkl' est introuvable. {e}")
    logging.error("Veuillez vous assurer que 'churn_prediction_model.pkl' est dans le même dossier que 'app.py'.")
    exit() # Quitte l'application si le fichier n'est pas trouvé
except Exception as e:
    logging.error(f"Erreur lors du chargement du modèle : {e}", exc_info=True)
    exit()

# --- Définition des routes de l'application ---

@app.route('/')
def home():
    """
    Affiche la page d'accueil avec le formulaire de saisie des données client individuelles.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Traite les données du formulaire individuel, effectue la prédiction de désabonnement
    et affiche le résultat.
    """
    if request.method == 'POST':
        try:
            # Récupérer les données du formulaire
            data = {
                'gender': request.form['gender'],
                'SeniorCitizen': int(request.form['SeniorCitizen']),
                'Partner': request.form['Partner'],
                'Dependents': request.form['Dependents'],
                'tenure': int(request.form['tenure']),
                'PhoneService': request.form['PhoneService'],
                'MultipleLines': request.form['MultipleLines'],
                'InternetService': request.form['InternetService'],
                'OnlineSecurity': request.form['OnlineSecurity'],
                'OnlineBackup': request.form['OnlineBackup'],
                'DeviceProtection': request.form['DeviceProtection'],
                'TechSupport': request.form['TechSupport'],
                'StreamingTV': request.form['StreamingTV'],
                'StreamingMovies': request.form['StreamingMovies'],
                'Contract': request.form['Contract'],
                'PaperlessBilling': request.form['PaperlessBilling'],
                'PaymentMethod': request.form['PaymentMethod'],
                'MonthlyCharges': float(request.form['MonthlyCharges']),
                'TotalCharges': float(request.form['TotalCharges'])
            }

            # Créer un DataFrame à partir des données d'entrée
            input_df = pd.DataFrame([data])
            # Réordonner les colonnes pour qu'elles correspondent à l'entraînement
            # Utiliser expected_input_columns pour un alignement parfait
            input_df = input_df[expected_input_columns] 

            # Effectuer la prédiction
            prediction_proba = model.predict_proba(input_df)[:, 1] # Probabilité de churn
            prediction = (prediction_proba >= 0.5).astype(int) # 0 ou 1

            churn_result = "Oui" if prediction[0] == 1 else "Non"
            probability = f"{prediction_proba[0]*100:.2f}%"
            logging.info(f"Prédiction individuelle effectuée: {churn_result} avec probabilité {probability}")

            return render_template('result.html', churn_result=churn_result, probability=probability)

        except ValueError as ve:
            logging.error(f"Erreur de saisie pour prédiction individuelle: {ve}")
            return render_template('result.html', churn_result="Erreur de saisie", probability=f"Veuillez vérifier les valeurs numériques. Erreur: {ve}")
        except Exception as e:
            logging.error(f"Erreur inattendue lors de la prédiction individuelle: {e}", exc_info=True)
            return render_template('result.html', churn_result="Erreur de prédiction", probability=f"Une erreur inattendue est survenue. Erreur: {e}")

@app.route('/upload')
def upload_file_form():
    """
    Affiche le formulaire pour télécharger un fichier dataset.
    """
    return render_template('upload.html')

@app.route('/upload_predict', methods=['POST'])
def upload_predict():
    """
    Gère le téléchargement du fichier, effectue des prédictions en masse
    et affiche les résultats.
    """
    if 'file' not in request.files:
        logging.warning("Aucun fichier sélectionné pour l'upload.")
        return render_template('upload.html', message='Aucun fichier sélectionné.')
    
    file = request.files['file']

    if file.filename == '':
        logging.warning("Nom de fichier vide pour l'upload.")
        return render_template('upload.html', message='Aucun fichier sélectionné.')
    
    # Vérifier l'extension du fichier
    file_extension = file.filename.rsplit('.', 1)[1].lower()

    if file_extension in ['csv', 'xls', 'xlsx']:
        try:
            if file_extension == 'csv':
                df_uploaded = pd.read_csv(io.BytesIO(file.read()))
            elif file_extension in ['xls', 'xlsx']:
                df_uploaded = pd.read_excel(io.BytesIO(file.read()), engine='openpyxl')
            
            logging.info(f"Fichier {file.filename} de type {file_extension} lu avec succès. Lignes: {len(df_uploaded)}")

            # Supprimer la colonne 'customerID' si elle existe, car elle n'est pas utilisée pour la prédiction
            if 'customerID' in df_uploaded.columns:
                df_uploaded = df_uploaded.drop('customerID', axis=1)
                logging.info("Colonne 'customerID' supprimée.")

            # --- Logique pour aligner les colonnes et gérer les NaN/valeurs inconnues ---
            df_processed = df_uploaded.copy()

            # 1. Ajouter les colonnes manquantes avec NaN et supprimer les colonnes supplémentaires
            # en alignant directement sur les colonnes attendues par le préprocesseur.
            for col in expected_input_columns:
                if col not in df_processed.columns:
                    df_processed[col] = np.nan
                    logging.warning(f"Colonne '{col}' manquante dans le fichier uploadé. Remplie avec NaN.")
            
            # Assurer que l'ordre des colonnes est exact
            df_processed = df_processed[expected_input_columns]
            logging.info("Colonnes alignées sur le schéma d'entraînement.")

            # 2. Assurer les bons types de données et gérer les NaN
            for col in expected_input_columns: # Itérer sur les colonnes dans l'ordre attendu
                if col in numeric_features_original:
                    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                    if df_processed[col].isnull().any():
                        # Remplir les NaN. Idéalement, utiliser la médiane/mode du jeu d'entraînement.
                        # Ici, nous utilisons la médiane du lot actuel ou 0 si tout est NaN.
                        median_val = df_processed[col].median() if not df_processed[col].isnull().all() else 0
                        df_processed[col].fillna(median_val, inplace=True)
                        logging.warning(f"NaNs dans la colonne numérique '{col}' remplis avec {median_val}.")
                elif col in categorical_features_original:
                    df_processed[col] = df_processed[col].astype(str).replace('nan', 'Unknown') # Remplace les NaN par 'Unknown'
                    # Vérifier si des catégories inconnues sont présentes
                    try:
                        cat_col_index = categorical_features_original.index(col)
                        known_categories = model.named_steps['preprocessor'].named_transformers_['cat'].categories_[cat_col_index]
                        unknown_values = df_processed[col][~df_processed[col].isin(known_categories)].unique()
                        if len(unknown_values) > 0:
                            logging.warning(f"Catégories inconnues détectées dans '{col}': {unknown_values}. Elles seront traitées comme 'ignore' par OneHotEncoder.")
                    except ValueError:
                        # Cela peut arriver si 'col' n'est pas dans categorical_features_original (par exemple, si c'est une colonne numérique)
                        # ou si l'indexation des catégories pose problème.
                        logging.debug(f"Impossible de vérifier les catégories connues pour la colonne '{col}', elle n'est peut-être pas catégorielle ou il y a un problème d'index.")
            
            logging.info("Types de données et NaN gérés.")

            # Effectuer les prédictions
            predictions_proba = model.predict_proba(df_processed)[:, 1]
            predictions = (predictions_proba >= 0.5).astype(int)
            logging.info(f"Prédictions effectuées pour {len(df_processed)} lignes.")

            # Ajouter les prédictions au DataFrame original uploadé
            df_uploaded['Predicted Churn'] = np.where(predictions == 1, 'Yes', 'No')
            df_uploaded['Churn Probability'] = [f"{p*100:.2f}%" for p in predictions_proba]
            logging.info("Colonnes de prédiction ajoutées au DataFrame.")

            # Stocker le DataFrame résultant dans la session pour le téléchargement
            session['predicted_df'] = df_uploaded.to_json(orient='split')

            # Afficher les 10 premières lignes des résultats
            display_df = df_uploaded.head(10).to_html(classes='table-auto w-full text-left whitespace-no-wrap', index=False)
            
            return render_template('batch_results.html', table=display_df, total_rows=len(df_uploaded))

        except Exception as e:
            logging.error(f"Erreur lors du traitement du fichier {file.filename}: {e}", exc_info=True)
            return render_template('upload.html', message=f"Erreur lors du traitement du fichier : {e}. Veuillez vérifier le format et les données.")
    elif file_extension == 'pdf':
        logging.warning(f"Tentative d'upload d'un fichier PDF: {file.filename}. Format non supporté pour l'extraction de données structurées.")
        return render_template('upload.html', message='Le format PDF n\'est pas directement supporté pour l\'extraction de données de prédiction. Veuillez utiliser un fichier CSV ou Excel.')
    else:
        logging.warning(f"Format de fichier non supporté: {file.filename} (extension: {file_extension}).")
        return render_template('upload.html', message='Format de fichier non supporté. Veuillez télécharger un fichier CSV ou Excel (.xls, .xlsx).')

@app.route('/download_results')
def download_results():
    """
    Permet de télécharger le DataFrame avec les prédictions au format CSV ou Excel.
    """
    if 'predicted_df' in session:
        df_to_download = pd.read_json(session['predicted_df'], orient='split')
        
        output_format = request.args.get('format', 'csv') # Par défaut CSV

        if output_format == 'csv':
            csv_buffer = io.StringIO()
            df_to_download.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            logging.info("Résultats téléchargés au format CSV.")
            return send_file(io.BytesIO(csv_buffer.getvalue().encode('utf-8')),
                             mimetype='text/csv',
                             as_attachment=True,
                             download_name='churn_predictions.csv')
        elif output_format == 'xlsx':
            excel_buffer = io.BytesIO()
            df_to_download.to_excel(excel_buffer, index=False, engine='openpyxl')
            excel_buffer.seek(0)
            logging.info("Résultats téléchargés au format Excel.")
            return send_file(excel_buffer,
                             mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                             as_attachment=True,
                             download_name='churn_predictions.xlsx')
        else:
            logging.warning(f"Demande de téléchargement avec format non valide: {output_format}. Redirection vers upload.")
            return redirect(url_for('upload_file_form')) # Rediriger si le format n'est pas valide
    else:
        logging.warning("Aucun DataFrame de prédiction trouvé en session pour le téléchargement. Redirection vers upload.")
        return redirect(url_for('upload_file_form'))


# --- Exécution de l'application Flask ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
