# Webapp et dataviz

## Objectif

Cette étape rend visibles les résultats de l'inférence sans modifier les images,
les prédictions ou le contrat JSON. Elle sert à démontrer le pipeline et à
préparer l'analyse d'erreurs.

La webapp actuelle est une interface locale FastAPI. Elle lit les fichiers JSON
produits par `src.inference`, affiche un dashboard HTML et journalise les vues
de cas dans SQLite.

## Commande

Après ingestion puis inférence :

```bash
make webapp-setup
python -m src.webapp \
  --predictions-dir data/predictions/baseline_v1 \
  --db-path logs/webapp.sqlite
```

Ouvrir ensuite :

```text
http://127.0.0.1:8000
```

Raccourci équivalent :

```bash
make webapp-run
```

Options :

- `--predictions-dir` : dossier contenant `predictions.jsonl`, `cases/` et
  `run_metadata.json` ;
- `--db-path` : base SQLite locale de consultation ;
- `--host` et `--port` : adresse de lancement local.

## Pages et routes

- `/` : dashboard HTML ;
- `/upload` : formulaire de dépôt d'image (PNG, JPEG ou DICOM) ;
- `POST /predict` : analyse live de l'image uploadée (prétraitement + inférence)
  et journalisation SQLite ;
- `/uploads/{case_id}/image` : image PNG prétraitée d'un upload ;
- `/about` : périmètre, source RSNA, limites et prochaines étapes ;
- `/cases/{case_id}` : détail d'un cas avec warning, prédiction, features et
  JSON complet ;
- `/cases/{case_id}/image` : image PNG prétraitée si elle est disponible ;
- `/errors` : revue visuelle des erreurs avec les radios ;
- `/report` : rapport HTML d'évaluation ;
- `/api/summary` : résumé JSON ;
- `/api/evaluation` : rapport d'évaluation JSON ;
- `/api/evaluation/errors.csv` : registre d'erreurs CSV ;
- `/api/cases` : liste JSON des cas.

## Upload et inférence live

Le dépôt d'image (`/upload` → `POST /predict`) réutilise le prétraitement
d'ingestion puis l'inférence. La variante utilisée est choisie par
`--variant` (par défaut `improved`). L'identifiant du cas est un condensé du
contenu de l'image, jamais un nom de fichier ; les images uploadées prétraitées
restent dans `logs/uploads/` (hors Git). Chaque analyse crée une ligne dans la
table SQLite `inference_events` (classe, qualité, confiance, versions, latence).

## Données affichées

Le dashboard affiche :

- nombre de cas analysés ;
- confiance moyenne ;
- latence moyenne ;
- répartition par classe prédite ;
- répartition par qualité image ;
- accuracy, macro-F1, rappel/sensibilité et matrice de confusion si les labels
  sont disponibles ;
- erreurs à analyser dans le registre ;
- nombre de consultations loggées ;
- métadonnées du run et hyperparamètres.

Les logs SQLite ne contiennent pas d'identité patient. Ils conservent uniquement
des preuves techniques locales : `case_id`, classe prédite, qualité image,
versions modèle/prompt et date de consultation.

## Limites actuelles

- le dashboard (hors page upload) dépend des sorties générées au préalable par
  `src.inference` ;
- la comparaison baseline vs amélioration est produite par `eval.compare` et le
  notebook `03_baseline_vs_improved.ipynb`, pas encore intégrée comme page web ;
- inférence statistique non clinique : voir limites du rapport.

## Critère de fin de l'étape 3

- le dashboard lit les sorties JSON existantes ;
- le warning non clinique est visible ;
- l'upload d'image produit une analyse live conforme au contrat ;
- les détails par cas sont consultables ;
- chaque vue de cas et chaque upload créent une ligne SQLite ;
- les tests couvrent lecture, rendu, upload live et journalisation.
