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
- `/cases/{case_id}` : détail d'un cas avec warning, prédiction, features et
  JSON complet ;
- `/cases/{case_id}/image` : image PNG prétraitée si elle est disponible ;
- `/api/summary` : résumé JSON ;
- `/api/cases` : liste JSON des cas.

## Données affichées

Le dashboard affiche :

- nombre de cas analysés ;
- confiance moyenne ;
- latence moyenne ;
- répartition par classe prédite ;
- répartition par qualité image ;
- nombre de consultations loggées ;
- métadonnées du run et hyperparamètres.

Les logs SQLite ne contiennent pas d'identité patient. Ils conservent uniquement
des preuves techniques locales : `case_id`, classe prédite, qualité image,
versions modèle/prompt et date de consultation.

## Limites actuelles

- pas encore d'upload direct depuis l'interface ;
- pas encore de comparaison avancée baseline vs modèle amélioré ;
- pas encore de dashboard d'évaluation complet avec matrice de confusion ;
- dépend des sorties générées au préalable par `src.inference`.

## Critère de fin de l'étape 3

- le dashboard lit les sorties JSON existantes ;
- le warning non clinique est visible ;
- les détails par cas sont consultables ;
- chaque vue de cas crée une ligne SQLite ;
- les tests couvrent lecture, rendu et journalisation.
