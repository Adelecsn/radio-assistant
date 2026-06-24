# Inférence et extraction d'information

## Objectif

Cette étape transforme les PNG prétraités en sorties JSON exploitables par
l'évaluation et la future interface. Elle ne lit pas les fichiers bruts et ne
modifie pas le manifeste d'ingestion.

La baseline actuelle est volontairement prudente : elle calcule des statistiques
d'image et applique des seuils documentés. Elle valide le workflow logiciel, pas
une performance médicale.

## Entrées

- manifeste CSV produit par `src.ingest` ;
- images PNG prétraitées référencées par `processed_path` ;
- contrat JSON dans `prompts/output_schema.json` ;
- prompt versionné `prompts/baseline_v1.txt`.

## Commande

```bash
python -m src.inference \
  --manifest data/manifests/ingest_manifest.csv \
  --output-dir data/predictions/baseline_v1
```

Options importantes :

- `--confidence-threshold` : seuil sous lequel la classe devient `uncertain` ;
- `--opacity-threshold` : seuil de la baseline pour `suspected_opacity` ;
- `--normal-threshold` : seuil de la baseline pour `normal` ;
- `--poor-quality-std-threshold` : contraste minimal attendu ;
- `--min-foreground-ratio` : proportion minimale de zone utile ;
- `--model-version` et `--prompt-version` : versions écrites dans chaque sortie.

## Sorties

Dans le dossier choisi :

- `cases/<case_id>.json` : enveloppe par cas avec prédiction, features et contexte
  non sensible du manifeste ;
- `predictions.jsonl` : index compact, une ligne JSON par cas ;
- `run_metadata.json` : version du contrat, compteurs et hyperparamètres.

Le champ `prediction` de chaque cas respecte `src/contracts.py`. Si la qualité
est mauvaise ou si la confiance est trop faible, la classe doit rester
`uncertain`.

## Features calculées

La baseline extrait notamment :

- moyenne et écart-type des intensités ;
- proportion de zone utile non noire ;
- proportion de pixels très clairs ;
- signal central de pixels clairs ;
- asymétrie gauche/droite ;
- densité approximative de contours.

Ces features sont des indicateurs techniques. Elles ne remplacent pas une
segmentation anatomique ni une expertise médicale.

## Critère de fin de l'étape 2

- chaque image prétraitée produit un JSON de prédiction valide ;
- les hyperparamètres du run sont stockés ;
- les sorties invalides sont rejetées ou basculent vers `uncertain` ;
- les tests couvrent la baseline, le batch et la CLI ;
- le prochain modèle pourra remplacer la baseline sans changer le contrat.
