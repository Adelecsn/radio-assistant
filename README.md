# Radio Assistant

Prototype pédagogique d'IA multimodale pour analyser prudemment une
radiographie thoracique frontale. Le projet compare une baseline à une version
améliorée, mesure leurs erreurs et conserve les preuves de chaque expérimentation.

> **Usage non clinique.** Ce projet n'est pas un dispositif médical et ne doit
> pas être utilisé pour diagnostiquer, trier ou orienter un patient. Toute sortie
> doit être vérifiée par un professionnel qualifié.

## Objectif

Le pipeline doit accepter une image et retourner une sortie JSON stable avec :

- la qualité de l'image (`good` ou `poor`) ;
- une classe (`normal`, `suspected_opacity` ou `uncertain`) ;
- un score de confiance et des observations strictement visuelles ;
- les versions du modèle et du prompt, ainsi que la latence ;
- un avertissement non clinique obligatoire.

Le contrat exact est défini dans [`prompts/output_schema.json`](prompts/output_schema.json)
et appliqué par [`src/contracts.py`](src/contracts.py).

## Étape 1 - Ingestion

Le premier pipeline accepte des images PNG, JPEG ou DICOM, supprime les
métadonnées lors de l'export, normalise les intensités et crée des PNG carrés
sans déformer l'image. Il génère en parallèle un manifeste dé-identifié.

```bash
python -m src.ingest \
  --input-dir data/raw \
  --output-dir data/processed \
  --manifest data/manifests/ingest_manifest.csv \
  --source-config data/source.local.json \
  --labels-csv data/raw/labels.csv
```

La préparation de la source et les contrôles humains obligatoires sont décrits
dans [`docs/data_ingestion.md`](docs/data_ingestion.md).

## Étape 2 - Extraction et JSON

Le second pipeline lit le manifeste d'ingestion, charge les PNG prétraités,
calcule des mesures d'image reproductibles, applique une baseline prudente et
écrit des sorties JSON validées par le contrat partagé.

```bash
python -m src.inference \
  --manifest data/manifests/ingest_manifest.csv \
  --output-dir data/predictions/baseline_v1
```

Les prédictions par cas sont écrites dans `cases/`, l'index dans
`predictions.jsonl` et les hyperparamètres dans `run_metadata.json`. Cette
baseline sert à tester le workflow IA; elle ne constitue pas un modèle médical.
La procédure est détaillée dans [`docs/inference.md`](docs/inference.md).

Une variante améliorée (`--variant improved`) ajoute un signal auxiliaire de
texture locale et une règle d'incertitude pour supprimer les faux négatifs
dangereux (opacité lue comme normale) :

```bash
python -m src.inference --variant improved \
  --manifest data/manifests/ingest_manifest.csv \
  --output-dir data/predictions/improved_v1
```

Un backend multimodal **MedGemma** (`--variant medgemma`) est branché derrière le
même contrat JSON, avec bascule automatique vers `uncertain` si la sortie est
invalide. Il nécessite un compte Hugging Face et l'acceptation de la licence du
modèle : voir le guide pas à pas [`docs/medgemma.md`](docs/medgemma.md).

## Étape 3 - Webapp et dataviz

La troisième brique lit les sorties JSON de l'inférence, affiche un dashboard
local et journalise les consultations dans SQLite sans donnée patient.

```bash
make webapp-setup
python -m src.webapp \
  --predictions-dir data/predictions/baseline_v1 \
  --db-path logs/webapp.sqlite
```

Interface par défaut : <http://127.0.0.1:8000>. Le dashboard montre le warning,
les compteurs par classe, la qualité image, les cas, les features et le JSON
complet par cas. La page `/upload` permet de **déposer une image** (PNG, JPEG ou
DICOM) et d'obtenir une analyse live conforme au contrat, journalisée en SQLite.
Les détails sont dans [`docs/webapp.md`](docs/webapp.md).

## Étape 4 - Évaluation

L'évaluation lit les prédictions labellisées, calcule les métriques principales
et génère un registre d'erreurs à relire humainement.

```bash
python -m eval.evaluate \
  --predictions-dir data/predictions/baseline_v1 \
  --output-dir eval/outputs/baseline_v1
```

Sorties générées :

- `metrics_report.json` : accuracy, macro-F1, rappel/sensibilité et matrice de
  confusion ;
- `error_register.csv` : cas où la prédiction ne correspond pas au label.

Ces métriques servent à analyser la baseline. Elles ne constituent pas une
validation médicale.

La baseline actuelle est `image-stat-baseline-v0.2`. Sur l'échantillon local RSNA
de 30 cas, elle obtient environ 70 % d'accuracy et 70 % de macro-F1.

### Comparaison baseline vs amélioration

La variante `image-stat-improved-v0.3` est comparée à la baseline sur les **mêmes**
30 cas :

```bash
python -m eval.compare \
  --baseline-dir data/predictions/baseline_v1 \
  --improved-dir data/predictions/improved_v1 \
  --output-dir eval/outputs/comparison
```

Gain mesuré : accuracy 0.70 → **0.77**, macro-F1 0.70 → **0.76**, sensibilité
`suspected_opacity` 0.70 → **0.90**, et faux négatifs dangereux (opacité→normal)
**2 → 0**, sans dégrader le rappel `normal`. Détails, hypothèse et décision dans
[`docs/rapport.md`](docs/rapport.md). Notebooks reproductibles :
[`notebooks/`](notebooks/) (`02_baseline.ipynb`, `03_baseline_vs_improved.ipynb`).

Ces chiffres sont indicatifs, calculés sur un petit échantillon de démonstration,
et ne constituent pas une validation médicale.

## Organisation

```text
radio-assistant/
|-- src/
|   |-- ingest/       # sélection, contrôle et prétraitement des images
|   |-- inference/    # modèle, prompts, parsing et règles d'incertitude
|   |-- webapp/       # interface, API et journalisation des runs
|   `-- contracts.py  # contrat partagé entre les trois pôles
|-- data/             # métadonnées uniquement; images sensibles exclues de Git
|-- prompts/          # prompts versionnés et schéma JSON
|-- eval/             # métriques, comparaisons et registre d'erreurs
|-- notebooks/        # exploration et baseline reproductible
|-- logs/             # documentation du stockage local SQLite
|-- tests/            # contrats et tests d'intégration
|-- docs/             # architecture, éthique, workflow et décisions
`-- .github/          # intégration continue et règles de pull request
```

La structure conserve les trois pôles définis dans le planning de l'équipe. Elle
s'inspire du dépôt pédagogique fourni pour la traçabilité et l'évaluation, sans
reprendre son organisation monolithique ni son prédicteur jouet fondé sur les noms
de fichiers. Voir [`docs/reference_analysis.md`](docs/reference_analysis.md).

## Installation et commandes

### 1. Préparer l'environnement

Créer l'environnement Python local du projet :

```bash
python3 -m venv .venv
```

Activer l'environnement :

```bash
source .venv/bin/activate
```

Mettre `pip` à jour :

```bash
python -m pip install --upgrade pip
```

Installer toutes les dépendances du projet, y compris les dépendances IA et web :

```bash
pip install -r requirements.txt
```

Pour travailler uniquement sur les tests sans installer les dépendances IA
lourdes, utiliser plutôt :

```bash
make setup
```

`make setup` crée `.venv` avec `python3` si nécessaire et installe seulement les
dépendances de test.

### 2. Vérifier que le projet fonctionne

Lancer les tests automatiques :

```bash
make test
```

Lancer les tests puis vérifier que les fichiers Python compilent correctement :

```bash
make check
```

Ces commandes valident le comportement logiciel du pipeline. Elles ne valident
pas une performance médicale.

### 3. Préparer les données d'entrée

Copier l'exemple de configuration de source vers un fichier local :

```bash
cp data/source.example.json data/source.local.json
```

Modifier ensuite `data/source.local.json` avec la vraie provenance du dataset :
nom, version, licence, URL d'accès et autorisation de redistribution.

Créer les dossiers locaux de données :

```bash
mkdir -p data/raw data/processed data/manifests data/predictions logs
```

Placer les images autorisées dans :

```text
data/raw/
```

Pour le dataset demandé dans le planning, télécharger RSNA Pneumonia depuis la
source officielle/Kaggle, puis le décompresser hors Git dans :

```text
data/external/rsna-pneumonia/
```

Extraire ensuite un échantillon local compatible avec notre pipeline :

```bash
python -m src.ingest.rsna_extract \
  --rsna-dir data/external/rsna-pneumonia \
  --output-dir data/raw/rsna_sample \
  --labels-csv data/raw/rsna_sample_labels.csv \
  --source-config data/source.local.json \
  --per-class 10
```

Cette commande copie un nombre limité de DICOM dans `data/raw/rsna_sample/`,
crée les labels locaux et remplit la provenance RSNA dans
`data/source.local.json`.

Optionnellement, préparer un fichier de labels en suivant l'exemple :

```bash
cp data/labels.example.csv data/raw/labels.csv
```

Les dossiers `data/raw/`, `data/processed/`, `data/predictions/` et les bases
SQLite restent hors Git.

### 4. Lancer l'ingestion

Transformer les images brutes en PNG prétraités et créer le manifeste :

```bash
python -m src.ingest \
  --input-dir data/raw/rsna_sample \
  --output-dir data/processed \
  --manifest data/manifests/ingest_manifest.csv \
  --source-config data/source.local.json \
  --labels-csv data/raw/rsna_sample_labels.csv
```

Si tu n'as pas encore de fichier `labels.csv`, lancer la même commande sans
`--labels-csv` :

```bash
python -m src.ingest \
  --input-dir data/raw \
  --output-dir data/processed \
  --manifest data/manifests/ingest_manifest.csv \
  --source-config data/source.local.json
```

Résultat attendu :

- images prétraitées dans `data/processed/images/` ;
- manifeste dé-identifié dans `data/manifests/ingest_manifest.csv`.

### 5. Lancer l'inférence

Lire le manifeste d'ingestion, analyser les images prétraitées et produire les
sorties JSON :

```bash
python -m src.inference \
  --manifest data/manifests/ingest_manifest.csv \
  --output-dir data/predictions/baseline_v1
```

Résultat attendu :

- `data/predictions/baseline_v1/cases/<case_id>.json` ;
- `data/predictions/baseline_v1/predictions.jsonl` ;
- `data/predictions/baseline_v1/run_metadata.json`.

### 6. Lancer la webapp

Avant ou après le lancement de la webapp, générer le rapport d'évaluation :

```bash
make evaluate-run
```

Le dashboard calcule aussi les métriques à la volée et affiche la matrice de
confusion quand les labels sont disponibles.

Le rapport est disponible dans la webapp :

```text
http://127.0.0.1:8000/report
```

La revue visuelle des erreurs est disponible ici :

```text
http://127.0.0.1:8000/errors
```

La page de synthèse du périmètre et des limites est disponible ici :

```text
http://127.0.0.1:8000/about
```

Installer les dépendances web minimales :

```bash
make webapp-setup
```

Lancer le dashboard avec les chemins par défaut :

```bash
make webapp-run
```

Ou lancer explicitement l'application :

```bash
python -m src.webapp \
  --predictions-dir data/predictions/baseline_v1 \
  --db-path logs/webapp.sqlite
```

Ouvrir ensuite dans le navigateur :

```text
http://127.0.0.1:8000
```

Pour arrêter la webapp, revenir dans le terminal où elle tourne et appuyer sur
`Ctrl+C`.

### 7. Commandes d'aide

Afficher les options de l'ingestion :

```bash
make ingest-help
```

Afficher les options d'extraction RSNA :

```bash
make rsna-extract-help
```

Afficher les options de l'inférence :

```bash
make inference-help
```

Afficher les options de l'évaluation :

```bash
make evaluate-help
```

Afficher les options de la webapp :

```bash
make webapp-help
```

### 8. Commandes Git utiles

Voir les fichiers modifiés :

```bash
git status --short
```

Créer un commit :

```bash
git add README.md docs/ src/ tests/ Makefile requirements*.txt
git commit -m "message clair du changement"
```

Pousser vers GitHub :

```bash
git push origin main
```

Avant un commit, vérifier qu'aucune donnée sensible ou image médicale n'a été
ajoutée par erreur.

## Workflow

1. Créer une issue ou une tâche avec un résultat vérifiable.
2. Partir de `main` avec une branche courte : `data/...`, `ia/...`, `web/...`,
   `docs/...` ou `chore/...`.
3. Ajouter les tests ou preuves liés au changement.
4. Ouvrir une pull request relue par un autre pôle.
5. Fusionner uniquement après la CI et la validation du contrat JSON.

Les détails, critères de fin et règles d'intégration sont dans
[`CONTRIBUTING.md`](CONTRIBUTING.md) et [`docs/workflow.md`](docs/workflow.md).

## Priorités

- **P1** : évaluation comparative et registre de 20 à 30 cas commentés ;
- **P2** : pipeline bout en bout, baseline, dataset documenté, interface et logs ;
- **P3** : sécurité, limites, licences et reproductibilité ;
- **P4** : LoRA/QLoRA ou autre extension seulement après stabilisation du socle.

Le plan de travail sur trois semaines est détaillé dans
[`docs/roadmap.md`](docs/roadmap.md).
