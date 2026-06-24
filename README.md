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

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Pour contribuer sans installer les dépendances IA lourdes :

```bash
make test
```

Cette commande crée automatiquement `.venv` avec `python3`, installe les
dépendances de test et lance `pytest`. Sans `make`, utiliser directement :

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-test.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q src eval tests
```

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
