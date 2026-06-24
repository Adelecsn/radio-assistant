# Ingestion des radiographies

## Objectif

Cette étape transforme des images obtenues légalement en entrées homogènes pour
le modèle. Elle ne produit aucune prédiction. Son résultat est un dossier de PNG
normalisés et un manifeste CSV traçable.

## Procédure

1. Choisir la source et vérifier ses conditions d'accès, sa version, sa licence
   et ses restrictions de redistribution.
2. Télécharger les données hors de Git dans `data/raw/`.
3. Contrôler la dé-identification des métadonnées et des pixels.
4. Lancer le prétraitement reproductible.
5. Faire une revue visuelle des images transformées avant de marquer les cas
   comme utilisables.
6. Figer les splits avant de comparer les modèles.

Copier `data/source.example.json` vers un fichier local et remplacer chaque
valeur générique par les informations exactes de la source. Pour importer les
labels, préparer un CSV conforme à `data/labels.example.csv`.

## Source RSNA Pneumonia

Le planning projet demande de partir du dataset RSNA Pneumonia. Télécharger le
dataset depuis la source officielle, accepter les conditions d'utilisation, puis
le décompresser hors Git, par exemple dans :

```text
data/external/rsna-pneumonia/
```

Le dossier doit contenir au minimum :

```text
stage_2_train_images/
stage_2_train_labels.csv
stage_2_detailed_class_info.csv
```

Ensuite, extraire un petit échantillon équilibré vers `data/raw/` :

```bash
python -m src.ingest.rsna_extract \
  --rsna-dir data/external/rsna-pneumonia \
  --output-dir data/raw/rsna_sample \
  --labels-csv data/raw/rsna_sample_labels.csv \
  --source-config data/source.local.json \
  --per-class 10
```

La conversion de classes utilisée est :

- `Lung Opacity` ou `Target=1` -> `suspected_opacity` ;
- `Normal` -> `normal` ;
- `No Lung Opacity / Not Normal` -> `uncertain`.

Le fichier `data/raw/rsna_sample_selected_cases.csv` reste local et permet de
tracer les `patientId` pseudonymisés RSNA utilisés dans l'échantillon.

## Commande

```bash
python -m src.ingest \
  --input-dir data/raw \
  --output-dir data/processed \
  --manifest data/manifests/ingest_manifest.csv \
  --source-config data/source.local.json \
  --labels-csv data/raw/labels.csv \
  --target-size 512
```

Formats acceptés : PNG, JPEG et DICOM mono-image en niveaux de gris. Les DICOM
compressés peuvent nécessiter un décodeur supplémentaire et sont sinon rejetés.

## Transformations appliquées

- lecture des pixels sans recopier les métadonnées source ;
- application des transformations d'intensité DICOM disponibles ;
- inversion de `MONOCHROME1` ;
- normalisation robuste des intensités ;
- conversion en niveaux de gris ;
- redimensionnement avec conservation du ratio et padding noir ;
- export PNG sans EXIF ni en-tête DICOM ;
- nom de sortie dérivé du SHA-256 du contenu.

## Contrôles de confidentialité

Le manifeste ne conserve ni nom de fichier source ni chemin brut. Cette mesure
ne détecte toutefois pas un nom, une date ou un identifiant directement gravé
dans les pixels. Tous les cas reçoivent donc le statut
`pending_manual_pixel_review` jusqu'à une revue visuelle humaine.

Le champ DICOM `BurnedInAnnotation` est conservé uniquement sous forme de signal
`YES`, `NO` ou `UNKNOWN`; il ne remplace jamais cette revue.

## Contrat du manifeste

Le manifeste contient l'identifiant technique, le chemin transformé, la
provenance, le split, le label optionnel, les dimensions, le hash, les contrôles
techniques et le statut de confidentialité. Un nom de patient ou un identifiant
source dans ce fichier constitue un défaut bloquant.

## Critère de fin de l'étape 1

- la provenance est complète ;
- toutes les images sélectionnées ont un PNG 512 x 512 ;
- le manifeste ne contient aucune donnée identifiante ;
- les rejets et doublons sont comptés ;
- la revue visuelle de confidentialité est terminée ;
- les splits sont figés et reproductibles.
