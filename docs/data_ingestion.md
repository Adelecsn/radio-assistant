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
