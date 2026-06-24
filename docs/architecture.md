# Architecture cible

## Flux principal

```text
Image + métadonnées autorisées
          |
          v
src/ingest       validation -> dé-identification -> prétraitement
          |
          v
src/inference    modèle -> parsing -> règle d'incertitude
          |
          v
src/contracts.py validation du JSON partagé
          |
          +--------------> eval/ métriques et registre d'erreurs
          |
          v
src/webapp       affichage non clinique + journalisation SQLite
```

## Responsabilités

### Ingest

Le pôle Data contrôle le format, la projection, la provenance et l'absence de
données identifiantes. Il produit une image prétraitée et des métadonnées
techniques stables. Il ne décide pas de la classe prédite.

### Inference

Le pôle IA charge le modèle, applique un prompt versionné, transforme la réponse
en dictionnaire et applique les règles d'incertitude. Il ne dépend ni de Gradio
ni de SQLite.

### Webapp

Le pôle Web reçoit l'image, orchestre les deux modules, affiche le warning et
journalise le run. L'interface ne corrige pas silencieusement une réponse
invalide : elle la fait basculer vers `uncertain` ou affiche une erreur contrôlée.

### Contrat partagé

[`src/contracts.py`](../src/contracts.py) constitue la frontière commune. Toute
modification de champ exige une pull request inter-pôles et la mise à jour du
schéma, du prompt, des tests, de l'interface et de l'évaluation.

## Stockage

- Git : code, prompts, métadonnées non sensibles et résultats agrégés légers ;
- stockage local/autorisé : images et poids de modèles ;
- SQLite local : identifiant technique du cas, versions, sortie, latence et
  commentaire d'évaluation, sans donnée patient.

## Premier jalon d'intégration

Une image autorisée traverse le pipeline, produit un JSON validé, apparaît dans
l'interface avec le warning et génère une ligne de log. Ce jalon précède le
dashboard et tout fine-tuning.
