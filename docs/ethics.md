# Éthique, sécurité et limites

## Position du projet

Radio Assistant est un prototype pédagogique. Il ne doit pas poser de
diagnostic, recommander un traitement, trier des patients ou remplacer une
expertise médicale. Une sortie techniquement valide n'est pas une validation
clinique.

## Avertissement obligatoire

Le texte suivant doit être visible dans l'interface et présent dans chaque
sortie structurée :

> Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise.

## Données autorisées

- données synthétiques ou datasets dont l'accès et la licence sont documentés ;
- images dé-identifiées, utilisées dans le respect des conditions de la source ;
- métadonnées minimales utiles à l'évaluation.

Ne jamais commiter de nom, date de naissance, identifiant patient, établissement
de santé ou autre donnée personnelle. Les images réelles et les secrets d'accès
restent hors de Git.

## Garde-fous

- retourner `uncertain` si l'image est de mauvaise qualité, si la confiance est
  inférieure à `0.60` ou si la sortie est invalide ;
- limiter `visual_findings` à des éléments visibles, sans inventer de contexte ;
- refuser les conclusions définitives et conserver l'avertissement ;
- journaliser modèle, prompt, latence et résultat pour chaque run ;
- examiner manuellement les hallucinations et les cas ambigus.

## Limites à communiquer

- données limitées et potentiellement non représentatives ;
- sensibilité au prompt, au modèle et au prétraitement ;
- confiance non nécessairement calibrée ;
- risque de sortie plausible mais incorrecte ;
- absence de validation clinique indépendante.
