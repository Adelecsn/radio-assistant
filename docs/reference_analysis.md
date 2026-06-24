# Analyse de la base de référence

## Périmètre audité

L'archive `assistant-radiologue-virtuel-main.zip`, les quatre PDF du projet et
l'historique du dépôt `Adelecsn/radio-assistant` ont été examinés le 22 juin
2026. L'archive est extraite localement sous `reference/` et exclue de Git.

## Ce que la référence apporte

| Élément | Décision pour Radio Assistant |
|---|---|
| CI avec tests courts | Conservée et adaptée aux modules de l'équipe |
| Schéma JSON et warning | Conservés avec les noms de champs imposés par nos PDF |
| Comparaison baseline/amélioration | Conservée comme expérience principale |
| SQLite et registre d'erreurs | Conservés comme preuves, avec propriétaires côté web et évaluation |
| API et deux interfaces | Réduit à une seule interface avant toute extension |
| `src/` monolithique | Remplacé par `ingest/`, `inference/` et `webapp/` |
| Prédicteur basé sur le nom de fichier | Rejeté : utile pour un smoke test, pas comme résultat IA |
| Données synthétiques incluses | Non copiées; notre provenance doit être indépendante et documentée |
| Notebooks et stubs de fine-tuning | Reportés après le pipeline reproductible |

## Écarts repérés

La référence utilise notamment `confidence`, `visual_evidence`, `model_name` et
`latency_ms`. Les documents de notre équipe imposent `confidence_score`,
`visual_findings`, `model_version` et `inference_latency_ms`. Le contrat de ce
dépôt suit les documents de l'équipe afin d'éviter une incompatibilité entre
l'inférence, l'interface et l'évaluation.

Le dépôt GitHub actuel ne contenait avant cette adaptation que deux commits, une
branche `main`, les trois paquets vides et la liste de dépendances. Il manquait
une CI, des tests, un contrat partagé, une documentation d'architecture et des
règles de collaboration.

## Principes retenus

1. Un seul contrat partagé, versionné et testé.
2. Chaque pôle possède son module, mais le pipeline est intégré chaque semaine.
3. Les résultats sont jugés sur des données fixes et une analyse d'erreurs, pas
   sur une démonstration isolée.
4. Les datasets, poids et bases locales ne sont jamais copiés dans Git.
5. Toute amélioration doit annoncer une hypothèse et mesurer son effet contre la
   même baseline.
