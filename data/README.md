# Données

Ce dossier ne doit contenir dans Git que des manifestes, métadonnées
dé-identifiées et petits exemples dont la redistribution est explicitement
autorisée. Les images médicales, archives téléchargées et données brutes restent
dans les dossiers ignorés.

Chaque manifeste doit au minimum documenter : `case_id`, chemin local, source,
version, licence ou condition d'accès, split, label de référence et contrôle de
dé-identification.

Le dataset d'évaluation est figé avant de comparer baseline et amélioration. Un
score obtenu sur des images synthétiques valide le logiciel, pas une performance
médicale.

Le pipeline et sa commande sont documentés dans
[`docs/data_ingestion.md`](../docs/data_ingestion.md). Utiliser
`source.example.json` et `labels.example.csv` comme contrats, jamais comme
informations réelles de provenance.

Les sorties d'inférence locales sont écrites dans `data/predictions/`, ignoré
par Git. Ne versionner que des résultats agrégés et relus, jamais un export brut
contenant des chemins locaux ou une donnée potentiellement sensible.
