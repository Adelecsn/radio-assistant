# Logs d'inférence

La base SQLite locale conserve uniquement les preuves techniques d'un run :
identifiant de cas, versions du modèle et du prompt, prédiction structurée,
latence et annotation d'erreur. Elle ne contient aucune identité patient.

Les fichiers `*.sqlite` et `*.db` sont générés localement et exclus de Git. Le
schéma SQL, lorsqu'il sera ajouté, reste versionné dans ce dossier.

La webapp initialise actuellement une table `case_views` avec : identifiant
technique du cas, classe prédite, qualité image, versions modèle/prompt et date
de consultation locale. Le fichier SQLite reste ignoré par Git.
