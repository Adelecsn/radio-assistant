# Workflow de travail

## Tableau de travail

Chaque tâche contient un propriétaire, une priorité (`P1` à `P4`), un critère
d'acceptation et une preuve attendue. Limiter le travail en cours à une tâche par
personne aide à terminer les intégrations avant de démarrer des extensions.

Colonnes recommandées : `Backlog`, `Prêt`, `En cours`, `En revue`, `Terminé`.

## Branches et commits

- `main` reste exécutable et protégée ;
- branches courtes : `data/...`, `ia/...`, `web/...`, `docs/...`, `fix/...` ;
- commits ciblés au format `type(scope): description`, par exemple
  `feat(inference): ajoute le seuil d'incertitude` ;
- aucun dataset, poids, secret ou fichier SQLite dans les commits.

## Pull requests

Une PR doit expliquer le problème, la solution, la commande de validation et
l'effet sur le contrat JSON. Une personne d'un autre pôle relit les changements
qui touchent une frontière d'intégration.

Critères de fusion :

- CI verte ;
- test ou preuve reproductible ;
- documentation mise à jour si le comportement change ;
- warning et règle `uncertain` préservés ;
- aucun fichier sensible ou généré ajouté.

## Synchronisation

- point court régulier : blocages et décision d'interface, pas compte rendu
  détaillé de toutes les actions ;
- intégration de bout en bout en fin de chaque semaine ;
- démonstration sur le même petit jeu de cas versionné ;
- décisions techniques importantes consignées dans `docs/`.

## Définition de terminé

Une tâche n'est terminée que si son code est fusionné, vérifié dans le pipeline
commun et accompagné de sa preuve. Un notebook qui fonctionne uniquement dans
la session de son auteur ou une interface branchée sur une réponse simulée ne
valide pas le pipeline final.
