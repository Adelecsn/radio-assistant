# Roadmap sur trois semaines

## Semaine 1 - Contrats et pipeline minimal

- Data : provenance, manifeste des cas et prétraitement reproductible ;
- IA : baseline, prompt `v1.0`, parsing et sortie conforme au contrat ;
- Web : upload, warning visible et schéma SQLite minimal ;
- équipe : test d'intégration image -> JSON -> interface -> log.

Livrable de sortie : un run bout en bout rejouable, même si sa performance est
encore faible.

## Semaine 2 - Mesure et robustesse

- figer un ensemble d'évaluation commun et dé-identifié ;
- ajouter la variante améliorée et la règle d'incertitude ;
- calculer accuracy, macro-F1, sensibilité et matrice de confusion ;
- remplir au moins 15 lignes du registre d'erreurs ;
- afficher les métriques essentielles depuis les logs.

Livrable de sortie : comparaison chiffrée sur les mêmes cas et première analyse
des faux positifs, faux négatifs, incertitudes et erreurs de format.

## Semaine 3 - Validation et soutenance

- compléter 20 à 30 cas commentés ;
- vérifier licences, provenance, limites et absence de données sensibles ;
- stabiliser la démonstration et documenter les commandes ;
- présenter réussites, échecs et décisions de conservation ou de rejet.

Livrable de sortie : dépôt reproductible, rapport critique et démonstration
fondée sur des preuves.

## Règle de priorité

Les extensions LoRA/QLoRA, la multiplication des interfaces ou les dashboards
avancés restent en attente tant que le pipeline, les métriques et le registre
d'erreurs ne sont pas stables.
