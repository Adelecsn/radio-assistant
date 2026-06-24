# Contribuer à Radio Assistant

## Avant de coder

1. Choisir une tâche avec un critère d'acceptation explicite.
2. Mettre `main` à jour et créer une branche correspondant au pôle.
3. Identifier si le contrat JSON ou une donnée sensible est concerné.

## Vérifications locales

```bash
make check
```

`make check` utilise le Python de `.venv` et crée cet environnement avec
`python3` s'il n'existe pas.

Les tests IA nécessitant un modèle lourd doivent être marqués séparément. La CI
standard doit rester rapide et ne pas télécharger de poids.

## Pull request

Utiliser le modèle fourni, garder la PR ciblée et demander une relecture externe
au pôle lorsqu'une interface partagée change. Ne pas fusionner une réponse de
modèle brute sans validation par `src/contracts.py`.

Consulter [`docs/workflow.md`](docs/workflow.md) pour les conventions complètes.
