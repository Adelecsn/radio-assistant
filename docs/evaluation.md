# Évaluation

## Objectif

Cette étape mesure la baseline sur les cas labellisés et prépare l'analyse
d'erreurs. Elle ne valide pas une performance médicale : elle indique seulement
comment le prototype se comporte sur l'échantillon choisi.

## Commande

Après l'inférence :

```bash
python -m eval.evaluate \
  --predictions-dir data/predictions/baseline_v1 \
  --output-dir eval/outputs/baseline_v1
```

Raccourci :

```bash
make evaluate-run
```

## Sorties

Le dossier d'évaluation contient :

- `metrics_report.json` : métriques globales, métriques par classe, matrice de
  confusion et erreurs ;
- `error_register.csv` : registre des cas à relire.

Le registre reprend le format de `eval/error_register_template.csv`.

## Métriques

- `accuracy` : proportion de prédictions égales au label ;
- `macro_precision` : précision moyenne sur les classes ;
- `macro_recall_sensitivity` : rappel/sensibilité moyenne ;
- `macro_f1` : F1 moyen ;
- matrice de confusion : lignes = label réel, colonnes = prédiction.

## Limites

Les labels RSNA sont utiles pour comparer des versions du prototype, mais ils ne
remplacent pas une validation clinique. Toute conclusion doit mentionner la
taille de l'échantillon, la source, les seuils utilisés et les erreurs observées.

## Résultat local actuel

Sur l'échantillon local de démonstration RSNA, la baseline
`image-stat-baseline-v0.2` corrige le défaut de la première version qui ne
préditait jamais `normal`. Les métriques locales observées sont environ :

- accuracy : 70 % ;
- macro-F1 : 70 % ;
- 9 erreurs à relire dans le registre.

Ces scores ne doivent pas être présentés comme une performance médicale. Ils
servent à montrer que le pipeline d'évaluation fonctionne et à guider l'analyse
d'erreurs.
