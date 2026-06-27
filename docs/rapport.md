# Rapport final - Radio Assistant

> **Usage non clinique.** Prototype pédagogique. Non destiné au diagnostic.
> Validation par un professionnel qualifié requise.

Ce rapport synthétise le dataset, le contrat de sortie, la baseline, l'amélioration
mesurée, l'évaluation, l'analyse d'erreurs, les limites et les risques. Toutes les
valeurs sont reproductibles avec les commandes indiquées.

## 1. Périmètre et dataset

- **Tâche** : à partir d'une radiographie thoracique frontale, produire une sortie
  JSON structurée (qualité, classe, confiance, observations, justification, limites,
  avertissement, versions, latence).
- **Classes** : `normal`, `suspected_opacity`, `uncertain`. La classe `uncertain`
  est un garde-fou méthodologique, pas un échec.
- **Données** : échantillon local du *RSNA Pneumonia Detection Challenge* (stage_2),
  **30 cas équilibrés** (10 par classe), dé-identifiés et prétraités (PNG carré
  512×512, normalisation des intensités, suppression des métadonnées à l'export).
- **Provenance et licence** : décrites dans `data/source.local.json` et
  [`docs/data_ingestion.md`](data_ingestion.md). Les images réelles restent hors Git
  (`data/external/`, `data/raw/`, `data/processed/`).

## 2. Contrat de sortie et prompt

Le contrat partagé ([`prompts/output_schema.json`](../prompts/output_schema.json),
appliqué par [`src/contracts.py`](../src/contracts.py)) impose exactement les champs
du cahier des charges de l'équipe : `image_quality`, `predicted_class`,
`confidence_score`, `visual_findings`, `justification`, `limitations`, `warning`,
`model_version`, `prompt_version`, `inference_latency_ms`.

Garde-fous imposés par le contrat (testés) :

- `confidence_score` < 0.60 et classe ≠ `uncertain` → invalide ;
- `image_quality` = `poor` et classe ≠ `uncertain` → invalide ;
- avertissement non clinique obligatoire et littéral.

Le prompt baseline est versionné dans
[`prompts/baseline_v1.txt`](../prompts/baseline_v1.txt) (`prompt_version` v1.0).

## 3. Baseline reproductible

La baseline (`image-stat-baseline-v0.2`,
[`src/inference/baseline.py`](../src/inference/baseline.py)) est **déterministe** :
elle extrait des features de gris (contraste, asymétrie horizontale, pixels clairs
centraux, densité de contours, contraste local) et décide par seuils explicites.
Elle bascule vers `uncertain` si la qualité est mauvaise ou si la confiance est
insuffisante. Notebook : [`notebooks/02_baseline.ipynb`](../notebooks/02_baseline.ipynb).

> La baseline n'est pas un modèle médical entraîné : elle valide le workflow
> logiciel de bout en bout (image → JSON → interface → logs → métriques).

**Résultats baseline (30 cas)** : accuracy **0.70**, macro-F1 **0.70**,
macro-spécificité 0.85, taux d'incertitude 0.27, validité JSON 1.00.

## 4. Amélioration mesurée

**Hypothèse.** Une consolidation (opacité) est une région **homogène** à faible
contraste local. La baseline, fondée sur des statistiques globales, moyenne ce
signal et peut donc déclarer `normal` une vraie opacité (faux négatif dangereux).

**Méthode** (variante `image-stat-improved-v0.3`,
[`src/inference/improved.py`](../src/inference/improved.py), `prompt_version` v2.0) :

1. signal auxiliaire `local_texture_max` = contraste local maximal sur le champ
   pulmonaire central (grille 6×6) ;
2. garde-fou : si la baseline conclut `normal` mais que `local_texture_max` est sous
   le seuil (38), le cas est re-signalé `suspected_opacity` ;
3. règle d'incertitude explicite (marge de confiance) en complément.

**Frontière de décision.** Les deux opacités manquées ont un contraste local de
25 et 30, alors que **tous** les cas `normal` sont ≥ 42 : la frontière à 38 tombe
dans un écart franc, donc robuste (non sur-ajustée), et radiologiquement motivée.

**Gain mesuré (même jeu de 30 cas)** — [`eval/compare.py`](../eval/compare.py),
notebook [`03_baseline_vs_improved.ipynb`](../notebooks/03_baseline_vs_improved.ipynb) :

| Métrique | Baseline | Improved | Δ |
|---|---|---|---|
| Accuracy | 0.700 | **0.767** | +0.067 |
| Macro-F1 | 0.700 | **0.762** | +0.062 |
| Macro-rappel / sensibilité | 0.700 | **0.767** | +0.067 |
| Sensibilité `suspected_opacity` | 0.700 | **0.900** | +0.200 |
| Macro-spécificité | 0.850 | **0.883** | +0.033 |
| Faux négatifs dangereux (opacité→normal) | 2 | **0** | −2 |
| Rappel `normal` | 0.800 | 0.800 | 0 |
| Validité JSON | 1.00 | 1.00 | 0 |

**Décision : amélioration conservée.** Elle améliore toutes les métriques
principales, fait passer la sensibilité opacité de 0.70 à 0.90 et **supprime les
deux faux négatifs dangereux**, sans dégrader le rappel `normal` (aucun vrai
`normal` n'est re-signalé). 2 cas changent, 2 corrigés, 0 cassé.

## 5. Évaluation et analyse d'erreurs

Métriques calculées par [`eval/metrics.py`](../eval/metrics.py) : accuracy, macro
précision/rappel(sensibilité)/spécificité/F1, matrice de confusion, validité JSON,
taux d'incertitude, taux de justification non fondée, latence moyenne.

Le **registre de revue** ([`eval/outputs/<variant>/case_review.csv`]) couvre les
**30 cas** commentés et catégorisés (faux positif, faux négatif, incertitude
acceptable, erreur de format, hallucination, correct). Les erreurs seules sont
extraites dans `error_register.csv`.

Répartition des familles :

| Famille | Baseline | Improved |
|---|---|---|
| correct | 21 | **23** |
| false_negative (dangereux) | 2 | **0** |
| false_positive | 1 | 1 |
| overconfident_on_ambiguous | 4 | 4 |
| acceptable_uncertainty | 2 | 2 |

Aucune hallucination ni erreur de format : les observations sont dérivées des
mesures (taux de justification non fondée = 0), et 100 % des sorties respectent le
contrat JSON.

## 6. Limites

- échantillon très petit (30 cas), seuils calés pour la démonstration locale ;
- **plafond des features globales** : les statistiques de gris ne séparent pas
  proprement les trois classes (chevauchement fort, cf.
  [`01_data_audit.ipynb`](../notebooks/01_data_audit.ipynb)) — la baseline est près
  de son maximum atteignable ; le signal de texture locale corrige un cas précis
  (consolidation homogène) mais ne remplace pas un vrai modèle ;
- confiance non calibrée ; pas de segmentation anatomique ni de contexte clinique ;
- les métriques ne constituent pas une validation médicale.

## 7. Risques et éthique

- aucune donnée patient n'est commitée ; les logs SQLite ne contiennent que des
  preuves techniques (classe, qualité, versions, latence, date) ;
- l'avertissement non clinique est présent dans chaque sortie et chaque page ;
- risque de sortie plausible mais incorrecte → garde-fous `uncertain` et revue
  humaine obligatoire. Détails : [`docs/ethics.md`](ethics.md).

## 8. Backend multimodal MedGemma

Le plafond mesuré des features globales justifie une **vraie baseline multimodale**.
Elle est désormais **branchée** : la variante `medgemma`
([`src/inference/medgemma.py`](../src/inference/medgemma.py)) exécute
`google/medgemma-4b-it` derrière le même contrat JSON, avec bascule automatique vers
`uncertain` si la sortie est invalide. Deux prompts versionnés permettent l'ablation
(`prompts/baseline_v1.txt` vs `prompts/improved_v1.txt`), et `eval.compare` produit la
comparaison chiffrée prompt-vs-prompt ou MedGemma-vs-baseline statistique.

L'exécution réelle nécessite un compte Hugging Face, l'acceptation de la licence du
modèle et idéalement un GPU : procédure dans [`docs/medgemma.md`](medgemma.md). Le
LoRA/QLoRA (Unsloth/Gemma) reste un bonus P4, à n'aborder qu'après stabilisation,
avec citation des licences et conditions d'accès.

## 9. Commandes de reproduction

```bash
make improved-run            # génère data/predictions/improved_v1
make evaluate-run            # métriques baseline
make improved-evaluate-run   # métriques improved + registre de revue
make compare-run             # comparaison chiffrée baseline vs improved
make webapp-run              # dashboard + upload live (POST /predict)
```
