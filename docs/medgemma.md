# Backend MedGemma (modèle multimodal)

> **Usage non clinique.** Prototype pédagogique. Non destiné au diagnostic.
> Validation par un professionnel qualifié requise.

MedGemma est branché comme **variante d'inférence** (`--variant medgemma`) derrière
le **même contrat JSON** que la baseline statistique. Le code propose les champs
cliniques via le modèle, puis impose le contrat par-dessus : avertissement littéral,
versions/latence ajoutées par nous, et **bascule automatique vers `uncertain`** si le
JSON est invalide, si la qualité est `poor` ou si la confiance est sous le seuil.

Le code vit dans [`src/inference/medgemma.py`](../src/inference/medgemma.py). Les
dépendances lourdes (`torch`, `transformers`) sont chargées **paresseusement** : le
reste du projet et les tests tournent sans GPU.

## Prérequis

- un **compte Hugging Face** ;
- **accepter la licence** du modèle `google/medgemma-4b-it` (Health AI Developer
  Foundations) sur sa page Hugging Face — c'est la variante **instruction-tuned**,
  seule à fournir le *chat template* requis par le backend. La variante `-pt`
  (pré-entraînée) n'a pas de chat template et ferait basculer toutes les sorties
  vers `uncertain` ;
- un **token Hugging Face** (`Settings > Access Tokens`, droit *read*) ;
- matériel : **GPU recommandé** (≈ 8–10 Go VRAM en bfloat16). Sur CPU ça fonctionne
  mais c'est lent ; réserver à 2–3 images de démonstration.

## Étape 1 — Installer les dépendances

```bash
source .venv/bin/activate
pip install -r requirements.txt   # inclut torch, transformers, accelerate, huggingface_hub
```

## Étape 2 — Accepter la licence du modèle

Ouvrir <https://huggingface.co/google/medgemma-4b-it>, se connecter, puis cliquer
sur **« Agree and access repository »**. Sans cette étape, le téléchargement renvoie
une erreur 401/403.

## Étape 3 — S'authentifier

```bash
huggingface-cli login        # colle ton token quand demandé
# ou, sans interaction :
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

## Étape 4 — Lancer la baseline MedGemma (prompt v1)

```bash
python -m src.inference --variant medgemma \
  --manifest data/manifests/ingest_manifest.csv \
  --output-dir data/predictions/medgemma_v1 \
  --device auto
```

Le premier lancement télécharge les poids (~8 Go) puis les met en cache. Les
prédictions par cas suivent exactement le contrat (`cases/`, `predictions.jsonl`,
`run_metadata.json`).

## Étape 5 — Comparer deux prompts (exigence du barème)

Relancer avec le **prompt renforcé** ([`prompts/improved_v1.txt`](../prompts/improved_v1.txt)) :

```bash
python -m src.inference --variant medgemma \
  --prompt-file prompts/improved_v1.txt \
  --prompt-version medgemma-v2.0 \
  --manifest data/manifests/ingest_manifest.csv \
  --output-dir data/predictions/medgemma_v2
```

## Étape 6 — Évaluer et comparer (automatique, même contrat)

```bash
python -m eval.evaluate --predictions-dir data/predictions/medgemma_v1 \
  --output-dir eval/outputs/medgemma_v1
python -m eval.evaluate --predictions-dir data/predictions/medgemma_v2 \
  --output-dir eval/outputs/medgemma_v2

# prompt baseline vs prompt renforcé
python -m eval.compare \
  --baseline-dir data/predictions/medgemma_v1 \
  --improved-dir data/predictions/medgemma_v2 \
  --output-dir eval/outputs/medgemma_prompt_compare

# ou MedGemma vs baseline statistique
python -m eval.compare \
  --baseline-dir data/predictions/baseline_v1 \
  --improved-dir data/predictions/medgemma_v1 \
  --output-dir eval/outputs/medgemma_vs_baseline
```

## Démo web avec MedGemma

```bash
python -m src.webapp --variant medgemma --predictions-dir data/predictions/medgemma_v1
```

Les uploads (`POST /predict`) sont alors analysés par MedGemma. Le premier upload
charge le modèle (lent), les suivants réutilisent le cache.

## Options utiles

- `--device auto|cuda|cpu` : placement du modèle (`device_map`) ;
- `--model-id` : autre modèle (ex. une variante MedGemma) ;
- `--max-new-tokens` : budget de génération (défaut 512) ;
- `--prompt-file` / `--prompt-version` : pour l'ablation de prompts.

## Dépannage

| Symptôme | Cause probable | Solution |
|---|---|---|
| `401/403 gated repo` | licence non acceptée ou token absent | Étapes 2 et 3 |
| `RuntimeError: MedGemma nécessite torch...` | dépendances manquantes | `pip install -r requirements.txt` |
| `CUDA out of memory` | VRAM insuffisante | `--device cpu`, ou réduire le lot |
| Très lent | exécution CPU | normal ; limiter à quelques images |
| `predicted_class: uncertain` partout | JSON modèle invalide | garde-fou volontaire ; vérifier le prompt et la latence |
| `does not have a chat template` | variante `-pt` utilisée au lieu de `-it` | utiliser `google/medgemma-4b-it` (défaut actuel) et accepter sa licence |

## Position honnête

MedGemma reste un prototype **non clinique** : ses sorties sont expérimentales,
journalisées et soumises à relecture humaine. Le but pédagogique est la méthode
(contrat, garde-fous, métriques, comparaison), pas un diagnostic.
