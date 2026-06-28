# Registre d'erreurs et analyse des cas — Assistant radiologue virtuel

**Périmètre :** baseline `image-stat-baseline-v0.2` évaluée sur un échantillon
équilibré de 30 radiographies thoraciques frontales issues de RSNA Pneumonia
Detection Challenge (10 cas par classe : `normal`, `suspected_opacity`,
`uncertain`).
**Rappel non clinique :** ce registre documente le comportement d'un prototype
pédagogique. Aucune ligne ci-dessous ne constitue un avis médical. Toute sortie
doit être validée par un professionnel qualifié.

---

## 1. Méthodologie de l'analyse d'erreurs

Chaque cas est confronté à son label de référence (gold). Les écarts sont classés
selon la taxonomie demandée :

- **Réussite** — la classe prédite correspond au gold.
- **Faux positif (FP)** — un cas `normal` est annoncé comme `suspected_opacity`.
- **Faux négatif dangereux (FN)** — un cas `suspected_opacity` est annoncé
  `normal` : c'est l'erreur la plus grave car elle « rate » un signe.
- **Incertitude acceptable** — l'outil répond `uncertain` : ce n'est pas compté
  comme une réussite stricte, mais c'est un comportement de sécurité, pas un échec.
- **Désaccord sur la classe `uncertain`** — le gold attendait `uncertain`
  (abstention), mais l'outil a tranché (`normal` ou `suspected_opacity`).
- **Erreur de format** — sortie JSON non conforme au contrat.
- **Hallucination** — affirmation visuelle absente de l'image.

Pour chaque erreur : cause probable + action corrective. Les confiances et
métriques proviennent de l'exécution réelle du pipeline (`metrics_report.json`,
`error_register.csv`).

---

## 2. Vue d'ensemble (30 cas)

| Indicateur | Valeur |
|---|---|
| Cas évalués | 30 |
| Réussites | 21 (70 %) |
| Erreurs | 9 (30 %) |
| Accuracy | 0,70 |
| Macro-F1 | 0,70 |
| Validité JSON | 100 % (0 erreur de format) |
| Hallucinations | 0 (baseline déterministe, justification gabarit) |
| Faux négatifs dangereux (baseline) | 2 |
| Faux négatifs dangereux (variante improved) | **0** |

**Répartition des 9 erreurs par type :**

| Type d'erreur | Nombre | Cas concernés |
|---|---|---|
| Faux positif (normal → opacité) | 1 | case_a33d53278ff47cc1 |
| Faux négatif dangereux (opacité → normal) | 2 | case_59322ea7aadf4144, case_9b969c62ec3c123e |
| Abstention sur un cas tranchable (→ uncertain) | 2 | case_0ccf0d54bcf04450, case_7bb580fe1b94090d |
| Désaccord sur la classe `uncertain` (→ normal) | 3 | case_4682fa9c204d63da, case_20c366743bf8f355, case_d31f2a58c3878fd7 |
| Désaccord sur la classe `uncertain` (→ opacité) | 1 | case_5cb081b4e4538efa |

Deux observations notables dès cette vue d'ensemble. D'abord, la **validité JSON
est de 100 %** et le **taux d'hallucination est nul** : la baseline est
déterministe et n'écrit pas de texte libre (sa justification est un gabarit), donc
elle ne peut pas inventer de signe absent. Ce point n'est pas un acquis définitif :
le risque d'hallucination redevient réel avec la variante MedGemma, qui génère du
texte, et devra être re-mesuré à ce moment-là. Ensuite, **les confiances sont
quasi quantifiées** (0,59 pour les `uncertain`, 0,88 pour les opacités détectées,
0,66–0,75 pour les `normal`) : le score reflète un palier de décision, pas une
probabilité calibrée. C'est une limite assumée (voir §6).

---

## 3. Tableau récapitulatif des 30 cas

| # | Cas | Gold | Prédiction | Confiance | Verdict |
|---|---|---|---|---|---|
| 1 | case_d3f74bcc6dc0c3e4 | normal | normal | 0,697 | ✅ Réussite |
| 2 | case_ad612c097bfd7e19 | normal | normal | 0,723 | ✅ Réussite |
| 3 | case_c0a57ae5ca4faa01 | normal | normal | 0,750 | ✅ Réussite |
| 4 | case_9bed95ffd0e0ecdd | normal | normal | 0,664 | ✅ Réussite |
| 5 | case_ae9ca72259051cee | normal | normal | 0,702 | ✅ Réussite |
| 6 | case_864c5d089261426d | normal | normal | 0,676 | ✅ Réussite |
| 7 | case_20b8e86f2a445b59 | normal | normal | 0,691 | ✅ Réussite |
| 8 | case_08e73cd027dcbac6 | normal | normal | 0,723 | ✅ Réussite |
| 9 | case_0ccf0d54bcf04450 | normal | uncertain | 0,590 | ⚠️ Abstention (sécurité) |
| 10 | case_a33d53278ff47cc1 | normal | suspected_opacity | 0,880 | ❌ Faux positif |
| 11 | case_7be8e40ac5526799 | suspected_opacity | suspected_opacity | 0,880 | ✅ Réussite |
| 12 | case_c39f62ff23b52932 | suspected_opacity | suspected_opacity | 0,880 | ✅ Réussite |
| 13 | case_74590b700a354dee | suspected_opacity | suspected_opacity | 0,880 | ✅ Réussite |
| 14 | case_6eac1f29ac64214a | suspected_opacity | suspected_opacity | 0,880 | ✅ Réussite |
| 15 | case_5ad6a53134a5a885 | suspected_opacity | suspected_opacity | 0,880 | ✅ Réussite |
| 16 | case_24cea45e7561bb2c | suspected_opacity | suspected_opacity | 0,880 | ✅ Réussite |
| 17 | case_327a59d7d603df29 | suspected_opacity | suspected_opacity | 0,880 | ✅ Réussite |
| 18 | case_7bb580fe1b94090d | suspected_opacity | uncertain | 0,590 | ⚠️ Abstention (sécurité) |
| 19 | case_59322ea7aadf4144 | suspected_opacity | normal | 0,731 | 🔴 Faux négatif dangereux |
| 20 | case_9b969c62ec3c123e | suspected_opacity | normal | 0,769 | 🔴 Faux négatif dangereux |
| 21 | case_4fbdff0c193b56d4 | uncertain | uncertain | 0,590 | ✅ Réussite (garde-fou) |
| 22 | case_bef2542016b7c567 | uncertain | uncertain | 0,590 | ✅ Réussite (garde-fou) |
| 23 | case_109eb958ce872bf3 | uncertain | uncertain | 0,590 | ✅ Réussite (garde-fou) |
| 24 | case_ff333c75129d82f7 | uncertain | uncertain | 0,590 | ✅ Réussite (garde-fou) |
| 25 | case_56090ca4fa68a726 | uncertain | uncertain | 0,590 | ✅ Réussite (garde-fou) |
| 26 | case_8915c92fd5543ea6 | uncertain | uncertain | 0,590 | ✅ Réussite (garde-fou) |
| 27 | case_4682fa9c204d63da | uncertain | normal | 0,697 | ❌ Désaccord (abstention attendue) |
| 28 | case_20c366743bf8f355 | uncertain | normal | 0,721 | ❌ Désaccord (abstention attendue) |
| 29 | case_d31f2a58c3878fd7 | uncertain | normal | 0,712 | ❌ Désaccord (abstention attendue) |
| 30 | case_5cb081b4e4538efa | uncertain | suspected_opacity | 0,880 | ❌ Désaccord (abstention attendue) |

*(Comptage : 21 réussites — dont 6 abstentions correctes sur la classe `uncertain` —
et 9 erreurs. 21 / 30 = 0,70.)*

---

## 4. Commentaires détaillés

### 4.1 Réussites (cas 1–8, 11–17, 21–26)

**Normaux correctement classés (cas 1 à 8).** Huit radios `normal` sur dix sont
bien reconnues. Les confiances s'étalent de 0,664 à 0,750 : la baseline les juge
« plutôt normales » sans excès d'assurance, ce qui est cohérent pour une
heuristique non médicale. Aucune action corrective nécessaire ; ces cas servent de
témoins de bon fonctionnement.

**Opacités correctement détectées (cas 11 à 17).** Sept radios
`suspected_opacity` sur dix sont bien repérées, toutes à 0,880. La confiance
identique révèle que la détection repose sur le franchissement d'un seuil unique
(concentration centrale de pixels clairs) plutôt que sur une mesure continue.
C'est efficace ici, mais fragile : c'est précisément ce que la variante improved
renforce avec un signal de texture locale.

**Abstentions correctes sur la classe `uncertain` (cas 21 à 26).** Six cas dont le
gold est `uncertain` reçoivent bien `uncertain` (0,590). C'est le cœur de la
politique de sécurité : quand l'image est ambiguë, l'outil refuse de trancher. Ces
réussites prouvent que le garde-fou se déclenche au bon moment et ne doit pas être
considéré comme un échec.

### 4.2 Faux positif — cas 10 (case_a33d53278ff47cc1)

- **Gold :** normal — **Prédiction :** suspected_opacity — **Confiance :** 0,880
- **Catégorie :** faux positif (alarme à tort).
- **Cause probable :** une zone centrale claire (probablement une structure
  médiastinale ou une surexposition locale) franchit le seuil
  `opacity_central_bright`. La baseline ne fait pas de segmentation anatomique :
  elle ne distingue pas un cœur/médiastin lumineux d'une vraie consolidation.
- **Conséquence :** moins grave qu'un faux négatif (une fausse alerte est
  « rattrapable » par la validation humaine), mais nuit à la précision sur la
  classe `normal`.
- **Action corrective :** introduire une pondération anatomique grossière
  (ignorer la bande médiastinale centrale) ou exiger une asymétrie minimale en plus
  de la luminosité centrale, pour éviter de confondre une structure normale claire
  avec une opacité.

### 4.3 Faux négatifs dangereux — cas 19 et 20 (case_59322ea7aadf4144, case_9b969c62ec3c123e)

- **Gold :** suspected_opacity — **Prédiction :** normal — **Confiances :** 0,731 et 0,769
- **Catégorie :** faux négatif dangereux (signe manqué). **C'est l'erreur la plus
  critique du projet.**
- **Cause probable :** opacités discrètes, à faible contraste ou en base
  pulmonaire, dont l'intensité reste sous le seuil `opacity_central_bright` de la
  baseline. Les statistiques globales d'image ne « voient » pas une consolidation
  localisée et peu marquée ; pire, la confiance reste moyenne-haute (0,73–0,77),
  donc l'outil se trompe avec assurance.
- **Action corrective (mise en œuvre) :** c'est exactement ce que corrige la
  variante `improved-v0.3`. En ajoutant un **signal de texture locale** (un patch
  homogène = consolidation possible) et une **règle d'incertitude**, ces deux cas
  ne sont plus annoncés `normal` : ils basculent vers une classe non dangereuse.
  Résultat mesuré sur l'échantillon : **faux négatifs dangereux 2 → 0**, sans
  dégrader le rappel sur les `normal`. C'est le gain de sécurité le plus important
  du projet et l'argument central de la soutenance.

### 4.4 Abstentions sur un cas tranchable — cas 9 et 18 (case_0ccf0d54bcf04450, case_7bb580fe1b94090d)

- **Cas 9 :** gold `normal`, prédit `uncertain` (0,590).
- **Cas 18 :** gold `suspected_opacity`, prédit `uncertain` (0,590).
- **Catégorie :** incertitude acceptable. L'outil n'a pas fait d'erreur
  *dangereuse* : il a refusé de trancher au lieu de se tromper de classe.
- **Cause probable :** signaux situés entre les seuils `normal` et `opacity`, donc
  la règle d'incertitude se déclenche. Pour le cas 18, c'est même prudent : plutôt
  `uncertain` que de rater l'opacité en disant `normal`.
- **Action corrective :** aucune en urgence. Ces cas illustrent le **compromis
  couverture / sécurité** : l'outil traite moins de cas mais évite les erreurs
  graves. À documenter comme un comportement voulu, pas comme un défaut. Un
  ajustement fin du seuil d'incertitude pourrait récupérer le cas 9 (un vrai
  normal), mais au risque de fragiliser la sécurité du cas 18.

### 4.5 Désaccords sur la classe `uncertain` — cas 27, 28, 29, 30

- **Cas 27, 28, 29 :** gold `uncertain`, prédit `normal` (0,697 / 0,721 / 0,712).
- **Cas 30 :** gold `uncertain`, prédit `suspected_opacity` (0,880).
- **Catégorie :** désaccord sur la classe d'abstention.
- **Lecture importante :** la classe `uncertain` du gold est une **construction du
  projet** (cas volontairement étiquetés « à ne pas trancher »). L'outil, lui, a
  jugé l'image suffisamment lisible pour répondre `normal` ou `opacité`. Ce ne sont
  donc pas des erreurs cliniques au sens strict, mais des désaccords sur le seuil
  d'abstention.
- **Cause probable :** la frontière entre « assez net pour répondre » et « trop
  ambigu » est intrinsèquement floue ; les seuils de la baseline ne reproduisent
  pas parfaitement le choix d'annotation. Le cas 30, classé opacité avec une forte
  confiance (0,880), est le plus discutable : l'outil voit un signal clair là où
  l'annotation préférait s'abstenir.
- **Action corrective :** élargir la **marge d'incertitude** (`uncertainty_margin`)
  pour que des signaux proches des seuils basculent en `uncertain`. C'est un
  réglage à double tranchant : il améliorerait l'accord sur ces 4 cas mais
  augmenterait le taux d'abstention global. À arbitrer en fonction de la priorité
  (sécurité vs couverture) et à mesurer, pas à supposer.

### 4.6 Erreurs de format et hallucinations

- **Erreurs de format JSON :** 0. Les 30 sorties respectent le contrat
  (`predicted_class`, `confidence_score`, `visual_findings`, `justification`,
  `limitations`, `warning`, versions, latence). La validité JSON est de 100 %.
- **Hallucinations :** 0. La baseline étant déterministe et sa justification étant
  un gabarit alimenté par des mesures réelles, elle ne peut pas affirmer un signe
  absent de l'image. **Attention :** ce constat ne vaut que pour la baseline. La
  variante MedGemma génère du texte libre ; le taux d'hallucination devra être
  re-mesuré et contrôlé (vérification automatique que la justification ne mentionne
  pas de structure non présente dans les features).

---

## 5. Synthèse des actions correctives

| Problème observé | Action corrective | Statut |
|---|---|---|
| 2 faux négatifs dangereux (opacités lues `normal`) | Signal de texture locale + règle d'incertitude (variante improved) | ✅ Mis en œuvre — FN 2 → 0 |
| 1 faux positif (structure claire normale lue opacité) | Pondération anatomique / asymétrie minimale | 🔧 Proposé |
| 4 désaccords sur la classe `uncertain` | Élargir la marge d'incertitude (compromis à mesurer) | 🔧 Proposé |
| Confiances non calibrées (paliers 0,59 / 0,88) | Calibration (reliability diagram, temperature scaling) | 🔧 Proposé (bonus) |
| Risque d'hallucination avec MedGemma | Vérification automatique justification ↔ features | 🔧 À prévoir si MedGemma activé |

---

## 6. Limites de cette analyse

- **Échantillon réduit (30 cas).** Les chiffres (0,70 d'accuracy, 2 FN) sont
  indicatifs, pas statistiquement robustes. Un échantillon plus large affinerait
  les conclusions.
- **Baseline non médicale.** L'`image-stat` mesure des statistiques d'intensité et
  de texture, sans aucune connaissance anatomique ni clinique. Ses réussites ne
  prouvent pas une compétence médicale, seulement la cohérence du pipeline.
- **Gold partiellement conventionnel.** La classe `uncertain` du gold reflète un
  choix d'annotation du projet ; les « désaccords » du §4.5 dépendent donc de ce
  choix autant que du modèle.
- **Confiance non calibrée.** Les scores reflètent des paliers de décision, pas des
  probabilités. Toute lecture probabiliste de la confiance serait abusive.

**Conclusion.** Sur 30 cas réels, le prototype atteint 70 % d'accuracy, ne produit
ni JSON invalide ni hallucination, et — surtout — son erreur la plus grave (les 2
faux négatifs dangereux) est éliminée par la variante améliorée (2 → 0). L'analyse
montre ce que le système sait faire (reconnaître des cas francs, s'abstenir quand
c'est ambigu), ce qu'il ne garantit pas (calibration, anatomie, opacités
discrètes), et comment chaque faiblesse identifiée est traitée ou pourrait l'être.
