# Expériences MLflow

## Métrique principale : F1 de la classe spam

Les classes sont déséquilibrées (≈ 12,4 % de spams) : un modèle qui répond toujours « ham » a
déjà ≈ 88 % d'accuracy. L'accuracy ne permet donc pas de comparer les modèles. On optimise le
**F1 de la classe spam** sur le held-out (plus haut = mieux), qui équilibre les deux erreurs :

- **faux positif** (ham classé spam) : un message légitime est caché à l'utilisateur ;
- **faux négatif** (spam classé ham) : une arnaque arrive dans la boîte de réception.

Les deux coûts sont réels et du même ordre ; le F1 pénalise un modèle qui sacrifie l'un pour
l'autre. Métriques secondaires suivies : précision spam, rappel spam, accuracy.

Tous les modèles sont évalués sur **le même held-out** (1 032 messages, 128 spams, identique
dans v1 et v2), jamais vu à l'entraînement.

## Règle de promotion (`ml/config/promotion.yaml`, figée avant toute évaluation)

| Paramètre | Valeur | Sens |
|---|---|---|
| `metric` | `f1_spam` | métrique comparée |
| `direction` | `maximize` | plus haut = mieux |
| `quality_floor` | 0,85 | le candidat doit au moins atteindre ce score |
| `min_improvement` | 0,01 | et battre le champion d'au moins ce delta |
| `retrain_when` | `new_data_version`, `champion_below_floor` | déclencheurs de réentraînement |

Sans champion, un candidat est promu s'il atteint le plancher. Avec un champion, il doit
atteindre le plancher **et** l'améliorer d'au moins 0,01 ; une égalité est rejetée.
Implémentation : fonctions pures `decide()` et `should_retrain()` dans `ml/promotion.py`,
testées dans `tests/test_promotion.py`.

## Modèles comparés

Pipeline scikit-learn : `ColumnTransformer` de `TfidfVectorizer` sur la colonne `text` →
`LogisticRegression` (solver liblinear). Le modèle est sérialisé par MLflow au format
**skops** (chargement sans exécution de code arbitraire, contrairement à pickle).

| Modèle | Fichier | Features | Classifieur |
|---|---|---|---|
| Baseline | `params_baseline.yaml` | mots unigrammes | C = 1,0, `class_weight=balanced` |
| Candidat | `params_candidate.yaml` | mots 1-2-grammes + caractères `char_wb` 3-5 | C = 10, `class_weight=balanced` |

### Écart au brief : `class_weight=balanced` sur la baseline

Le brief prévoyait une baseline « unigrammes, C = 1,0 » sans pondération. Mesurée sur le
held-out, elle n'atteint que **F1 = 0,247 sur v1** (précision 1,0, rappel 0,14) : avec
seulement 93 spams sur 800 messages et une régularisation forte, le modèle ne prédit
presque jamais « spam ». Elle ne passerait jamais le plancher de 0,85, et l'amorçage du CT
(« la baseline devient champion si elle passe le plancher ») serait impossible.

Variantes mesurées sur le held-out avant de trancher :

| Variante de baseline | F1 v1 | F1 v2 |
|---|---|---|
| unigrammes, C = 1 (brief) | 0,247 | 0,824 |
| idem sans `sublinear_tf` | 0,247 | 0,821 |
| idem solver lbfgs | 0,197 | 0,824 |
| unigrammes, C = 10 | 0,813 | 0,913 |
| **unigrammes, C = 1, `class_weight=balanced`** (retenue) | **0,884** | 0,908 |
| candidat (pour comparaison) | 0,929 | 0,953 |

Décision d'équipe : ajouter `class_weight=balanced` à la baseline, seul changement qui la fait
passer le plancher sur v1 en gardant un modèle simple.

## Tracking

Chaque run (`python -m ml.train`) logge dans l'expérience `sms-spam-training` :

- **tags** : `git_sha`, `git_dirty`, `data_version`, `dvc_md5` (md5 du `.dvc` de la release),
  `manifest_sha256`, `params_file`, `model_family` ;
- **paramètres** : tous les hyperparamètres (aplatis), tailles du train et du held-out ;
- **métriques** : `f1_spam`, `precision_spam`, `recall_spam`, `accuracy` (held-out) ;
- **artefacts** : `evaluation/confusion_matrix.png`, `evaluation/classification_report.txt`,
  `data/manifest.json`, `config/params_*.yaml`, et le modèle avec signature et exemple d'entrée.

Avec `--register`, une version est créée dans le registry avec les tags `data_version`,
`git_sha`, `f1_spam` et `status=candidate` (puis `approved` / `rejected` par le CT).

```bash
# local (registry SQLite) ou DagsHub selon MLFLOW_TRACKING_URI
python -m ml.train --data-version v1 --params ml/config/params_baseline.yaml --model-name sms-spam --register
python -m ml.train --data-version v1 --params ml/config/params_candidate.yaml --model-name sms-spam --register
python -m ml.evaluate --model-uri models:/sms-spam@champion --data-version v1
python -m scripts.check_registry_aliases   # vérifie le support des alias par le registry
```

## Runs

Runs exécutés le 2026-11-22 sur le serveur MLflow partagé DagsHub
(<https://dagshub.com/Victoriabrl/Final-Project-ML-In-Prod.mlflow>, expérience
`sms-spam-training`), depuis un checkout propre du commit `d3fecad` (`git_dirty=false`).
Les scores sont identiques aux runs locaux du commit `e70655f` : l'entraînement est
déterministe (seed 42, mêmes données v1, même holdout).

| Run id | Modèle | Données | Commit | F1 spam | Précision | Rappel | Accuracy |
|---|---|---|---|---|---|---|---|
| `40a7cc0c64324b82ba12ed37dc3322ba` | baseline | v1 | `d3fecad` | 0,8837 | 0,8769 | 0,8906 | 0,9709 |
| `55a36cd235b24c23a7dac77fb5a71af2` | candidat | v1 | `d3fecad` | 0,9295 | 0,9912 | 0,8750 | 0,9835 |

Ces runs ne sont pas enregistrés dans le registry (`--register` non utilisé) : la
version 1 de `sms-spam` sera créée par l'amorçage du CT (phase 6), comme prévu par le brief.

### Support des alias par DagsHub

Vérifié le 2026-11-22 avec `python -m scripts.check_registry_aliases` : un modèle factice
`alias-check` a été enregistré, l'alias `probe` posé puis relu
(`get_model_version_by_alias`) et le modèle rechargé via `models:/alias-check@probe`.
Les alias sont donc utilisables sur DagsHub pour `champion`.

## Continuous training et rollback

Le CT (`ct.yml` → `python -m ml.ct_pipeline`) et le rollback (`rollback.yml` →
`python -m ml.rollback`) loggent leurs décisions dans l'expérience **`sms-spam-ct`** :

- run parent **`ct-decision`** : tags `data_version`, `dvc_md5`, `manifest_sha256`, `git_sha`,
  `decision` (`promoted` / `rejected` / `kept` / `failed`), `reason`, `retrain_reasons`,
  `champion_version_before`, `champion_version_after`, `candidate_version`, `deployed` ;
  paramètres `metric`, `direction`, `quality_floor`, `min_improvement` ; métriques
  `champion_f1_spam` (champion rescoré sur le held-out de la release) et `candidate_f1_spam` ;
- le run d'entraînement du candidat est **imbriqué** sous `ct-decision` (mêmes tags et artefacts
  que les runs de `ml.train`) ;
- run **`rollback`** : `champion_version_before`, `champion_version_after`, `changed`, `reason`,
  `git_sha`.

Sur la version du registry : `status=candidate` à l'enregistrement, puis `approved` (promu) ou
`rejected` + `status_reason` ; la version promue porte `previous_champion` (cible par défaut du
rollback).

```bash
# en local, contre le registry de MLFLOW_TRACKING_URI (sans déploiement)
python -m ml.ct_pipeline --data-version v1                 # amorçage : baseline -> champion
python -m ml.ct_pipeline --data-version v2                 # nouvelles données -> candidat
python -m ml.ct_pipeline --data-version v2 --force-train   # réentraînement forcé
python -m ml.rollback --model-name sms-spam --version 1
```

### Démonstrations (phase 9)

Scénario prévu d'après les scores mesurés (entraînement déterministe) :

| # | Lancement | Résultat attendu |
|---|---|---|
| 1 | `ct.yml` v1 (registry vide) | amorçage : baseline v1 (0,884 ≥ 0,85) → version 1 = champion, déployée |
| 2 | publier v2 (tag `data-v2`), puis `ct.yml` v2 | `new_data_version` → candidat v2 (≈0,953) vs champion rescoré (0,884) : +0,069 → promu (version 2) |
| 3 | `ct.yml` v2 + `force_train` | candidat v2 identique : amélioration 0 < 0,01 → rejeté, champion 2 conservé |
| 4 | `rollback.yml` vers 1, puis vers 2 | la prod sert la version 1, puis à nouveau la meilleure (2) |

À éviter : `ct.yml` v1 avec `force_train` après l'amorçage promouvrait le candidat v1 (0,930,
+0,046) et décalerait les numéros de version du scénario.

Résultats réels (2026-11-23 et 2026-11-24, modèle `sms-spam`, workflows lancés depuis `main`) :

| # | Run `ct-decision` / `rollback` | Décision | Champion avant → après | Lien Actions |
|---|---|---|---|---|
| 1 | `904c7bbde38a4111acebc8046346e4c1` | promoted (amorçage, baseline 0,8837 ≥ 0,85) | aucun → 1 | [run 45247481242](https://github.com/Victoriabrl/Final-Project-ML-In-Prod/actions/runs/45247481242) |
| 2a | `5fc1a9d3…`, `a56c691c…`, `204ef291…` | failed → alias restauré | 1 → 1 (versions 2, 3, 4 `rejected`) | [run 45405778332](https://github.com/Victoriabrl/Final-Project-ML-In-Prod/actions/runs/45405778332) (tentatives 1 à 3) |
| 2b | `eb8e9a7a1e64494abcbd6954cebd0169` | promoted (0,9528 vs 0,8837, +0,0690) | 1 → 5 | [run 45423387046](https://github.com/Victoriabrl/Final-Project-ML-In-Prod/actions/runs/45423387046) |
| 3 | `1b6983e83f6f49c58d98e9f1827c9386` | rejected (0,9528 vs 0,9528, +0,0000 < 0,01), version 6 | 5 → 5, pas de redéploiement | [run 45424921198](https://github.com/Victoriabrl/Final-Project-ML-In-Prod/actions/runs/45424921198) |
| 4a | `2eeadb3137be42bab6a5c00df8b66dfa` (rollback) | cible par défaut = `previous_champion` | 5 → 1 | [run 45426082262](https://github.com/Victoriabrl/Final-Project-ML-In-Prod/actions/runs/45426082262) |
| 4b | `897702d4a37d44a4b25ab8a04337201f` (rollback) | cible explicite `5` | 1 → 5 | [run 45426583521](https://github.com/Victoriabrl/Final-Project-ML-In-Prod/actions/runs/45426583521) |

**Écart au scénario (démo 2a).** Les trois premiers lancements du CT v2 ont bien promu le
candidat, mais la vérification de production a échoué : le CT redéploie **la même image**
(seul l'alias change) et Render garde l'ancienne instance en service tant que la nouvelle
n'est pas prête. Le smoke test lisait donc `/ready` sur l'ancienne instance (version 1) et
échouait immédiatement. À chaque fois, le pipeline a fait ce qui est prévu en cas d'échec :
alias remis sur la version 1, version candidate marquée `rejected` (`status_reason` :
« production did not serve the new version after promotion »), production revérifiée en
version 1, workflow en échec. La production n'a jamais servi de modèle non vérifié.
Correctif (PR #26) : `scripts/smoke_test.py` attend que `/ready` annonce la version de
modèle attendue (valable aussi pour `rollback.yml`). Les tentatives 2 et 3 étaient des
« Re-run » du run d'origine, qui réutilisent le commit `2d63c16` d'avant le correctif ; le
lancement suivant depuis `main` (`36807d9`) a réussi (2b). Les versions 2 à 4 sont gardées
dans le registry comme trace. Les numéros de version diffèrent donc du scénario prévu
(champion final = version 5 au lieu de 2).

## Choix du modèle de production

Le modèle servi en production est **`sms-spam` version 5** (alias `champion`) : le candidat
TF-IDF (mots 1-2-grammes + caractères `char_wb` 3-5) + régression logistique (C = 10) entraîné sur la release **v2**, run
`029562c467804cdc9430c28e2f9ee6b7`, commit `36807d9`.

| Version | Modèle | Données | F1 spam (held-out) | Statut |
|---|---|---|---|---|
| 1 | baseline | v1 | 0,8837 | approved (premier champion, `previous_champion` de la 5) |
| 2, 3, 4 | candidat | v2 | 0,9528 | rejected (vérification de production, voir démo 2a) |
| **5** | candidat | v2 | **0,9528** | **approved, champion** |
| 6 | candidat | v2 | 0,9528 | rejected (amélioration +0,0000 < 0,01) |

Pourquoi ce modèle : c'est le meilleur sur la métrique principale (F1 spam 0,953 contre
0,884 pour la baseline, +0,069), au-dessus du plancher de 0,85, avec une précision spam de
0,960 (peu de faux positifs, donc peu de vrais SMS bloqués) et un rappel de 0,945. Il a été
choisi par la règle de promotion figée (`ml/config/promotion.yaml`), sans intervention
manuelle, puis vérifié en production par le smoke test. La version 1 reste disponible comme
cible de rollback (démo 4a).
