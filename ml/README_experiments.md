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

| # | Run `ct-decision` / `rollback` | Décision | Champion avant → après | Lien Actions |
|---|---|---|---|---|
| 1 | _à compléter_ | | | |
| 2 | _à compléter_ | | | |
| 3 | _à compléter_ | | | |
| 4 | _à compléter_ | | | |

## Choix du modèle de production

_À compléter après les démonstrations CT (phase 9)._ Scénario attendu d'après les mesures
ci-dessus : la baseline v1 (0,884) devient le premier champion ; le candidat entraîné sur v2
(0,953) l'améliore de +0,069 et est promu ; un nouveau candidat v2 identique (égalité) est
rejeté et le champion est conservé.
