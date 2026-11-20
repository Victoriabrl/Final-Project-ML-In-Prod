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

> Runs exécutés en local (`MLFLOW_TRACKING_URI=sqlite:///mlflow.db`) en attendant DagsHub.
> À refaire sur DagsHub et à remplacer par les run ids DagsHub.

| Run id | Modèle | Données | Commit | F1 spam | Précision | Rappel | Accuracy |
|---|---|---|---|---|---|---|---|
| `7e620799a58a4a4285f94332ef1919da` (local) | baseline | v1 | `e70655f` | 0,8837 | 0,8769 | 0,8906 | 0,9709 |
| `b41f4abc3a574d67a11cf5a708bb2319` (local) | candidat | v1 | `e70655f` | 0,9295 | 0,9912 | 0,8750 | 0,9835 |

## Choix du modèle de production

_À compléter après les démonstrations CT (phase 9)._ Scénario attendu d'après les mesures
ci-dessus : la baseline v1 (0,884) devient le premier champion ; le candidat entraîné sur v2
(0,953) l'améliore de +0,069 et est promu ; un nouveau candidat v2 identique (égalité) est
rejeté et le champion est conservé.
