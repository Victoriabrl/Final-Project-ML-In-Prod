# Données — SMS Spam Collection

## Source et licence

| | |
|---|---|
| Jeu de données | *SMS Spam Collection* — UCI Machine Learning Repository, id 228 |
| URL | https://archive.ics.uci.edu/dataset/228/sms+spam+collection |
| Archive téléchargée | https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip |
| Licence | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Citation | Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84 |
| Date de téléchargement | 2026-11-20 |
| SHA-256 du fichier brut `SMSSpamCollection` | `7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d` |

Le fichier brut contient un message par ligne au format `label<TAB>texte` (labels `ham` / `spam`).
Il est lu sans interprétation des guillemets (découpage sur la première tabulation).

## Reproduire

```bash
python -m ml.download_data              # -> data/raw/ (SMSSpamCollection + readme)
python -m ml.make_splits --release v1   # -> data/releases/v1/
python -m ml.validate_data --version v1 # contrat de données, code de sortie != 0 si échec
```

Ou, avec accès au remote DVC : `dvc pull data/releases/v1.dvc`.

## Nettoyage (`ml/make_splits.py`)

| Étape | Lignes |
|---|---|
| Messages bruts | 5 574 |
| Textes vides (après suppression des espaces en bord) | 0 supprimé |
| Textes présents avec des labels contradictoires | 0 |
| Doublons exacts de texte supprimés | 414 |
| **Messages uniques conservés** | **5 160** (dont 642 spams, 12,4 %) |

## Design du split

Tout est déterministe (seed 42) : relancer le script produit des fichiers identiques octet pour octet.

1. **Held-out** : 20 % des messages uniques, tirage stratifié sur le label → `holdout.csv`
   (1 032 messages, 128 spams). Il n'est **jamais** utilisé pour l'entraînement et il est
   **identique dans toutes les releases** (même fichier, même SHA-256), ce qui permet de comparer
   champion et candidat sur les mêmes observations.
2. **Pool** : les 4 128 messages restants sont mélangés, puis découpés en lots qui simulent
   l'arrivée progressive de nouveaux messages étiquetés.
3. **v1** : `train.csv` = les 800 premiers messages du pool (premier lot ; 93 spams, ≥ 80 requis).
4. **v2** : `train.csv` = tout le pool (lot de v1 + lots suivants), soit 4 128 messages, 514 spams.

Les textes de train et de held-out sont disjoints (vérifié par `validate_data.py`).

### Chronologie

Le temps ne joue pas de rôle ici : le fichier UCI ne contient **aucune date**, et l'ordre des
messages ne reflète pas leur ordre d'arrivée. Il n'y a donc pas de chronologie à préserver, et un
tirage aléatoire stratifié (seed fixe) est le split le plus représentatif. L'« arrivée » de
nouvelles données entre v1 et v2 est simulée par des lots reproductibles tirés du pool, jamais
par le held-out.

### Prétraitement : entraînement et évaluation séparés

- Le **nettoyage** (espaces en bord, textes vides, doublons exacts, labels contradictoires) est
  fait **avant** le split, ligne par ligne, sans rien apprendre des données. Supprimer les
  doublons avant le split évite qu'un même SMS se retrouve à la fois dans le train et dans le
  held-out.
- Le **prétraitement appris** (vectorisation TF-IDF : vocabulaire et poids IDF) fait partie du
  pipeline scikit-learn enregistré dans MLflow. Il est ajusté (`fit`) **uniquement sur le train**
  de la release, puis seulement appliqué (`transform`) au held-out et aux requêtes de l'API.
- Aucune augmentation de données n'est utilisée.

Chaque release contient `train.csv`, `holdout.csv` et `manifest.json` (version, seed, source,
statistiques de nettoyage, nombre de lignes et de spams par fichier, SHA-256 de chaque CSV,
description du changement).

## Contrat de données (`ml/validate_data.py`)

- colonnes exactement `label,text` ;
- labels ∈ {`ham`, `spam`} ; aucune valeur nulle ; aucun texte vide ;
- taux de spam entre 5 % et 30 % dans chaque fichier ;
- intersection vide entre les textes de train et de held-out ;
- SHA-256 et nombre de lignes conformes au manifeste.

## Changement v1 → v2

v2 simule l'arrivée de nouveaux messages étiquetés : le jeu d'entraînement passe de 800 à
4 128 messages (93 → 514 spams). Le held-out ne change pas.

## Versions

| Version | Train (lignes / spams) | Held-out (lignes / spams) | SHA-256 held-out | Tag Git | Commit | Hash DVC (md5 du `.dvc`) |
|---|---|---|---|---|---|---|
| raw | 5 574 messages bruts | — | — | — | _à compléter après merge_ | `6a7944dd155067d167bf352e1841a3f2.dir` |
| v1 | 800 / 93 | 1 032 / 128 | `ab9f9cb9…a67e9088d` | `data-v1` | `4c44f41` | `807e5f8cf19f3c946f158ea14b13fe2e.dir` |
| v2 | 4 128 / 514 | 1 032 / 128 | `ab9f9cb9…a67e9088d` | `data-v2` | _publiée plus tard_ | _publiée plus tard_ |

Le tag `data-v1` pointe sur le commit `4c44f41` (« data: publish release v1 »), intégré
à `dev` par la PR #2. Les données v1 et brutes ont été envoyées le 2026-11-22 sur le remote
DVC Azure Blob (`azure://dvc-store/sms-spam`, compte `projectsmsdevops`) ; un `dvc pull`
depuis un clone neuf suivi de `validate_data --version v1` a été vérifié.
