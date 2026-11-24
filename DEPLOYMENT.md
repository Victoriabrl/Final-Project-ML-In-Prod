# Déploiement

Comment le code, les données et le modèle arrivent en production, et ce qui se passe quand
quelque chose échoue.

## Architecture

```
                 ┌──────────────────────── GitHub ─────────────────────────┐
  développeuse   │  feature/* ─PR─▶ dev ─PR─▶ staging ─PR─▶ main            │
  ──push/PR────▶ │     ci.yml        ci.yml   staging.yml   deploy.yml      │
                 │                            ct.yml / rollback.yml (manuels)│
                 └───────┬──────────────────────┬──────────────┬───────────┘
                         │ push image           │ deploy hook   │ dvc pull
                         ▼                      ▼ (imgURL)      ▼
             ┌────────────────────┐   ┌──────────────────┐   ┌──────────────────┐
             │ GHCR               │──▶│ Render           │   │ Azure Blob       │
             │ ghcr.io/<owner>/   │   │ web service      │   │ remote DVC       │
             │ <repo>:sha-<commit>│   │ (image Docker)   │   │ dvc-store/       │
             └────────────────────┘   └────────┬─────────┘   │ sms-spam         │
                                               │ au démarrage └──────────────────┘
                                               ▼ alias champion → version → modèle
                                      ┌──────────────────────────────┐
                                      │ DagsHub (MLflow)             │
                                      │ tracking + model registry    │
                                      │ sms-spam@champion            │
                                      └──────────────────────────────┘
```

- **Code** : un seul dépôt GitHub ; image Docker construite par Actions, taguée par commit
  (`sha-<commit>`) et stockée sur GHCR.
- **Modèle** : jamais dans l'image. Au démarrage, l'application résout l'alias `champion` du
  modèle `sms-spam` en version concrète, vérifie `status=approved`, télécharge
  `models:/sms-spam/<version>` et fait une prédiction de test avant de se déclarer prête.
- **Données** : versionnées avec DVC (fichiers `.dvc` dans Git, contenu sur Azure Blob).
  Utilisées par l'entraînement et le CT, jamais par l'application.
- **Production** : un web service Render basé sur l'image ; health check sur `/ready` ;
  déploiement déclenché uniquement par le deploy hook (auto-deploy désactivé).

## Configuration et secrets

Tout passe par des variables d'environnement (`app/settings.py`, `.env.example`). Aucune valeur
secrète dans Git, dans l'image ou dans les logs.

| Nom | Où | Rôle |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `.env`, secret GitHub, env Render | `https://dagshub.com/Victoriabrl/Final-Project-ML-In-Prod.mlflow` |
| `MLFLOW_TRACKING_USERNAME` | idem | utilisateur DagsHub |
| `MLFLOW_TRACKING_PASSWORD` | idem (secret) | token DagsHub (lecture suffit pour Render) |
| `MODEL_NAME` / `MODEL_ALIAS` | env (défauts `sms-spam` / `champion`) | modèle et alias résolus au démarrage |
| `DRIFT_WINDOW_SIZE` / `DRIFT_RECOMPUTE_EVERY` | env (défauts `500` / `100`) | fenêtre de prédictions récentes et fréquence de recalcul du drift (`feature_drift`) |
| `GRAFANA_ADMIN_USER` / `GRAFANA_ADMIN_PASSWORD` | `.env` (stack locale) | connexion Grafana |
| `GF_SMTP_*` | `.env` (stack locale, secret) | serveur mail des alertes Grafana |
| `PROMETHEUS_CONFIG` | `.env` (défaut `prometheus.yml`) | `prometheus.prod.yml` pour scraper la prod (démo) |
| `LOG_LEVEL` | env (défaut `INFO`) | niveau des logs JSON (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `MLFLOW_CONFIGURE_LOGGING`, `MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR` | image (`false`) | MLflow n'installe pas son propre format de log ni de barre de progression : toutes les lignes restent du JSON |
| `PORT` | fourni par Render | port d'écoute (défaut 8000) |
| `AZURE_STORAGE_CONNECTION_STRING` | `.dvc/config.local`, secret GitHub | accès au remote DVC |
| `RENDER_DEPLOY_HOOK_URL` | secret GitHub | URL secrète du deploy hook |
| `PROD_URL` | variable GitHub | URL https publique |
| `GITHUB_TOKEN` | automatique | push des images sur GHCR (`packages: write`) |
| `GIT_SHA` | `--build-arg` au build | commit de l'image, exposé par `/health` |

## Les 12 facteurs

| Facteur | Mise en œuvre |
|---|---|
| I. Codebase | Un seul dépôt Git, plusieurs déploiements de la même base de code : CI (conteneur de test), staging (conteneur dans Actions), production (Render). |
| II. Dépendances | Versions exactes épinglées dans `requirements.txt` (service), `requirements-train.txt` et `requirements-dev.txt` ; même scikit-learn à l'entraînement et au service ; dépendances isolées dans l'image. |
| III. Config | Tout par variables d'environnement lues par `app/settings.py` au démarrage ; `.env.example` sans valeurs ; aucun secret par défaut. |
| IV. Services adossés | Registry MLflow DagsHub et Azure Blob sont des ressources attachées par URL + credentials : changer d'URL suffit (la CI utilise un serveur MLflow local à la place de DagsHub). |
| V. Build, release, run | Build : image `sha-<commit>` construite et vérifiée par Actions. Release : cette image + la config Render (variables). Run : le conteneur Render. Une même image passe de staging à la prod sans être reconstruite différemment. |
| VI. Processus | Processus sans état : aucun fichier local ; le modèle est rechargé depuis le registry à chaque démarrage. |
| VII. Port binding | Uvicorn écoute sur `$PORT` (`--host 0.0.0.0 --port ${PORT:-8000}`) ; l'application est autonome, sans serveur web externe. |
| VIII. Concurrence | Montée en charge horizontale par instances (même image, sans état). Limitée ici par l'offre gratuite Render : une seule instance, mise en veille après ~15 min d'inactivité. |
| IX. Jetable | Démarrage rapide (le chargement du modèle se fait en tâche de fond, `/health` répond tout de suite) ; arrêt propre via le lifespan FastAPI (`exec` dans la commande pour que uvicorn reçoive SIGTERM) ; readiness (`/ready`) séparée de la liveness (`/health`). |
| X. Parité dev/prod | Même Dockerfile et même image en CI, staging et prod ; seuls le registry et la config changent. Le même code de chargement tourne dans les tests (registry SQLite local). |
| XI. Logs | Flux d'événements JSON sur stdout (structlog, `app/logging_config.py`), une ligne par événement : `model_loaded`, `request_completed` (méthode, route, statut, durée), `prediction_made` (version, label, score, latence, taille du texte — jamais le texte). Chaque ligne porte un `request_id` (en-tête `X-Request-ID`, reçu ou généré, renvoyé au client ; `null` hors requête). Les logs d'uvicorn et de MLflow passent par le même format ; le log d'accès d'uvicorn est coupé (`--no-access-log`), remplacé par `request_completed`. Niveau par `LOG_LEVEL`. Aucun fichier : Docker (rotation) et Render (onglet *Logs*) collectent stdout. Les secrets connus sont masqués (`***`) même si une bibliothèque les écrit. |
| XII. Processus d'admin | Le CT (`python -m ml.ct_pipeline`) et le rollback (`python -m ml.rollback`) sont des tâches ponctuelles, lancées par `workflow_dispatch` ou en local, avec le même code et la même config. |

## Quelle branche déclenche quoi

| Événement | Workflow | Jobs / étapes | Effet en cas d'échec |
|---|---|---|---|
| PR vers `dev`, `staging`, `main` | `ci.yml` | `lint`, `tests`, `docker-smoke` | merge bloqué (checks requis) |
| merge dans `staging` | `staging.yml` | build `sha-<commit>`, conteneur contre DagsHub, smoke test, push GHCR | image non publiée |
| merge dans `main` | `deploy.yml` | build + vérification, push GHCR, deploy hook, attente du commit sur `/health`, smoke test live | job rouge ; Render garde l'instance précédente |
| manuel | `ct.yml` | voir [CT](#continuous-training-ctyml) | alias restauré, job rouge |
| manuel | `rollback.yml` | voir [Rollback](#rollback-rollbackyml) | alias restauré, job rouge |

## Déploiement continu (`deploy.yml`)

1. Résolution de la version attendue : `python -m scripts.resolve_champion` (alias `champion`,
   refuse une version non `approved`).
2. Build de l'image avec `GIT_SHA=<commit>`, lancement dans Actions contre DagsHub et smoke test
   (`/ready`, version attendue, prédiction spam et ham).
3. Push de `ghcr.io/<owner>/<repo>:sha-<commit>`.
4. Deploy hook Render avec `imgURL` = cette image (`scripts/deploy_render.sh`).
5. Attente que `PROD_URL/health` renvoie ce commit, puis smoke test live (vraie prédiction +
   version attendue). Délai large (15 min) : l'offre gratuite peut mettre plusieurs minutes.

`concurrency: production` : un seul déploiement (ou CT / rollback qui déploie) à la fois.

### Reproduire un déploiement à la main

```bash
IMAGE=ghcr.io/victoriabrl/final-project-ml-in-prod:sha-<commit>   # image déjà vérifiée
RENDER_DEPLOY_HOOK_URL=... sh scripts/deploy_render.sh "$IMAGE"
python scripts/smoke_test.py --url "$PROD_URL" --expected-git-sha <commit> \
       --expected-version <version du champion> --timeout 900
```

Redéployer une ancienne image (`sha-<ancien commit>`) revient à revenir à l'ancien code ; le
modèle, lui, est toujours celui de l'alias `champion` au démarrage.

## Continuous training (`ct.yml`)

Lancé à la main (Actions → ct → Run workflow) depuis `main`. Entrées : `dataset_version`
(v1 / v2), `model_name` (`sms-spam`, ou `sms-spam-staging` pour une démo contrôlée, jamais
déployée), `force_train`, `deploy`. Toute la logique est dans `ml/ct_pipeline.py`, testable en
local : `python -m ml.ct_pipeline --data-version v2 [--force-train]`.

1. `dvc pull data/releases/<version>.dvc` puis contrat de données ; échec → arrêt, rien n'est
   modifié dans le registry ni en production.
2. Score du champion sur le held-out de la release ; règle `should_retrain` (nouvelle version
   de données, ou champion sous le plancher). Sans raison et sans `force_train` : champion
   conservé.
3. Entraînement du candidat (`params_candidate.yaml` ; `params_baseline.yaml` s'il n'y a pas
   encore de champion = amorçage), enregistrement `status=candidate`, comparaison avec
   `decide()` (plancher 0,85 et amélioration ≥ 0,01 sur le même held-out).
4. Promu : `status=approved`, alias `champion` déplacé, tag `previous_champion`, deploy hook
   (le redémarrage résout à nouveau l'alias), smoke test sur `PROD_URL` avec la nouvelle
   version.
5. Rejeté : le champion reste, le candidat passe `status=rejected` avec la raison en tag.
6. Toujours : run MLflow parent `ct-decision` (expérience `sms-spam-ct`) avec version et hash
   des données, SHA Git, scores, seuils, décision, raison, versions avant / après ; le run
   d'entraînement est son enfant. Résumé dans le `$GITHUB_STEP_SUMMARY`.

## Rollback (`rollback.yml`)

Lancé à la main, entrée `target_version` (vide = tag `previous_champion` du champion actuel).
Une version sans `status=approved` est refusée, sans rien modifier. L'alias `champion` est
déplacé (aucun tag changé, donc un rollback s'annule par un autre rollback), l'opération est
loggée dans un run MLflow `rollback`, puis production est redéployée et vérifiée avec la version
cible.

En local :

```bash
python -m ml.rollback --model-name sms-spam --version 1
RENDER_DEPLOY_HOOK_URL=... sh scripts/deploy_render.sh "$IMAGE"
python scripts/smoke_test.py --url "$PROD_URL" --expected-version 1 --timeout 900
```

## Comportement en cas d'échec

| Situation | Ce qui se passe |
|---|---|
| Registry injoignable, alias absent ou version non approuvée au démarrage | L'application démarre (`/health` = 200) mais `/ready` et `/predict` renvoient 503 ; le chargement est réessayé en tâche de fond (backoff exponentiel). Jamais de prédiction sans modèle. |
| Nouvelle instance Render pas prête (health check `/ready` en échec) | Render ne bascule pas le trafic : l'instance précédente continue de servir ; après 15 min Render annule le déploiement. |
| Smoke test live en échec après `deploy.yml` | Le job échoue (visible sur la PR / l'onglet Actions) ; l'ancienne instance sert toujours. |
| Prod ne sert pas la nouvelle version après une promotion (CT) | L'alias est remis sur l'ancien champion, le candidat passe `rejected`, la prod est redéployée et vérifiée, le job échoue avec un message clair. |
| Prod ne sert pas la version cible après un rollback | L'alias est remis sur la version d'avant, redéploiement + vérification, job en échec. |
| Données invalides (contrat) | Le CT s'arrête avant toute modification. |
| Pas encore de champion (avant le premier CT) | `staging.yml` / `deploy.yml` vérifient seulement que l'image démarre (`/health`, `/ready` = 503) et la publient ; le déploiement Render est sauté avec un avertissement. |

## Écarts au brief (et pourquoi)

- **Smoke test CI via un serveur MLflow HTTP** au lieu d'un fichier SQLite monté dans le
  conteneur : l'image contient `mlflow-skinny`, sans SQLAlchemy, qui ne lit pas un registry
  SQLite. Plus proche de la prod (DagsHub est aussi un serveur HTTP).
- **`skops`** ajouté aux dépendances : MLflow 3 sérialise les modèles scikit-learn avec skops.
- **`exec` dans la commande Docker** : sans lui, `sh` ne transmet pas SIGTERM à uvicorn et
  l'arrêt n'est pas propre.
- **Commit Git dans l'image (`/health`)** : Render garde l'ancienne instance pendant un
  déploiement ; vérifier le commit prouve que l'URL live sert la nouvelle image.
- **Version du modèle attendue par le smoke test** : le CT et le rollback redéploient la
  **même image** (seul l'alias change), donc le commit ne distingue pas l'ancienne instance
  de la nouvelle. `scripts/smoke_test.py` attend que `/ready` annonce la version de modèle
  attendue avant de vérifier la prédiction (ajouté par la PR #26 après les premiers essais du
  CT v2, où la vérification lisait encore l'ancienne instance ; voir
  `ml/README_experiments.md`, démo 2a).
- **Amorçage sans champion** géré par staging et deploy (voir ci-dessus).

## Offre gratuite Render

Pas de carte bancaire demandée. Une instance, mise en veille après ~15 min d'inactivité
(premier appel ~1 min) : d'où les délais larges des smoke tests (`--timeout 900`).
