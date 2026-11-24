# Détecteur de spam SMS en production

Projet de fin d'études — *Machine learning en production* (Efrei, S9). Une application web
classe un SMS (anglais) en **spam** ou **ham** avec une probabilité. Le modèle servi est la
version MLflow pointée par l'alias `champion` du registry DagsHub, résolue en version concrète
au démarrage et affichée par l'application.

## URL de production

**<https://final-project-ml-in-prod-sms-spam.onrender.com>**

- `/` : page web pour tester un SMS (zone de texte + exemples cliquables)
- `/model-info` : modèle servi (`model_name`, `alias`, `version`, `run_id`, `data_version`, `loaded_at`)
- `/docs` : documentation interactive de l'API (Swagger)

Offre gratuite Render : le service s'endort après ~15 min sans visite, la première requête
peut prendre environ une minute.

## Exemples d'entrées

| SMS | Attendu |
|---|---|
| `WINNER!! You have been selected to receive a £900 prize reward! Call 09061701461 to claim.` | spam |
| `URGENT! Your mobile number has won a £2000 bonus. Text CLAIM to 87575 now.` | spam |
| `Free entry in a weekly competition to win FA Cup final tickets. Text FA to 87121.` | spam |
| `Hey, are we still meeting for lunch tomorrow?` | ham |
| `I'll be home late tonight, don't wait for me for dinner.` | ham |
| `Can you send me the notes from today's class? Thanks!` | ham |

```bash
curl -X POST "$PROD_URL/predict" -H "Content-Type: application/json" \
     -d '{"text": "URGENT! Your mobile number has won a £2000 bonus. Text CLAIM to 87575 now."}'
# {"label":"spam","spam_probability":0.9…,"model_name":"sms-spam","model_version":"<version du champion>"}
```

Le texte est nettoyé (espaces en bord) et doit faire de 1 à 1000 caractères, sinon `422`.
Tant qu'aucun modèle approuvé n'est chargé, `/predict` et `/ready` renvoient `503`.

## API

| Route | Rôle |
|---|---|
| `GET /` | page HTML (zone de texte, bouton, résultat, exemples) |
| `POST /predict` | `{"text": str}` → `{"label", "spam_probability", "model_name", "model_version"}` |
| `GET /model-info` | version servie et sa provenance (run MLflow, version de données) |
| `GET /health` | liveness : le processus tourne (+ commit Git de l'image) |
| `GET /ready` | readiness : `200` seulement si le modèle approuvé est chargé et qu'une prédiction de test réussit |
| `GET /metrics` | métriques Prometheus (requêtes, latence, prédictions par classe et version, `model_info`, `feature_drift`) — voir [docs/MONITORING.md](docs/MONITORING.md) |

## Lancement local

Prérequis : Python 3.11, Docker (pour l'image), un accès DagsHub (token) dans `.env`.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env               # puis remplir MLFLOW_TRACKING_* (jamais commité)

# Qualité (comme la CI)
ruff check . && ruff format --check . && pytest

# Application, avec le registry DagsHub
set -a; source .env; set +a        # charge .env dans le shell (bash)
uvicorn app.main:app --reload      # http://localhost:8000

# Ou l'image Docker
docker build --build-arg GIT_SHA=$(git rev-parse HEAD) -t sms-spam:local .
docker run --rm -p 8000:8000 --env-file .env sms-spam:local
python scripts/smoke_test.py --url http://localhost:8000
```

Données (DVC, remote Azure Blob) : `dvc remote modify --local azure connection_string "..."`
puis `dvc pull data/releases/v1.dvc` (voir [ml/README_data.md](ml/README_data.md)).
Entraînement et CT en local : voir [ml/README_experiments.md](ml/README_experiments.md).

## Flux de branches

```
feature/<sujet> ──PR──▶ dev ──PR──▶ staging ──PR──▶ main
                         │            │               │
                    CI (lint,     CI + image testée  CI + image vérifiée,
                    tests,        dans Actions avec   push GHCR, déploiement
                    docker-smoke) le vrai registry,   Render, smoke test
                                  push GHCR           sur l'URL live
```

- Tout travail part d'une branche `feature/<sujet>` créée depuis `dev`.
- Aucun commit direct sur `dev`, `staging` ou `main` : uniquement des pull requests, relues et
  approuvées par un autre membre de l'équipe que l'auteur.
- Promotion : PR `dev → staging`, puis PR `staging → main`, jamais de raccourci.
- Les branches longues (`dev`, `staging`, `main`) sont protégées : PR obligatoire, 1 approbation,
  checks CI requis (`lint`, `tests`, `docker-smoke`), branche à jour, pas de force-push.

## Workflows

| Workflow | Déclencheur | Effet |
|---|---|---|
| `ci.yml` | PR vers `dev`, `staging`, `main` | Jobs `lint` (ruff), `tests` (pytest, registry MLflow local), `docker-smoke` (build de l'image, serveur MLflow de test, conteneur lancé, `smoke_test.py`). Un échec bloque le merge. |
| `staging.yml` | push sur `staging` (= merge) | Build de l'image `ghcr.io/<owner>/<repo>:sha-<commit>`, lancée dans Actions contre le vrai registry DagsHub, smoke test avec la version du champion, puis push sur GHCR. |
| `deploy.yml` | push sur `main` (= merge) | Build + vérification de l'image en conteneur, push GHCR, deploy hook Render avec `imgURL`, attente que l'URL live serve ce commit (`/health`), smoke test sur `PROD_URL` (vraie prédiction + version attendue). Un seul déploiement à la fois. |
| `ct.yml` | `workflow_dispatch` (`dataset_version`, `model_name`, `force_train`, `deploy`) | Continuous training : `dvc pull` + validation → score du champion → règle de réentraînement → entraînement et comparaison (plancher + amélioration minimale) → promotion + redéploiement + vérification live, ou champion conservé ; alias restauré si la prod ne suit pas. |
| `rollback.yml` | `workflow_dispatch` (`target_version`, `model_name`, `deploy`) | Remet l'alias `champion` sur une version approuvée (par défaut `previous_champion`), redéploie, smoke test avec la version cible ; restaure l'alias précédent si la prod ne la sert pas. |

Détails (architecture, 12 facteurs, reproduction d'un déploiement, échecs) :
[DEPLOYMENT.md](DEPLOYMENT.md).

## Structure du dépôt

```
app/        API FastAPI (chargement du champion, routes, page HTML)
ml/         données (DVC), entraînement, évaluation, promotion, CT, rollback
scripts/    registry de test, smoke test, résolution du champion, deploy hook Render
tests/      pytest (prédiction, validation, chargement, promotion, données, CT, rollback)
data/       fichiers .dvc uniquement (les données sont sur Azure Blob)
.github/    workflows CI / CD / CT / rollback
```

## Documentation

- [DEPLOYMENT.md](DEPLOYMENT.md) : architecture, 12 facteurs, CD, CT, rollback, comportement en cas d'échec
- [ml/README_data.md](ml/README_data.md) : source et licence des données, split, versions v1 / v2
- [ml/README_experiments.md](ml/README_experiments.md) : métrique, règle de promotion, runs MLflow
- [docs/MONITORING.md](docs/MONITORING.md) : monitoring (rejeu de trafic, logs, métriques, dashboard, alertes)

## Équipe

| Membre | Rôle | Branches principales |
|---|---|---|
| Lilou | A — Données / ML | `feature/data-pipeline`, `feature/promotion-rule`, `feature/training-mlflow`, `feature/ct-workflow` |
| Camille | B — App / tests | `feature/api-model-loader` |
| Victoria | C — DevOps | `feature/repo-setup`, `feature/docker`, `feature/ci`, `feature/rollback`, `feature/docs-deployment` |

Chaque membre pousse ses branches et ouvre ses PR depuis son compte ; un autre membre relit et
approuve. Les PR de promotion (`dev → staging → main`) sont ouvertes à tour de rôle.
