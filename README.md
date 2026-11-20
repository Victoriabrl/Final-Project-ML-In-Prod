# Détecteur de spam SMS en production

Projet de fin d'études — *Machine learning en production*. Une application web classe un SMS
(anglais) en **spam** ou **ham** avec une probabilité. Le modèle servi est la version MLflow
pointée par l'alias `champion` du registry DagsHub.

> Sections à compléter au fil des phases : URL de production, exemples d'entrées, lancement
> local, tableau des workflows, équipe.

## URL de production

_À venir (phase 5)._

## Lancement local

_À venir (phase 4)._ Prérequis : Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env             # puis remplir les valeurs
ruff check . && ruff format --check . && pytest
```

## Flux de branches

```
feature/<sujet> ──PR──▶ dev ──PR──▶ staging ──PR──▶ main
                         │            │               │
                    CI (lint,     CI + validation   CI + déploiement
                    tests,        de l'image        automatique sur Render
                    docker-smoke) en conteneur      + smoke test live
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
| `ci.yml` | PR vers `dev`, `staging`, `main` | _phase 5_ |
| `staging.yml` | push sur `staging` | _phase 5_ |
| `deploy.yml` | push sur `main` | _phase 5_ |
| `ct.yml` | `workflow_dispatch` | _phase 6_ |
| `rollback.yml` | `workflow_dispatch` | _phase 7_ |

## Équipe

| Membre | Rôle | Branches principales |
|---|---|---|
| Lilou | A — Données / ML | `feature/data-pipeline`, `feature/training-mlflow`, `feature/promotion-rule`, `feature/ct-workflow` |
| Camille | B — App / tests | `feature/api-skeleton`, `feature/model-loader`, `feature/web-ui`, `feature/tests` |
| Victoria | C — DevOps | `feature/repo-setup`, `feature/docker`, `feature/ci`, `feature/deploy-render`, `feature/rollback`, `feature/docs-deployment` |

Chaque membre pousse ses branches et ouvre ses PR depuis son compte ; un autre membre relit et
approuve. Les PR de promotion (`dev → staging → main`) sont ouvertes à tour de rôle.
