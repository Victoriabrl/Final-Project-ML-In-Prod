# CLAUDE.md — Détecteur de spam SMS en production

Source de vérité : `docs/BRIEF_CLAUDE_CODE.pdf`. Ce fichier résume les règles (section 1) et les
décisions techniques (section 3). En cas de doute, relire le brief.

## Règles de travail (obligatoires)

### Actions manuelles : prévenir et s'arrêter
Toute étape qui demande une interface web, un compte, un secret, un paiement ou une validation
humaine : ne rien deviner ni contourner. Afficher ce bloc exact puis attendre la réponse :

```
=============== ACTION MANUELLE REQUISE ===============
Pourquoi : <raison en une phrase>
Où       : <site / page exacte>
Étapes   :
  1. ...
Valeurs à saisir : <noms exacts des secrets/variables, jamais de valeur secrète>
Vérification     : <commande ou contrôle>
Répondez « fait » quand c'est terminé (ou collez la sortie de la vérification).
========================================================
```

Toujours manuel : création de comptes (GitHub, DagsHub, Azure, Render), tokens/mots de passe,
secrets GitHub et variables Render, protection des branches, approbation et merge des PR,
passage public du package GHCR, lancement des `workflow_dispatch` de démo, captures d'écran.

### Git et pull requests
- Jamais de commit direct sur `main`, `staging` ou `dev`. Toujours `feature/<sujet>` créée depuis
  `dev`, puis PR vers `dev`.
- Promotion : PR `dev → staging`, puis PR `staging → main`. Aucun raccourci.
- PR ouvertes via `gh` si disponible ; revue, approbation et merge faits par un humain (autre
  membre que l'auteur).
- À chaque nouvelle branche, proposer quel membre la porte (push + PR depuis son compte) —
  l'historique des PR sert à noter les contributions.
- Commits petits, style Conventional Commits (`feat:`, `fix:`, `ci:`, `docs:`, `test:`, `chore:`).
- Les commits sont faits au nom de l'utilisateur uniquement : **aucune** ligne
  `Co-Authored-By: Claude` ni autre attribution à Claude.
- Avant chaque commit : `ruff check .`, `ruff format --check .` et `pytest` passent en local.

### Secrets et données
- Aucun secret dans Git, l'image Docker, les logs ou les messages. `.env` (ignoré) +
  `.env.example` sans valeurs.
- Credentials DVC uniquement dans `.dvc/config.local` (ignoré) ou en variables d'environnement.
- Aucune donnée brute/traitée dans Git : seulement `.dvc`, `dvc.lock` éventuel, manifestes JSON.
- Confirmation obligatoire avant toute action destructive (suppression de fichiers,
  `git push --force`, suppression de versions MLflow, réécriture d'historique).

### Manière de travailler
- Phase par phase (section 6 du brief). Fin de phase : résumé, fichiers créés, commandes de
  vérification, puis arrêt.
- Si une hypothèse du brief est fausse (version, URL, comportement de plateforme) : le signaler et
  proposer une alternative, ne pas bricoler en silence.
- Vérifier dans la doc officielle les points marqués « à vérifier ».
- Code en anglais (noms, commentaires). Documentation du rendu (README, DEPLOYMENT,
  README_data, README_experiments) en français.

## Décisions techniques (déjà prises)

| Sujet | Choix |
|---|---|
| Application | Classer un SMS anglais en `spam` / `ham`, avec probabilité |
| Dataset | UCI SMS Spam Collection (id 228), ~5 574 messages, ~13 % spam, CC BY 4.0 (à vérifier/citer) |
| Langage | Python 3.11 |
| Modèle | Pipeline scikit-learn `TfidfVectorizer` + `LogisticRegression`; même version de scikit-learn épinglée à l'entraînement et au service |
| Métrique principale | F1 de la classe spam sur le held-out (maximiser). Secondaires : précision/rappel spam, accuracy |
| API / web | FastAPI + Uvicorn ; page HTML statique servie par FastAPI (pas de framework front) |
| Tracking / registry | MLflow hébergé sur DagsHub. Modèles : `sms-spam` (prod), `sms-spam-staging` (démos) |
| Données | DVC, remote Azure Blob Storage |
| Images | GHCR : `ghcr.io/<owner>/<repo>:sha-<commit>` |
| Hébergement | Render, web service basé sur une image, déclenché par deploy hook secret |
| Qualité | ruff (lint + format), pytest, httpx (TestClient) |

### Dépendances séparées (versions exactes épinglées)
- `requirements.txt` (service / image) : fastapi, uvicorn, pydantic, scikit-learn, pandas,
  mlflow-skinny (vérifier qu'il suffit pour charger un modèle sklearn, sinon mlflow).
- `requirements-train.txt` : `-r requirements.txt` + mlflow, dvc[azure], matplotlib, pyyaml.
- `requirements-dev.txt` : `-r requirements-train.txt` + pytest, httpx, ruff.

### Configuration (variables d'environnement, lues par `app/settings.py`)
`MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` (secret),
`MODEL_NAME` (défaut `sms-spam`), `MODEL_ALIAS` (défaut `champion`), `PORT` (défaut 8000),
`AZURE_STORAGE_CONNECTION_STRING` (secret), `RENDER_DEPLOY_HOOK_URL` (secret GitHub),
`PROD_URL` (variable GitHub). Aucune valeur par défaut ne contient de secret.

## Commandes utiles
```bash
pip install -r requirements-dev.txt
ruff check . && ruff format --check . && pytest
```
