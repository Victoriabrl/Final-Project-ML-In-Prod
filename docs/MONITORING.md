# Monitoring

Logs structurés, métriques Prometheus, dashboard Grafana et alertes email du service de
détection de spam SMS. Ce document est complété étape par étape (voir la PR de chaque étape).

## Générer du trafic : lots de rejeu et `scripts/replay.py`

### 1. Construire les lots (reproductibles)

```bash
dvc pull data/releases/v2.dvc
python -m ml.make_replay_sets --data-version v2      # -> .exercise-data/
```

| Fichier | Contenu | Effet attendu |
|---|---|---|
| `normal.csv` | held-out v2 mélangé (1 032 SMS, ~12 % de spam) | trafic semblable à l'entraînement : drift sous 0,20 |
| `campaign.csv` | 600 spams (train + held-out v2), la moitié en MAJUSCULES | « campagne de spam » : longueur, chiffres, majuscules, liens et score montent, drift au-dessus de 0,20 |
| `manifest.json` | nombre de lignes et sha256 de chaque fichier | même release + même graine = mêmes fichiers |

`.exercise-data/` est ignoré par Git et par l'image Docker (aucune donnée versionnée ni
embarquée). Les SMS viennent de la release DVC : rien n'est téléchargé d'ailleurs.

### 2. Rejouer contre l'API

```bash
# trafic normal, 10 requêtes/s
python scripts/replay.py --file .exercise-data/normal.csv --url http://127.0.0.1:8000 --rate 10 --shuffle
# avec ~20 % de requêtes invalides (texte vide, champ manquant, > 1000 caractères, JSON cassé)
python scripts/replay.py --file .exercise-data/normal.csv --broken --limit 200
# campagne de spam (fait monter le drift)
python scripts/replay.py --file .exercise-data/campaign.csv --rate 10
# contre la production (le service gratuit se réveille en ~1 min)
python scripts/replay.py --file .exercise-data/normal.csv --url https://final-project-ml-in-prod-sms-spam.onrender.com --rate 2 --limit 50
```

Options : `--file` (CSV avec une colonne `text`), `--url`, `--rate` (requêtes/s, 0 = sans
limite), `--limit`, `--shuffle`, `--seed`, `--broken` / `--broken-ratio`, `--timeout`.

Chaque requête a son propre `X-Request-ID` (UUID). Le script affiche le nombre de réponses
par statut HTTP, par type de requête, les labels prédits, les versions de modèle qui ont
répondu et la latence p50 / p95. Code de sortie 1 si l'API renvoie des 5xx ou ne répond pas
(les 4xx sont normaux en mode `--broken`).

Exemple (API locale, champion v5, 2026-11-24) :

| Lot | Statuts | Labels | Latence p50 / p95 |
|---|---|---|---|
| `normal.csv --broken --limit 200` | 160 × 200, 40 × 422 | 142 ham, 18 spam | 16 / 32 ms |
| `campaign.csv --limit 100` | 100 × 200 | 2 ham, 98 spam | 16 / 47 ms |

Écarts à l'exercice : pas de `download_data.py` ni de jeton SAS (les lots sont construits à
partir des releases DVC, déjà configurées) ; pas de CSV « v3 » : la dérive vient du lot
`campaign.csv` et les erreurs du mode `--broken`. Les requêtes invalides donnent des **4xx**
(validation), pas des 5xx : l'alerte « taux d'erreur » (5xx) se déclenche autrement (voir la
section alertes, étape 7).

Sous Windows, utiliser `127.0.0.1` plutôt que `localhost` (qui essaie d'abord l'IPv6 et
ajoute ~2 s par requête quand uvicorn n'écoute qu'en IPv4).

## Lire les logs JSON avec `jq`

Le service écrit **une ligne JSON par événement** sur stdout (`app/logging_config.py`).

### Où trouver les logs

| Où tourne l'API | Commande |
|---|---|
| uvicorn en local | la sortie du terminal (`... 2>&1 \| tee api.log` pour la garder) |
| conteneur Docker | `docker logs <conteneur>` |
| stack `docker compose` (étape 6) | `docker compose logs api --no-log-prefix` |
| production Render | tableau de bord Render → service → onglet **Logs** (recherche texte : coller un `request_id`, ou `"level":"error"`) |

Installer `jq` : `winget install jqlang.jq` (Windows), `brew install jq` (macOS), `apt install jq`
(Linux ; déjà présent sur les runners GitHub Actions).

### Champs

| Champ | Présent sur | Contenu |
|---|---|---|
| `timestamp`, `level`, `logger`, `event` | toutes les lignes | heure UTC ISO, niveau, module, nom de l'événement |
| `request_id` | toutes les lignes | id de la requête (en-tête `X-Request-ID`), `null` hors requête (démarrage) |
| `model_name`, `model_version`, `alias`, `run_id`, `data_version` | `model_loaded` | version concrète chargée au démarrage |
| `method`, `path`, `status`, `duration_ms` | `request_completed` | une ligne par requête HTTP ; `path` = route (`/predict`), niveau `error` si statut ≥ 500 |
| `model_version`, `prediction`, `score`, `latency_ms`, `n_chars`, `n_words` | `prediction_made` | sortie du modèle ; **jamais le texte du SMS** |

### Requêtes utiles

`-R` + `fromjson?` lit chaque ligne comme du texte et ignore celles qui ne sont pas du JSON
(préfixe oublié, ligne coupée) au lieu d'arrêter `jq`. Remplacer `docker compose logs api
--no-log-prefix` par la source voulue (`docker logs <conteneur>`, `cat api.log`...).

**Erreurs seules** (5xx, échec de chargement du modèle) :

```bash
docker compose logs api --no-log-prefix | jq -R -c 'fromjson? | select(.level == "error")'
```

**Toutes les lignes d'une requête** (id affiché par la page web, renvoyé dans l'en-tête
`X-Request-ID`, ou envoyé par `scripts/replay.py`) — de `prediction_made` à `request_completed` :

```bash
docker compose logs api --no-log-prefix | jq -R -c 'fromjson? | select(.request_id == "<id>")'
```

**Nombre de prédictions par `model_version`** (utile après un rollback : quelle version a servi
combien de prédictions) :

```bash
docker compose logs api --no-log-prefix | jq -R -s -c '[split("\n")[] | fromjson? | select(.event == "prediction_made")] | group_by(.model_version) | map({model_version: .[0].model_version, predictions: length})'
```

Autres requêtes :

```bash
# prédictions par version et par label (spam / ham)
jq -R -s -c '[split("\n")[] | fromjson? | select(.event == "prediction_made")] | group_by([.model_version, .prediction]) | map({model_version: .[0].model_version, prediction: .[0].prediction, n: length})'
# requêtes lentes (> 500 ms)
jq -R -c 'fromjson? | select(.event == "request_completed" and .duration_ms > 500) | {timestamp, request_id, path, status, duration_ms}'
# nombre de 5xx par route
jq -R -s -c '[split("\n")[] | fromjson? | select(.event == "request_completed" and .status >= 500)] | group_by(.path) | map({path: .[0].path, errors: length})'
# chronologie des modèles chargés (redémarrage, CT, rollback)
jq -R -c 'fromjson? | select(.event == "model_loaded") | {timestamp, model_version}'
```

Ces filtres sont exécutés par `tests/test_log_queries.py` sur de vrais logs de l'application
(en CI, où `jq` est installé), pour que la documentation reste juste.

## Métriques Prometheus (`GET /metrics`)

Exposées par `app/metrics.py` (`prometheus_client`, format texte). Chaque instance de l'app a
son propre registre ; les compteurs repartent de 0 à chaque démarrage (déploiement, CT,
rollback, réveil de l'instance gratuite) : dans PromQL, toujours `rate()` / `increase()`, jamais
la valeur brute.

| Métrique | Type | Labels | Mis à jour |
|---|---|---|---|
| `http_requests_total` | counter | `method`, `path`, `status` | chaque requête (middleware) |
| `http_request_duration_seconds` | histogram | `path` | chaque requête (buckets par défaut, 5 ms → 10 s) |
| `predictions_total` | counter | `prediction` (`spam` / `ham`), `model_version` | chaque prédiction réussie |
| `model_info` | gauge = 1 | `model_version` | à chaque scrape, depuis le modèle chargé ; **absente tant qu'aucun modèle n'est chargé** |
| `feature_drift` | gauge | `feature` (les 5 features de `app/drift.py`) | déclarée ; valeurs ajoutées par le suivi du drift (étape 5) |

Règles de labels (faible cardinalité) :

- `path` est le **modèle de route** (`/predict`, `/model-info`), jamais l'URL brute (pas de
  query string) ; une route inconnue donne `<unmatched>` (pas un label par URL tapée) ;
- jamais de `request_id`, de texte de SMS ni de valeur d'entrée en label ;
- `model_version` est la **version concrète** chargée (ex. `5`), jamais l'alias `champion`.

`/metrics` répond même avant le chargement du modèle (Prometheus voit l'API « up ») ;
`/ready` reste la seule source de vérité pour « le modèle peut prédire ». Les scrapes de
`/metrics` sont comptés dans `http_requests_total` mais logués au niveau `debug` seulement
(une ligne toutes les 5 s sinon). L'endpoint est public : il n'expose rien de sensible
(compteurs, version du modèle, PSI).

Exemple réel (API locale, champion v5, après `replay.py --limit 100 --broken` puis le smoke test) :

```text
http_requests_total{method="POST",path="/predict",status="200"} 83.0
http_requests_total{method="POST",path="/predict",status="422"} 19.0
predictions_total{model_version="5",prediction="ham"} 65.0
predictions_total{model_version="5",prediction="spam"} 18.0
model_info{model_version="5"} 1.0
```

Requêtes PromQL utilisées par le dashboard (étape 6) :

```promql
sum(rate(http_requests_total{path="/predict"}[1m]))                                    # requêtes/s
100 * sum(rate(http_requests_total{path="/predict",status=~"5.."}[2m]))
    / sum(rate(http_requests_total{path="/predict"}[2m]))                              # % de 5xx
histogram_quantile(0.95, sum by (le) (rate(http_request_duration_seconds_bucket{path="/predict"}[5m])))  # latence p95
sum by (prediction) (rate(predictions_total[5m]))                                       # prédictions par classe
max by (feature) (feature_drift)                                                        # drift par feature
max by (model_version) (model_info)                                                     # version servie
```

**Smoke test** : `scripts/smoke_test.py` (CI `docker-smoke`, staging, déploiement, CT,
rollback) vérifie aussi que `/metrics` répond, que `model_info` porte la **version attendue**
et que `predictions_total` a compté les prédictions du test pour cette version.
