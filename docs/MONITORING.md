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

## Drift des entrées en production (`feature_drift`)

Même mesure que le CT (`app/drift.py` : 5 features de texte + score, PSI, seuil **0,20**),
appliquée au trafic réel par `app/drift_monitor.py` :

1. **Au chargement du modèle**, l'app télécharge `monitoring/training-profile.json` depuis le
   run d'entraînement du champion (le `run_id` de la version servie ; ajouté a posteriori à la
   v5 par `ml.backfill_profile`). Log `drift_profile_loaded`, ou `drift_profile_missing`
   (warning) si le run n'a pas de profil : l'app reste prête, le drift n'est simplement pas
   calculé (c'est un signal, pas une dépendance).
2. **À chaque prédiction**, les 5 features du message (pas le texte) entrent dans une
   `deque(maxlen=500)` : mémoire bornée.
3. **Toutes les 100 prédictions**, une fois la fenêtre pleine, le PSI de chaque feature est
   recalculé avec les bins du profil → jauge `feature_drift{feature}` et log `drift_computed`
   (niveau `warning` si une feature dépasse 0,20). Attendre 500 messages évite les fausses
   alertes : sur 100 messages, le hasard suffit à dépasser 0,20.

Réglages (variables d'environnement) : `DRIFT_WINDOW_SIZE` (500), `DRIFT_RECOMPUTE_EVERY` (100).

Exemple réel (API locale, champion v5, profil v2) :

| Après | PSI max | Features ≥ 0,20 |
|---|---|---|
| 600 SMS de `normal.csv` | 0,016 | aucune |
| + 100 SMS de `campaign.csv` | 0,23 | `spam_probability` |
| + 200 | 0,68 | les 5 |
| + 600 (fenêtre entièrement « campagne ») | 6,9 | les 5 |

La montée est progressive parce que la fenêtre glissante se remplit petit à petit de messages
de campagne ; elle redescend de la même façon quand le trafic redevient normal.

**Un drift déclenche une enquête, pas un réentraînement** : regarder quelles features bougent
(campagne de spam ? nouveau type de messages ? bug du client ?), comparer avec le `drift.json`
du CT (mêmes valeurs, même seuil), puis décider de publier une nouvelle release de données et
de lancer le CT à la main.

Le compteur repart à zéro à chaque démarrage (nouveau déploiement, CT, rollback, réveil de
l'instance gratuite) : la jauge n'apparaît qu'après 500 nouvelles prédictions.

## Stack locale Prometheus + Grafana (`docker compose`)

```bash
cp .env.example .env            # puis remplir MLFLOW_* (et GF_SMTP_* pour les alertes)
docker compose up -d --build    # api :8000, prometheus :9090, grafana :3000
python scripts/replay.py --file .exercise-data/normal.csv --rate 10 --shuffle
```

| Service | Image | Rôle |
|---|---|---|
| `api` | construite depuis le `Dockerfile` de production | même image qu'en prod, champion chargé depuis DagsHub |
| `prometheus` | `prom/prometheus:v3.5.0` | scrape `api:8000/metrics` toutes les **5 s** (`monitoring/prometheus.yml`, job `api`) |
| `grafana` | `grafana/grafana:12.1.0` | datasource et dashboard **provisionnés** au démarrage (aucun clic) ; SMTP par variables d'environnement |

- Prometheus : <http://localhost:9090/targets> → la cible `api` doit être **UP**.
- Grafana : <http://localhost:3000> (identifiants `GRAFANA_ADMIN_USER` / `GRAFANA_ADMIN_PASSWORD`
  du `.env`, `admin` / `admin` par défaut en local) → dossier *SMS spam* → **SMS spam API**.
- Arrêt : `docker compose down` (ajouter `-v` pour effacer aussi l'historique Grafana).

Fichiers : `docker-compose.yml`, `monitoring/prometheus.yml`, `monitoring/prometheus.prod.yml`,
`monitoring/grafana-datasource.yml`, `monitoring/grafana-dashboards.yml` (provider) et
`monitoring/dashboard.json` (dashboard exporté, versionné, sans secret). La CI vérifie leur
syntaxe (`promtool check config`, `docker compose config`) et `tests/test_monitoring_config.py`
leur contenu (images épinglées, credentials uniquement par variables, 6 panneaux, seuil 0,20…).

### Dashboard « SMS spam API » (6 panneaux)

| Panneau | Requête | Lecture |
|---|---|---|
| Requests per second (/predict) | `sum by (status) (rate(http_requests_total{job="api",path="/predict"}[$__rate_interval]))` | trafic par statut (200 / 422 / 503) |
| 5xx error rate (/predict) | `100 * sum(rate(...{status=~"5.."}...)) / sum(rate(...))` | % d'erreurs serveur, ligne rouge à 5 % (seuil de l'alerte) |
| Latency p95 (/predict) | `histogram_quantile(0.95, sum by (le) (rate(http_request_duration_seconds_bucket{path="/predict"}[...])))` | latence p95 en secondes |
| Predictions by class and model version | `sum by (prediction, model_version) (rate(predictions_total[...]))` | spam / ham **par version** : montre quelle version a servi chaque prédiction |
| Input drift by feature (PSI) | `max by (feature) (feature_drift)` | une courbe par feature, **ligne rouge à 0,20** |
| Served model version | `max by (model_version) (model_info)` | version concrète chargée (ex. `v5`) |

Les panneaux de requêtes sont filtrés sur `path="/predict"` (pas les sondes `/ready`, ni les
scrapes `/metrics`). Les compteurs repartent de 0 à chaque redémarrage : tous les panneaux
utilisent `rate()`.

Vérification réelle (stack locale, champion v5, 2026-11-24) : cible `api` UP ; 600 requêtes
`normal.csv --broken` → ~8,6 req/s en 200 et ~2 req/s en 422, p95 ≈ 18 ms, version `v5` ;
`feature_drift` max 0,025 après du trafic normal, puis 2,1 à 6,9 sur les 5 features après
600 SMS de `campaign.csv`.

Captures du dashboard : à ajouter dans `docs/img/` lors de la démonstration (trafic normal,
puis campagne de spam, puis rollback).

### Surveiller la production (option, pour la démo)

```bash
PROMETHEUS_CONFIG=prometheus.prod.yml docker compose up -d prometheus grafana
```

`monitoring/prometheus.prod.yml` scrape `https://final-project-ml-in-prod-sms-spam.onrender.com/metrics`
toutes les 30 s sous le même job `api` : le dashboard et les alertes fonctionnent sans
changement. Chaque scrape est une requête : l'instance gratuite ne s'endort plus tant que la
stack tourne, donc on l'arrête après la démo (`docker compose down`). En dehors de ça, la
production expose les mêmes logs (onglet *Logs* de Render) et le même `/metrics`.

### Observer un rollback

1. Trafic continu : `python scripts/replay.py --file .exercise-data/normal.csv --rate 5`.
2. Lancer le rollback (Actions → rollback, ou `python -m ml.rollback --version 1`) : l'alias
   `champion` passe à la version 1.
3. L'API résout l'alias **au démarrage** : en prod le workflow redéploie ; en local,
   `docker compose restart api`.
4. Dashboard : « Served model version » passe de `v5` à `v1`, « Predictions by class and model
   version » montre de nouvelles séries `(v1)` ; logs : `model_loaded` avec `model_version: "1"`
   puis des `prediction_made` en version 1 (`jq` : prédictions par `model_version`).
5. Retour au champion : rollback vers 5, puis `docker compose restart api` → `v5` à nouveau.
