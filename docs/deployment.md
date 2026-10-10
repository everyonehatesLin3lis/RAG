# Deployment: Neon + Cloud Run + Firebase Hosting

How the Movie Research Copilot runs in the cloud. The developer chose this setup on 2026-10-10 instead of the
plan's Google Cloud SQL: a cloud database was required, not Cloud SQL specifically, and Neon gives managed
PostgreSQL 17 + pgvector on a free plan (see the README decision log).

```text
Browser ── HTTPS ──> Firebase Hosting       static Next.js export (frontend/out)
   │
   └──── HTTPS / SSE ──> Cloud Run          FastAPI container (backend/Dockerfile)
                            ├── MCP server  child process in the same container (stdio)
                            ├── Neon        PostgreSQL 17 + pgvector + HNSW, Frankfurt
                            └── OpenRouter  MiMo (answers), Gemini Flash Lite (translation), text-embedding-3-small
```

What runs in the cloud: only the chat path (translate, embed the question, hybrid search, answer, tools). The data
pipeline (download, ingest, embed) and the evaluation were run once locally; their result, the database, is copied
to Neon, so nothing is embedded again. Tests and evaluation stay local.

Commands are for Windows PowerShell, from the repo root unless a step says otherwise. Values in `<angle brackets>`
are yours to fill in. Never commit a connection string or key.

## What it costs

| Part | Plan | Expected cost |
|---|---|---|
| Neon | Free: 0.5 GB storage, 100 compute-hours a month, sleeps after 5 min idle | $0 |
| Cloud Run | Pay per request time, scales to zero; free tier covers a demo | ~$0 |
| Cloud Build + Artifact Registry | A few builds a month, one ~100 MB image | ~$0 (cents at most) |
| Firebase Hosting | Spark (free): 10 GB storage, 360 MB/day transfer | $0 |
| OpenRouter | ~$0.001 per question (evaluation runs) | what visitors ask |

These are the providers' published free tiers as of October 2026, not measured bills. The only cost that grows with
use is OpenRouter, which is why the API is rate limited and the key has its own credit limit (step 5).

## 1. Check the database fits Neon's free plan

```powershell
docker compose up -d
docker exec movie-copilot-db psql -U movie_copilot -d movie_copilot -c "SELECT pg_size_pretty(pg_database_size('movie_copilot'));"
```

It must be well under 0.5 GB. (10,540 chunks × 1,536 floats × 4 bytes ≈ 65 MB of vectors, plus the HNSW index, the
text and the full-text index.)

## 2. Create the Neon database

1. Sign up at neon.com, create a project `movie-copilot`: **PostgreSQL 17**, region **AWS Europe Central 1
   (Frankfurt)**, close to Lithuania and to Cloud Run's `europe-west1`.
2. Create a database `movie_copilot` (or rename the default one).
3. Copy the connection string with **connection pooling off** (the *direct* endpoint, host without `-pooler`).
   It looks like `postgresql://<user>:<password>@ep-....eu-central-1.aws.neon.tech/movie_copilot?sslmode=require&channel_binding=require`.

Why direct, not pooled: the pooler (PgBouncer) is for hundreds of short-lived connections. We have at most two Cloud
Run instances with a small SQLAlchemy pool each, and psycopg's prepared statements and `pg_restore` are simplest on a
direct connection.

The app needs the SQLAlchemy form, the same URL with `postgresql+psycopg://` at the start. Below, `<neon-url>` is the
form Neon gives (`postgresql://...`) and `<neon-sqlalchemy-url>` the `postgresql+psycopg://...` one.

## 3. Build the schema with the migrations, then copy the data

The schema comes from our Alembic migrations, as for the local database, so Neon's schema is the same one the
migrations describe. Then only the data is copied.

```powershell
cd backend
.venv\Scripts\activate
$env:DATABASE_URL = "<neon-sqlalchemy-url>"     # this terminal only; overrides backend/.env
alembic upgrade head                            # creates pgvector, the tables, HNSW + GIN indexes (0001–0004)
alembic check                                   # "No new upgrade operations detected."
cd ..
```

Export the data from the local container. Left out: `alembic_version` (the migrations wrote it already) and the
local test conversations (the public site starts with an empty history).

```powershell
docker exec movie-copilot-db pg_dump -U movie_copilot -d movie_copilot --data-only -Fc `
  --exclude-table-data=alembic_version --exclude-table-data=messages --exclude-table-data=conversations `
  -f /tmp/data.dump
```

Load it into Neon. `pg_restore` runs inside the local container, which has the PostgreSQL 17 tools, so nothing has
to be installed on Windows:

```powershell
docker exec movie-copilot-db pg_restore --data-only --no-owner --no-acl --single-transaction `
  -d "<neon-url>" /tmp/data.dump
docker exec movie-copilot-db psql "<neon-url>" -c "ANALYZE;"
```

`--single-transaction`: all or nothing, so a failure leaves Neon empty instead of half-loaded. The HNSW index is
built as rows arrive, so this takes a minute or two. `ANALYZE` gives the planner fresh statistics, so it picks the
HNSW index the way it does locally. The `search_vector` column is generated, so PostgreSQL computes it on load.

Check both databases hold the same rows:

```powershell
$check = "SELECT (SELECT count(*) FROM movies) AS movies, (SELECT count(*) FROM reviews) AS reviews, (SELECT count(*) FROM rag_chunks) AS chunks, (SELECT count(*) FROM rag_chunks WHERE embedding IS NOT NULL) AS embedded;"
docker exec movie-copilot-db psql -U movie_copilot -d movie_copilot -c $check
docker exec movie-copilot-db psql "<neon-url>" -c $check
```

## 4. Run the local backend against Neon

Put `<neon-sqlalchemy-url>` as `DATABASE_URL` in `backend/.env` (keep the local line commented out to switch back),
then start everything as usual:

```powershell
cd backend; uvicorn app.main:app --reload --port 8000      # terminal 1
cd frontend; npm run dev                                    # terminal 2
```

Ask on http://localhost:3000 and open the panels under each answer:

| Question | Shows that ... works |
|---|---|
| What do critics think about Black Swan? | vector + keyword search and fusion (RAG panel step 5), sources |
| Which is rated higher, Zodiac or Prisoners? | MCP tools on Neon ("via MCP server" in the Tool calls panel) |
| Tell me about Zodiac. → Who directed it? → reload the page | conversation history in Neon |

`pytest` against Neon also works (tests run inside rolled-back transactions), but local Docker stays the database
for development and tests.

When this works, the database move is done.

## 5. Protect the OpenRouter credit before going public

1. On openrouter.ai → Keys, create a **new key just for the deployment** with a **credit limit** (for example $5).
   When it is used up, OpenRouter refuses that key's calls (the page shows an error) instead of draining the account.
   This is the hard ceiling; the rate limits below only slow abuse down.
2. The backend refuses more than `RATE_LIMIT_PER_MINUTE` (10) questions per visitor IP per minute and
   `RATE_LIMIT_PER_DAY` (200) questions in total per day, with `RATE_LIMITED` (429) (`app/rate_limit.py`). Both
   are set in `deploy/cloudrun-env.yaml` and off locally.
3. Already in place: 2,000 characters per question, at most 3 tool rounds, 90 s per question, output-token caps on
   every model call.

## 6. Google Cloud project

Install the Google Cloud CLI, then:

```powershell
gcloud auth login
gcloud projects create <project-id> --name "Movie Research Copilot"   # or use an existing project
gcloud config set project <project-id>
# Link billing (Console → Billing) — Cloud Run needs it even inside the free tier.
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com
```

**Budget alerts** (Console → Billing → Budgets & alerts): one budget, e.g. €10, with alerts at 10 %, 50 % and
100 % (€1, €5, €10). They only email you; they do not stop anything.

## 7. Secrets

The two secrets go into Secret Manager, never into the image, the repo, `deploy/cloudrun-env.yaml` or the
frontend. Cloud Run hands them to the container as environment variables when it starts; the deploy command only
names them (`openrouter-api-key:latest`), so it contains nothing secret.

Create them in the browser, so the values are never typed into a terminal (PowerShell saves every command line in
its history file, `(Get-PSReadLineOption).HistorySavePath`, and a key in a command would stay there):

1. Console → Security → **Secret Manager** → **Create secret**.
2. Name `openrouter-api-key`, secret value: the deployment key from step 5. Create.
3. Again: name `database-url`, value `<neon-sqlalchemy-url>`. Create.

Check the names (this shows names, not values):

```powershell
gcloud secrets list
```

A service account of its own for the backend, allowed to read exactly these two secrets and nothing else:

```powershell
gcloud iam service-accounts create movie-copilot-api --display-name "Movie Copilot API"
$sa = "movie-copilot-api@<project-id>.iam.gserviceaccount.com"
gcloud secrets add-iam-policy-binding openrouter-api-key --member "serviceAccount:$sa" --role roles/secretmanager.secretAccessor
gcloud secrets add-iam-policy-binding database-url --member "serviceAccount:$sa" --role roles/secretmanager.secretAccessor
```

`EMBEDDING_API_KEY` is not needed: it falls back to `OPENROUTER_API_KEY`.

## 8. (Optional) Try the container locally

With Docker Desktop, from `backend/`:

```powershell
docker build -t movie-copilot-api .
docker run --rm -p 8080:8080 --env-file .env -e REQUEST_LOG_PATH=stdout movie-copilot-api
# another terminal:
curl.exe http://localhost:8080/api/health
```

Tested in development (2026-10-10, against an empty pgvector database): the image is ~100 MB compressed, migrations
run inside it, the MCP server starts as a child process and answers `tools/list` and `tools/call`, idle memory is
~210 MB, rate limiting and stdout logging work.

## 9. Deploy the backend to Cloud Run

Edit `deploy/cloudrun-env.yaml`: replace `<project-id>` in `CORS_ORIGINS` (the Firebase site will be
`https://<project-id>.web.app`). Then, from the repo root:

```powershell
gcloud run deploy movie-copilot-api `
  --source backend --region europe-west1 `
  --service-account movie-copilot-api@<project-id>.iam.gserviceaccount.com `
  --allow-unauthenticated `
  --cpu 1 --memory 512Mi --min-instances 0 --max-instances 2 --timeout 300 `
  --env-vars-file deploy/cloudrun-env.yaml `
  --set-secrets "OPENROUTER_API_KEY=openrouter-api-key:latest,DATABASE_URL=database-url:latest"
```

`--source backend` uploads `backend/` (minus `.dockerignore`), Cloud Build builds `backend/Dockerfile`, the image
goes to Artifact Registry and Cloud Run runs it: the plan's Cloud Build → Artifact Registry → Cloud Run in one
command. The first run asks to create the Artifact Registry repository; say yes.

Why these flags:
- `europe-west1` (Belgium): close to Neon in Frankfurt (each question makes several database round trips) and in
  Cloud Run's cheapest price tier.
- `--allow-unauthenticated`: the website calls the API straight from visitors' browsers, so it must be public. Abuse
  is limited by the rate limits and the OpenRouter key's credit limit.
- `--min-instances 0`: nothing runs, and nothing is billed, while nobody asks. The price is a cold start of a few
  seconds on the first question after a quiet spell (container start + MCP server start + Neon waking up).
- `--max-instances 2`: caps cost and database connections.
- `--memory 512Mi`: idle use was ~210 MB for both processes (API + MCP server). If Cloud Logging reports
  "Memory limit exceeded", redeploy with `--memory 1Gi`.
- `--timeout 300`: longer than `CHAT_TIMEOUT_S` (90 s), so our own time limit answers first, with a clear error.

Cloud Run prints the service URL, `https://movie-copilot-api-<hash>.europe-west1.run.app`. Test it:

```powershell
$api = "<service-url>"
curl.exe "$api/api/health"
Invoke-RestMethod -Method Post -Uri "$api/api/chat" -ContentType "application/json" `
  -Body '{"message": "Which is rated higher, Zodiac or Prisoners?"}' | ConvertTo-Json -Depth 6
# Streaming: curl.exe -N prints events as they arrive. The body comes from a file, which avoids
# PowerShell's quoting rules for JSON on the command line.
Set-Content -Path body.json -Value '{"message": "What do critics think about Black Swan?"}'
curl.exe -N -X POST "$api/api/chat/stream" -H "Content-Type: application/json" --data "@body.json"
Remove-Item body.json
```

The stream must show `status`, `token`, `sources`, `metadata` and `done` events arriving one by one, not all at
the end. Logs: Console → Cloud Run → movie-copilot-api → Logs; each question is one JSON line ("chat success").

To change a setting later without rebuilding: edit the YAML and run
`gcloud run services update movie-copilot-api --region europe-west1 --env-vars-file deploy/cloudrun-env.yaml`.
After a code change: the same `gcloud run deploy` command again.

## 10. Deploy the frontend to Firebase Hosting

The page is exported as static files (`output: "export"` in `frontend/next.config.ts`): it needs no server of its
own, it runs in the browser and calls the API. `NEXT_PUBLIC_API_BASE_URL` is built into the files, so set it
before building. It is only the API address, not a secret; no key ever goes into the frontend.

```powershell
npm install -g firebase-tools
firebase login
firebase projects:addfirebase <project-id>      # adds Firebase to the same Google Cloud project

cd frontend
Set-Content -Path .env.production -Value "NEXT_PUBLIC_API_BASE_URL=<service-url>"
npm run build                                   # writes out/
firebase deploy --only hosting --project <project-id>
```

`frontend/firebase.json` already says what to serve (`out/`). The site is `https://<project-id>.web.app` (the
deploy prints the address; if Firebase gave the site another name, put that one in `CORS_ORIGINS` and update the
service as in step 9).
`.env.production` is git-ignored like every `.env.*`; it only matters on the machine that builds.

## 11. Test the live site

On `https://<project-id>.web.app`:

- **RAG**: "What do reviewers think about Black Swan?" streams an answer with sources; the RAG panel shows
  translation, vector + keyword results, fusion and the 8 selected chunks.
- **MCP**: "Which is rated higher, Zodiac or Prisoners?" shows a `compare_movies` call "via MCP server".
- **Memory**: "Tell me about Zodiac." → "Who directed it?" → reload → "What else did critics say about it?".
- **Rate limit**: 11 quick questions → the 11th shows `RATE_LIMITED` with a "Try again".
- **Errors**: the page shows a clear error, never a blank one. A safe way to check on the live service is a wrong key
  for one revision (`gcloud secrets versions add openrouter-api-key --data-file=...` with a bad key, redeploy,
  expect `LLM_AUTH_FAILED`, then add the real key back). The full error matrix is tested locally (`pytest`,
  `tests/test_errors.py`).
- **CORS**: the browser console shows no CORS errors. If it does, the site address is missing from `CORS_ORIGINS`.

## Not deployed

Tests, `evaluation/`, the dataset download, ingestion and embedding scripts, and the embedding comparison stay in
the repository and run locally. The container has only `app/`, `migrations/` and `alembic.ini`, and only the
runtime dependencies (`backend/requirements-runtime.txt`; no `datasets`, no `pytest`).

## Later

- Automatic deployment from GitHub (Cloud Build trigger for `backend/`, Firebase GitHub Action for `frontend/`),
  once the manual deployment above works.
- Moving the data again: repeat step 3 against a fresh Neon database.
