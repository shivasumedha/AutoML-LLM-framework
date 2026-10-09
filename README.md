# AutoML LLM Framework

## Run locally on Windows

Open two PowerShell terminals in the project folder.

### Backend

```powershell
cd C:\Users\shiva\AutoML-LLM-Framework
venv\Scripts\python.exe -m pip install -r requirements.txt
if (!(Test-Path backend\.env)) { Copy-Item backend\.env.example backend\.env }
# Edit backend\.env and set GROQ_API_KEY to your key.
venv\Scripts\python.exe -m uvicorn backend.main:app --reload --reload-dir backend --port 8000
```

The Groq key stays on the backend. The chatbot uses Groq's `openai/gpt-oss-120b` model. These commands use the Python executable directly, which avoids stale `uvicorn.exe` launchers. If `venv\Scripts\python.exe` is missing or the venv still points at an old folder, recreate it once with `Remove-Item -Recurse -Force venv` followed by `python -m venv venv`.

### Frontend

```powershell
cd C:\Users\shiva\AutoML-LLM-Framework\frontend
npm install
npm run dev
```

Open the local URL Vite prints (normally `http://localhost:5173`). Upload a CSV on the Dataset page, then use AI Chat to ask questions, request charts, or suggest data cleanup. Chat edits ask for confirmation and save a version that can be restored by asking “undo that”.

The backend initializes a local SQLite database at `backend/automl.sqlite3`. It stores chat sessions and message history, plus each chat's uploaded and edited dataset versions and trained model. A new chat starts with a clean dataset, analysis, and model workspace; returning to an older chat restores its own saved workspace. Chat titles are generated from the first message. The database is local to this installation and is excluded from Git. The profile control is a local workspace label; this project does not currently include user accounts or authentication.

Training is a one-time step for the current dataset. After training, the Models page shows the saved model and comparison scores, and Prediction uses that same saved model when you navigate there. The chatbot can also train and activate a selected classifier (such as SVM) when you explicitly ask it to; it will not retrain just because you visit another section. A dataset edit invalidates the saved model because it was fit on the previous data; undoing the edit restores the dataset but you may need to train again for that restored version.

If the frontend cannot reach the backend, create `frontend/.env` with `VITE_API_URL=http://127.0.0.1:8000`, then restart Vite.

## Deploy the API on Render and the frontend on Vercel

The Render Blueprint in `render.yaml` deploys the FastAPI backend and mounts a persistent disk at `/var/data`. SQLite, uploads, model files, and dataset versions are written under `AUTOML_DATA_DIR`, so they survive service restarts. Render persistent disks require a paid web service; the Blueprint uses the `starter` plan. Review Render's current pricing before creating the service. The backend uses one worker because its active chat workspace is held in process memory.

1. Push this project to a GitHub repository.
2. In Vercel, import the repository. Set **Root Directory** to `frontend`, framework preset to **Vite**, build command to `npm run build`, and output directory to `dist`. Deploy once to reserve/get your production Vercel URL; the API won't work until the next step.
3. In Render, choose **New → Blueprint**, connect the same repository, and deploy `render.yaml`. When prompted, enter your Groq API key and the Vercel production origin (scheme and host only) for `GROQ_API_KEY` and `CORS_ORIGINS`, for example `https://your-app.vercel.app`.
4. Copy the Render service URL, for example `https://automl-api.onrender.com`. In Vercel **Project Settings → Environment Variables**, add `VITE_API_URL` with this URL (no trailing slash), then redeploy the frontend.
5. Open the Vercel site, upload a small CSV, and verify chat history and the dataset still exist after a Render restart. The backend health endpoint is `https://your-render-service.onrender.com/health`.

Use the exact Vercel production domain in `CORS_ORIGINS`; if you use a Vercel preview deployment, add that preview URL there too. Keep `GROQ_API_KEY` only in Render's environment settings, never in Vercel or a committed `.env` file. Vercel's `VITE_` variables are included in the browser build, so `VITE_API_URL` should contain only the public API URL, never a secret. Render Blueprint secret values declared with `sync: false` are entered through its dashboard.

This project currently has no account login or authentication. The deployed site and API are public, so avoid uploading private or sensitive datasets until authentication and access controls are added.

## Deploy with Docker Compose

The included deployment serves the built React app with Nginx and proxies `/api` requests to FastAPI. SQLite, uploaded datasets, dataset versions, and trained models are persisted together in the `automl-data` Docker volume mounted at `/data`.

1. Install Docker Engine with the Compose plugin (or Docker Desktop).
2. Copy `.env.example` to `.env` and set `GROQ_API_KEY` on the deployment host. Keep `.env` private.
3. From the project root, run:

   ```sh
   docker compose up --build -d
   docker compose logs -f
   ```

4. Open `http://127.0.0.1:8080` (or the port set by `APP_PORT`).

The web port is bound to localhost by default, and FastAPI is not published directly. For an internet-facing deployment, put a TLS reverse proxy with authentication in front of this app and proxy to `127.0.0.1:8080`; the app itself does not provide user login. On managed container platforms, mount a persistent disk at `/data` and configure the platform's ingress to route web traffic to the frontend container. Do not rely on a container's temporary filesystem for the SQLite database or uploaded data.

To stop the services while keeping data, run `docker compose down`. The named volume remains and is reused next time you start the stack.
