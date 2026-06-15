# Streamlit Demo

## Run

```bash
uv run --group demo streamlit run demo_app.py
```

or

```bash
make demo
```

## Deploy as a web server with Docker Compose

This repository includes a Docker Compose setup for running the Streamlit demo
behind Nginx:

- `metadata-demo`: builds the app image from `Dockerfile` and runs Streamlit on
  port `8501` inside the Compose network.
- `nginx`: exposes port `80` on the host and proxies browser traffic to the
  Streamlit container.

### 1. Prepare the server

Install Docker Engine and the Compose plugin on the server, then clone the
repository:

```bash
git clone <repository-url>
cd metadata_agent
```

The server must allow inbound HTTP traffic on port `80`. If you plan to put TLS
in front of this app, terminate HTTPS in your outer reverse proxy or load
balancer and forward plain HTTP to this Compose stack.

### 2. Create `.env`

Docker Compose loads `.env` for both Compose variables and application
configuration. Create it in the repository root:

```bash
LLM_PROVIDER=surf
SURF_API_KEY=your_surf_api_key
SURF_API_BASE=your_surf_api_url
LLM_MODEL=your_llm_model
DEFAULT_TOPOLOGY=single
SERVER_NAME=metadata.example.org
```

Provider choices:

```bash
# Google
LLM_PROVIDER=google
GOOGLE_API_KEY=your_google_api_key

# OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key

# SURF/OpenAI-compatible endpoint
LLM_PROVIDER=surf
SURF_API_KEY=your_surf_api_key
SURF_API_BASE=your_surf_api_url
```

`DEFAULT_TOPOLOGY=single` is a good web-demo default because it avoids the extra
multi-player debate calls used by heavier topologies.

Set `SERVER_NAME` to your domain. For local or IP-only testing, omit it or set:

```bash
SERVER_NAME=_
```

### 3. Build and start the service

```bash
docker compose up -d --build
```

Open the app:

```text
http://metadata.example.org/
```

For local testing on the server:

```bash
curl -I http://localhost/
```

### 4. Watch logs

Use the repository helper:

```bash
make docker-logs
```

or plain Compose:

```bash
docker compose logs -f metadata-demo
docker compose logs -f nginx
```

The Streamlit container should show `streamlit run demo_app.py`. Nginx should
proxy requests to `http://metadata-demo:8501`.

### 5. Update the deployment

Pull the latest code, rebuild the image, and restart:

```bash
git pull
docker compose up -d --build
```

To stop the service:

```bash
docker compose down
```

### 6. Troubleshooting

If the page does not load, check:

```bash
docker compose ps
docker compose logs metadata-demo
docker compose logs nginx
```

Common causes:

- Missing API key in `.env`: the app starts, but metadata generation fails when
  the selected provider is called.
- Port `80` already in use: stop the existing service or change the published
  port in `docker-compose.yml`.
- Domain not routed to the server: test with:
  ```bash
  curl -H 'Host: metadata.example.org' http://SERVER_IP/
  ```
- Long metadata generation: keep `DEFAULT_TOPOLOGY=single` for the demo, then
  use `fast`, `default`, or `thorough` only when you explicitly want more LLM
  calls.

## Folder roles

### `demo_app.py`

Streamlit entry point that configures and launches the demo app.

### `demo/`

Everything related to the web demo.

```text
Streamlit pages
workflow wrappers
UI components
```

### `demo/workflows`
Main workflows used for the apps. 
```text
metadata_generation.py
```

### `demo/pages/`

One UI page per workflow.

```text
metadata_generation.py
```

Each page handles the UI for one workflow.
