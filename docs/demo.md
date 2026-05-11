# Streamlit Demo
## Run
```bash
uv run --group demo streamlit run demo_app.py
```
or 
```bash
make demo
```

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
