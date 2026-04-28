# Streamlit Demo
## Run
```bash
streamlit run demo/pages/metadata_generation.py
```

## Folder roles

### `demo/`

Everything related to the web demo.

```text
Streamlit pages
workflow wrappers
demo-specific pipeline helpers
demo-specific agent helpers
UI components
artifact saving
demo configuration
```

### `demo/pages/`

One UI page per workflow.

```text
metadata_generation.py
```

Each page handles the UI for one workflow.