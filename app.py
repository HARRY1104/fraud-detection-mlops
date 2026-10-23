"""Root entrypoint for Streamlit dashboard.

Allows running either:
    streamlit run app.py
or
    streamlit run src/app/main.py
"""

from pathlib import Path
import runpy

app_path = Path(__file__).resolve().parent / "src" / "app" / "main.py"
runpy.run_path(str(app_path), run_name="__main__")
