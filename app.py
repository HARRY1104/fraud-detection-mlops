"""Root entrypoint for Streamlit dashboard on local and Streamlit Community Cloud.

Allows running either:
    streamlit run app.py
or
    streamlit run src/app/main.py
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.app.main import main  # noqa: E402

if __name__ == "__main__":
    main()