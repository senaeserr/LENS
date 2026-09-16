from pathlib import Path
import runpy
import streamlit as st

from lens_core import ensure_session_defaults

st.set_page_config(
    page_title="LENS",
    page_icon="🔎",
    layout="wide",
)

# Initialize defaults for this specific Streamlit session on every rerun.
ensure_session_defaults()

# IMPORTANT:
# Streamlit reruns app.py after every interaction. A plain `import lens_ui`
# executes lens_ui only on the first Python import because modules are cached,
# which can leave the page blank after upload + st.rerun().
# run_path executes the UI entry script on every Streamlit rerun while the
# definition-only core/dashboard/findings modules remain normally importable.
ui_path = Path(__file__).with_name("lens_ui.py")
runpy.run_path(str(ui_path), run_name="__lens_ui__")
