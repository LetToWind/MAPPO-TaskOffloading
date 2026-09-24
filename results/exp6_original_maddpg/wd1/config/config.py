from pathlib import Path
from dynaconf import Dynaconf

_CONFIG_DIR = Path(__file__).resolve().parent

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=[str(_CONFIG_DIR / 'settings.py'), str(_CONFIG_DIR / '.secrets.toml')],
)

