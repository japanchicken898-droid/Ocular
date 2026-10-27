"""Ocular: Offline Visual Compiler.

Moves multimodal vision inference from runtime to compile time.
"""

from pathlib import Path
try:
    from dotenv import load_dotenv
    _env_file = Path(__file__).resolve().parent.parent / ".env"
    if _env_file.exists():
        load_dotenv(dotenv_path=_env_file)
    else:
        load_dotenv()
except ImportError:
    pass

__version__ = "0.1.0"
