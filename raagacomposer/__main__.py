"""Entry point: ``python -m raagacomposer``."""
from __future__ import annotations

import sys


def main() -> int:
    from .core.logging_setup import setup_logging
    from .core.settings import Settings

    settings = Settings.load()
    setup_logging(settings.log_level)
    from .ui.main_window import run
    return run()


if __name__ == "__main__":
    sys.exit(main())
