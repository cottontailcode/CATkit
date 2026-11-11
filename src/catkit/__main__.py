"""Allow running CATkit as a module: ``python -m catkit``."""

from catkit.cli import main

raise SystemExit(main())
