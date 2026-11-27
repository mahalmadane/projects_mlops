"""Command line interface: ``mlstrat new <name>``."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from mlstrat import __version__
from mlstrat.scaffold import create_project


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="mlstrat", description="Outils pour projets de machine learning.")
    parser.add_argument("--version", action="version", version=f"mlstrat {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    new = sub.add_parser("new", help="Créer la structure d'un nouveau projet ML")
    new.add_argument("name", help="Nom du projet (et du dossier créé)")
    new.add_argument("--path", default=".", help="Dossier dans lequel créer le projet (défaut : dossier courant)")

    args = parser.parse_args(argv)

    if args.command == "new":
        try:
            result = create_project(args.name, args.path)
        except ValueError as exc:
            print(f"Erreur : {exc}", file=sys.stderr)
            return 1
        print(f"Projet créé dans {result.root.resolve()}")
        print(f"  {len(result.created)} fichiers créés")
        if result.skipped:
            print(f"  {len(result.skipped)} fichiers existants conservés (non écrasés)")
        print(
            "\nPour commencer :\n"
            f"  cd {result.root}\n"
            "  python -m venv .venv && source .venv/bin/activate\n"
            "  pip install -r requirements.txt   # dépendances (ZenML, MLflow... : à ajuster avant)\n"
            "  pip install -e .                  # rend src/ importable, n'installe rien d'autre\n"
            "\nDétails dans README.md."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
