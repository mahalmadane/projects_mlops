"""Command line interface: ``mlstrat new <name>`` and ``mlstrat docs``."""

from __future__ import annotations

import argparse
import sys
import webbrowser
from importlib.resources import files
from pathlib import Path
from typing import List, Optional

from mlstrat import __version__
from mlstrat.scaffold import create_project


def docs_path() -> Path:
    """Path of the guide page shipped inside the package."""
    return Path(str(files("mlstrat") / "docs" / "index.html"))


def _docs(print_only: bool) -> int:
    path = docs_path()
    if not path.exists():
        print(f"Erreur : page introuvable ({path})", file=sys.stderr)
        return 1
    print(f"Guide mlstrat : {path.as_uri()}")
    if not print_only and not webbrowser.open(path.as_uri()):
        print("Impossible d'ouvrir le navigateur : copie le lien ci-dessus dans ton navigateur.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mlstrat",
        description="Crée des projets de machine learning et fournit des outils d'analyse.",
        epilog="Guide complet : mlstrat docs",
    )
    parser.add_argument("--version", action="version", version=f"mlstrat {__version__}")
    sub = parser.add_subparsers(dest="command")

    new = sub.add_parser("new", help="Créer la structure d'un nouveau projet ML")
    new.add_argument("name", help="Nom du projet (et du dossier créé)")
    new.add_argument("--path", default=".", help="Dossier dans lequel créer le projet (défaut : dossier courant)")

    docs = sub.add_parser("docs", help="Ouvrir le guide (page web incluse, fonctionne hors ligne)")
    docs.add_argument("--print", dest="print_only", action="store_true", help="Afficher le lien sans ouvrir le navigateur")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "docs":
        return _docs(args.print_only)

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
            "\nDétails dans README.md, guide complet : mlstrat docs"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
