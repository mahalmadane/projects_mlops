#!/usr/bin/env python3
"""Lance le serveur MLflow, l'API FastAPI et l'interface Streamlit depuis la racine du projet.

Usage:
    python3 run_all.py
    python3 run_all.py --zenml            # lance aussi le dashboard ZenML local
    python3 run_all.py --no-streamlit      # ne lance pas l'interface Streamlit
"""
import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_BIN = ROOT / "env" / "bin"
STREAMLIT_BIN = ROOT / "env_streamlit" / "bin" / "streamlit"

MLFLOW_URL = "http://127.0.0.1:5001"
API_URL = "http://127.0.0.1:8000"

processes = []  # liste de (nom, Popen), dans l'ordre de démarrage


def wait_for_http(url: str, timeout: float = 60.0, interval: float = 1.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(interval)
    return False


def start(name: str, cmd: list) -> subprocess.Popen:
    print(f"[run_all] Démarrage: {name} -> {' '.join(str(c) for c in cmd)}")
    proc = subprocess.Popen(cmd, cwd=ROOT)
    processes.append((name, proc))
    return proc


def stop_all() -> None:
    for name, proc in reversed(processes):
        if proc.poll() is None:
            print(f"[run_all] Arrêt: {name}")
            proc.terminate()
    for name, proc in reversed(processes):
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print(f"[run_all] {name} ne répond pas, arrêt forcé")
            proc.kill()


def check_prereqs(need_streamlit: bool) -> None:
    required = [ENV_BIN / "mlflow", ENV_BIN / "uvicorn"]
    for path in required:
        if not path.exists():
            sys.exit(
                f"Introuvable: {path}. Crée et installe le venv `env/` d'abord "
                "(voir COMMANDS.md)."
            )
    if need_streamlit and not STREAMLIT_BIN.exists():
        sys.exit(
            f"Introuvable: {STREAMLIT_BIN}. Crée le venv Streamlit "
            "(voir COMMANDS.md, étape 9)."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-mlflow", action="store_true", help="Ne pas lancer le serveur MLflow")
    parser.add_argument("--no-api", action="store_true", help="Ne pas lancer l'API FastAPI")
    parser.add_argument("--no-streamlit", action="store_true", help="Ne pas lancer l'interface Streamlit")
    parser.add_argument("--zenml", action="store_true", help="Lancer aussi le dashboard ZenML local")
    args = parser.parse_args()

    check_prereqs(need_streamlit=not args.no_streamlit)

    try:
        if args.zenml:
            zenml_env = os.environ.copy()
            zenml_env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
            subprocess.run([str(ENV_BIN / "zenml"), "login", "--local"], cwd=ROOT, env=zenml_env)

        if not args.no_mlflow:
            start("mlflow", [str(ENV_BIN / "mlflow"), "server", "--host", "0.0.0.0", "--port", "5001"])
            print("[run_all] Attente du serveur MLflow...")
            if not wait_for_http(MLFLOW_URL):
                sys.exit("Le serveur MLflow n'a pas démarré à temps.")
            print(f"[run_all] MLflow prêt: {MLFLOW_URL}")

        if not args.no_api:
            start("api", [str(ENV_BIN / "uvicorn"), "api.app:app", "--host", "127.0.0.1", "--port", "8000"])
            print("[run_all] Attente de l'API...")
            if not wait_for_http(API_URL):
                sys.exit(
                    "L'API n'a pas démarré à temps (vérifie qu'un modèle a l'alias "
                    "'champion' dans le registre MLflow)."
                )
            print(f"[run_all] API prête: {API_URL}")

        if not args.no_streamlit:
            start("streamlit", [str(STREAMLIT_BIN), "run", "streamlit_app/app.py"])

        print("\n[run_all] Tout est lancé. Ctrl+C pour tout arrêter.\n")
        while True:
            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"[run_all] {name} s'est arrêté (code {proc.returncode}).")
                    sys.exit(proc.returncode)
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[run_all] Interruption reçue, arrêt en cours...")
    finally:
        stop_all()


if __name__ == "__main__":
    main()

# export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=true
# projects_mlops % zenMl login --local 