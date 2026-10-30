"""Shared start-up helpers for the command line tools."""
import os
import sys


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def reexec_into_venv(module_file):
    """Re-run under the project .venv if it exists and we are not already in it.

    Lets `python idol_tools/generate.py ...` work straight after `scripts/setup.sh` without activating the
    environment. Also keeps ~/.local/lib out of the way: a stale user-site package there can shadow the venv's
    and break diffusers' peft loader."""
    venv_python = os.path.join(repo_root(), '.venv', 'bin', 'python')
    if os.path.isfile(venv_python) and os.path.realpath(sys.executable) != os.path.realpath(venv_python):
        os.environ.setdefault('PYTHONNOUSERSITE', '1')
        os.execv(venv_python, [venv_python, os.path.abspath(module_file)] + sys.argv[1:])
