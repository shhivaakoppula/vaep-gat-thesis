# ==============================================================================
# common.py  -  paths and small helpers shared by every script in this folder
# ==============================================================================
# Every script used to carry its own copy of the same path block, the same
# header printer and the same "is the file there" check. They now live here once.
# This file only imports the standard library on purpose: the torch scripts must
# import torch BEFORE numpy/pandas on Windows (DLL clash), so nothing in here may
# pull numpy in behind their back.
# ==============================================================================

import os  # needed for environment variables
import sys  # needed for sys.exit and the interpreter path
import traceback  # prints the full error trace when a script fails
from pathlib import Path  # path objects work the same on Windows and Linux

# Same workaround as the originals: two copies of the OpenMP runtime can end up loaded on Windows.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # tells OpenMP to carry on instead of aborting

# The original scripts hard-coded one Windows folder. I keep it as the default so nothing changes on
# your laptop, but you can point to another folder by setting FOOTBALL_PROJECT_ROOT before running.
PROJECT_ROOT = Path(  # the folder that contains data/, output/ and Prototype/
    os.environ.get(  # look for the override first
        "FOOTBALL_PROJECT_ROOT",  # name of the environment variable
        r"C:\Users\Shiva Goud\Desktop\Football data explorer",  # your original path
    )
)
DATA_DIR = PROJECT_ROOT / "data"  # raw StatsBomb json files live here
OUTPUT_DIR = PROJECT_ROOT / "output"  # every CSV, checkpoint and chart goes here
PROTOTYPE_DIR = PROJECT_ROOT / "Prototype"  # the 34-match prototype area (steps 00-15)

# The originals refused to run unless the interpreter path contained ".vaep-venv". That check can only
# pass on one machine, so it is now opt-in: set REQUIRE_ENV_NAME=.vaep-venv to get the old behaviour.
EXPECTED_ENV_NAME = os.environ.get("REQUIRE_ENV_NAME", "")  # empty string means "do not check"


def verify_environment():
    """Stop with a clear message if the wrong Python is running (only when REQUIRE_ENV_NAME is set)."""
    if not EXPECTED_ENV_NAME:  # no environment name requested
        return  # nothing to check, carry on
    python_executable = Path(sys.executable).resolve()  # full path of the running interpreter
    if EXPECTED_ENV_NAME.lower() not in str(python_executable).lower():  # name missing from the path
        print(f"WRONG PYTHON ENVIRONMENT: need '{EXPECTED_ENV_NAME}', running {python_executable}")  # say why
        sys.exit(1)  # stop before any work is done
    print(f"[PASS] Python environment: {python_executable}")  # confirm the environment is right


def print_header(title):
    """Print a title between two lines of '=' so the console log is easy to scan."""
    print()  # blank line before the block
    print("=" * 80)  # top rule
    print(title)  # the title itself
    print("=" * 80)  # bottom rule


def require_file(path, description):
    """Raise a readable error if a file the script depends on does not exist."""
    path = Path(path)  # accept strings as well as Path objects
    if not path.exists():  # the file is missing
        raise FileNotFoundError(f"{description} does not exist:\n{path}")  # stop with the reason
    print(f"[PASS] {description}: {path}")  # tell the user the file was found


def patch_numpy_legacy_types():
    """socceraction 1.x still asks numpy for names like np.long that numpy 2.x removed.
    This hands back plain int for them. Call it BEFORE importing socceraction."""
    import numpy as np  # imported here so scripts that do not need it never pay for it

    original_getattr = getattr(np, "__getattr__", None)  # numpy's own fallback, if it has one

    def legacy_type_fallback(name):  # called only when numpy cannot find an attribute
        if name in ("long", "ulong", "clong", "longdouble"):  # the removed legacy names
            return int  # plain int is what the old code expected
        if original_getattr:  # numpy has its own handler
            return original_getattr(name)  # let it deal with everything else
        raise AttributeError(f"module 'numpy' has no attribute '{name}'")  # genuinely unknown name

    np.__getattr__ = legacy_type_fallback  # install the fallback on the numpy module


def run_main(main, label):
    """Run main(), print a readable error block on failure and exit with its code."""
    try:  # protect the whole script
        exit_code = main()  # run the script body
    except Exception as exc:  # anything that went wrong
        print()  # blank line
        print("=" * 80)  # top rule
        print(f"{label} ERROR")  # which step failed
        print("=" * 80)  # bottom rule
        print(f"Error type: {type(exc).__name__}")  # exception class
        print(f"Error message: {exc}")  # exception text
        traceback.print_exc()  # full trace for debugging
        exit_code = 1  # non-zero tells the shell something failed
    sys.exit(exit_code if exit_code is not None else 0)  # leave with the right code
