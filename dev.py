"""A simple development script to run common tasks. 

Available: test, bump, docs, lint, setup.
Usage: `python dev.py <function_name> [args...]`
"""
import sys
import subprocess


def test(*args):
    subprocess.run(["pytest", "--cov=romtools", "--cov-report", "html:htmlcov", "tests"] + list(args))


def bump(*args):
    subprocess.run(["cz", "bump"] + list(args))


def docs(*args):
    subprocess.run(["mkdocs", "serve"] + list(args))


def lint(*args):
    subprocess.run(["ruff", "check", "src", "tests"] + list(args))


def setup(*args):
    subprocess.run(["uv", "sync", "--group", "dev"])
    subprocess.run(["pre-commit", "install", "--allow-missing-config"])
    


FUNCTIONS = {
    "test": test, 
    "bump": bump,
    "docs": docs,
    "lint": lint,
    "setup": setup
}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dev.py <function_name> [args...]")
        sys.exit(1)
    
    func_name = sys.argv[1]
    args = sys.argv[2:]
    
    if func_name not in FUNCTIONS:
        print(f"Error: Function '{func_name}' not found")
        sys.exit(1)
    
    func = FUNCTIONS[func_name]
    func(*args)