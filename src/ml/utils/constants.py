"""
Define your constants here.

Example:
import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = os.path.join(ROOT_DIR, "data")
EXPERIMENTS_DIR = os.path.join(ROOT_DIR, ".experiments")
"""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.parent
print(ROOT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, ".data")
EXPERIMENTS_DIR = os.path.join(ROOT_DIR, ".experiments")
LOGGING_DIR = os.path.join(EXPERIMENTS_DIR, "logs")

print(ROOT_DIR, DATA_DIR, EXPERIMENTS_DIR)
