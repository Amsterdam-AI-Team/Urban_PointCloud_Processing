# Helper script to allow importing from parent folder.
import sys
import os

module_path = os.path.abspath(os.path.join('../src'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)
