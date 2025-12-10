import os
import sys

# Ensure the fca directory is accessible throughout the tests package
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
print(src_path)
if src_path not in sys.path:
    sys.path.insert(0, src_path)