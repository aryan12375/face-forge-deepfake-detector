"""
conftest.py
===========
Adds the project root to sys.path so pytest can import `app.*` modules
without needing an editable install.
"""

import sys
from pathlib import Path

# Insert the backend root so `from app.xxx import ...` works in tests
sys.path.insert(0, str(Path(__file__).parent.parent))
