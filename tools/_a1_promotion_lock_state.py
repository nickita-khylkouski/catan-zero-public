"""Process-local state shared by every promotion transaction module identity.

Running ``a1_promotion_transaction.py`` as a script names that module
``__main__``. Post-promotion handoff replay imports the same source as
``tools.a1_promotion_transaction``. Keeping the thread-local ownership state
here ensures both module identities see the same already-held kernel lock.
"""

from __future__ import annotations

import threading


LOCK_STATE = threading.local()
