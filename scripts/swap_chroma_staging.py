"""
Swaps the staged Chroma index (chroma_db_staging/, built by scripts/
stage_chroma_index_rebuild.py while the live chroma_db/ was locked by
other running processes) into the live chroma_db/ path.

Safety: refuses to run if anything still appears to have chroma_db/ open
(checked by attempting a rename first, which fails cleanly with the same
PermissionError as a direct rmtree would, rather than partially deleting
anything) -- same "fail closed, not partially" behavior confirmed safe
earlier. The OLD chroma_db/ is renamed aside (chroma_db_old_pre_banglish/),
not deleted, so nothing is destroyed even after a successful swap -- it
can be restored by a simple directory rename if anything looks wrong.

Usage: python scripts/swap_chroma_staging.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CHROMA_DIR = ROOT / "chroma_db"
STAGING_DIR = ROOT / "chroma_db_staging"
OLD_BACKUP_DIR = ROOT / "chroma_db_old_pre_banglish"


def main():
    if not STAGING_DIR.exists():
        raise SystemExit(f"{STAGING_DIR} does not exist -- run scripts/stage_chroma_index_rebuild.py first")

    if OLD_BACKUP_DIR.exists():
        raise SystemExit(f"{OLD_BACKUP_DIR} already exists -- a previous swap wasn't cleaned up; "
                          f"check it manually before re-running")

    if CHROMA_DIR.exists():
        try:
            CHROMA_DIR.rename(OLD_BACKUP_DIR)
        except PermissionError as e:
            raise SystemExit(f"chroma_db/ is still in use by another process -- swap NOT performed, "
                              f"nothing changed. Retry once other jobs finish. ({e})")
        print(f"Moved old {CHROMA_DIR} -> {OLD_BACKUP_DIR} (not deleted, kept as a safety copy)")

    STAGING_DIR.rename(CHROMA_DIR)
    print(f"Staged index is now live at {CHROMA_DIR}")
    print(f"Old index preserved at {OLD_BACKUP_DIR} -- delete it manually once you've confirmed "
          f"the new one works correctly")


if __name__ == "__main__":
    main()
