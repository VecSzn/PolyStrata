"""Second round: the prompt-cache question at ~118K context.

Both arms run the winning combination (FATTN + FDEC) and differ only in how
much conversation state the engine keeps:
  E  more checkpoints (--prompt-cache 16 --prompt-cache-every 1024)
  F  no checkpoints at all (--prompt-cache 0); parking alone must resume
"""
import json
import os
import shutil

import run_arms as R

here = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(here, "results.json")):
    shutil.copy(os.path.join(here, "results.json"), os.path.join(here, "results-arms1.json"))

BOTH = {"STRATA_FATTN": "1", "STRATA_FDEC": "1"}
R.ARMS = [
    ("E-cache16", dict(BOTH), ["--prompt-cache", "16", "--prompt-cache-every", "1024"]),
    ("F-nocache", dict(BOTH), ["--prompt-cache", "0"]),
]

if __name__ == "__main__":
    R.ARMS = [
        ("E-cache16", dict(BOTH), ["--prompt-cache", "16", "--prompt-cache-every", "1024"]),
        ("F-nocache", dict(BOTH), ["--prompt-cache", "0"]),
    ]
    R.main()
