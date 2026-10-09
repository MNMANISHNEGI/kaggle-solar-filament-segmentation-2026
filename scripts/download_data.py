"""Download the competition dataset with kagglehub.

Requires KAGGLE_API_TOKEN (in the environment or a .env file) and accepted competition rules:
https://www.kaggle.com/competitions/filament-segmentation-2026/rules

Usage:
    python scripts/download_data.py
"""

from dotenv import load_dotenv

load_dotenv()  # must run before kagglehub reads credentials

import kagglehub  # noqa: E402

from filament_seg.data import COMPETITION  # noqa: E402

if __name__ == "__main__":
    path = kagglehub.competition_download(COMPETITION)
    print("Dataset downloaded to:", path)
