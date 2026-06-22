import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class Settings:
    DATA_FOLDER = Path(os.getenv("DATA_FOLDER"))
    TEST_FOLDER = Path(os.getenv("TEST_FOLDER"))
    RECON_OUTPUT_FOLDER = Path(os.getenv("RECON_OUTPUT_FOLDER"))

    PORT = int(os.getenv("PORT", 8000))

    def create_dirs(self):
        self.TEST_FOLDER.mkdir(parents=True, exist_ok=True)
        self.RECON_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.create_dirs()
