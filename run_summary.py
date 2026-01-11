from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)

from src.summarizer.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
