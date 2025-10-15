import logging
from pathlib import Path

# Create logs folder if it doesn’t exist
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Configure the logger
LOG_FILE = LOG_DIR / "app.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()  # also log to console
    ]
)

# Shortcut function
def get_logger(name: str):
    return logging.getLogger(name)
