"""Fetches raw XML for CFR parts from the eCFR API with local caching."""

import requests
import logging
from config import cfg

logger = logging.getLogger(__name__)


def get_latest_issue_date() -> str:
    url = f"{cfg.data.ECFR_API_BASE}/titles.json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        for t in response.json().get('titles', []):
            if t.get('number') == cfg.data.ECFR_TITLE:
                return t.get('latest_issue_date')
    except Exception as e:
        logger.warning(f"Failed to fetch latest issue date: {e}. Falling back to 2026-05-18.")
    return "2026-05-18"


def fetch_ecfr_part(part_number: int, force_download: bool = False) -> str:
    cache_dir = cfg.data.ECFR_RAW_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"part_{part_number}.xml"

    if cache_file.exists() and not force_download:
        logger.info(f"Loading Part {part_number} from local cache.")
        return cache_file.read_text(encoding="utf-8")

    stable_date = get_latest_issue_date()
    url = f"{cfg.data.ECFR_API_BASE}/full/{stable_date}/title-{cfg.data.ECFR_TITLE}.xml?part={part_number}"
    logger.info(f"Downloading Part {part_number} from {url}...")

    response = requests.get(url, headers={"Accept": "application/xml"})
    if response.status_code == 404:
        logger.warning(f"Part {part_number} does not exist or was removed.")
        return ""

    response.raise_for_status()
    cache_file.write_text(response.text, encoding="utf-8")
    logger.info(f"Successfully downloaded and cached Part {part_number}.")
    return response.text


def fetch_all_initial_parts():
    for part in cfg.data.ECFR_PARTS:
        if part not in cfg.data.ECFR_WITHHELD:
            fetch_ecfr_part(part)

if __name__ == "__main__":
    fetch_all_initial_parts()
