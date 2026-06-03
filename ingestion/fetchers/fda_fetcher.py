"""Downloads key FDA Guidance PDFs from predefined URLs."""

import requests
import logging
from config import cfg

logger = logging.getLogger(__name__)

FDA_GUIDANCE_DOCS = {
    "design_controls_guidance": "https://web.archive.org/web/20231201000000/https://www.fda.gov/media/116573/download",
    "software_validation_guidance": "https://web.archive.org/web/20231201000000/https://www.fda.gov/media/73141/download",
    "deciding_when_to_submit_510k": "https://web.archive.org/web/20231201000000/https://www.fda.gov/media/99785/download",
    "mdr_adverse_event_reporting": "https://web.archive.org/web/20231201000000/https://www.fda.gov/media/86598/download",
    "cybersecurity_premarket": "https://web.archive.org/web/20231201000000/https://www.fda.gov/media/119933/download"
}


def fetch_fda_guidance(force_download: bool = False):
    cache_dir = cfg.data.FDA_GUIDANCE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    for doc_name, url in FDA_GUIDANCE_DOCS.items():
        pdf_path = cache_dir / f"{doc_name}.pdf"
        if pdf_path.exists() and not force_download:
            logger.info(f"Guidance '{doc_name}' already downloaded.")
            continue

        logger.info(f"Downloading FDA Guidance: {doc_name}...")
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, stream=True)
            response.raise_for_status()
            with open(pdf_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Successfully saved {doc_name}.pdf")
        except Exception as e:
            logger.error(f"Failed to download {doc_name}: {e}")

if __name__ == "__main__":
    fetch_fda_guidance()
