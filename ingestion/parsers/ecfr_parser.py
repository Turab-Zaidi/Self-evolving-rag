"""Parses eCFR XML into parent (section) and child (paragraph) Chunks."""

import re
import logging
from lxml import etree
from typing import List
from pipeline.state import Chunk
from config import cfg

logger = logging.getLogger(__name__)


def parse_ecfr_xml(xml_content: str, part_number: int) -> List[Chunk]:
    if not xml_content.strip():
        return []
    try:
        root = etree.fromstring(xml_content.encode('utf-8'))
    except etree.XMLSyntaxError as e:
        logger.error(f"Failed to parse XML for Part {part_number}: {e}")
        return []

    chunks = []
    sections = root.xpath("//*[@TYPE='SECTION']")

    for section in sections:
        section_number = section.get('N', '')
        if not section_number:
            continue

        heading_elem = section.find('./HEAD')
        heading = heading_elem.text.strip() if heading_elem is not None and heading_elem.text else ""
        parent_id = f"ecfr-{cfg.data.ECFR_TITLE}-{part_number}-{section_number}"

        parent_text = " ".join(section.itertext()).strip()
        parent_text = re.sub(r'\s+', ' ', parent_text)
        cross_refs = list(set(re.findall(r'(?:Part\s+\d+|§\s*\d+\.\d+)', parent_text)))

        metadata = {
            "source_type": "ecfr",
            "cfr_title": cfg.data.ECFR_TITLE,
            "cfr_part": part_number,
            "cfr_section": section_number,
            "section_heading": heading,
            "cross_references": cross_refs
        }

        parent_meta = {**metadata, "chunk_type": "section"}
        chunks.append(Chunk(
            chunk_id=parent_id, parent_id=None, text=parent_text,
            source_type="ecfr", collection=cfg.storage.ECFR_COLLECTION, metadata=parent_meta
        ))

        for i, p_elem in enumerate(section.findall('.//P')):
            p_text = re.sub(r'\s+', ' ', "".join(p_elem.itertext()).strip())
            if len(p_text.split()) < 10:
                continue
            child_meta = {**metadata, "chunk_type": "paragraph"}
            chunks.append(Chunk(
                chunk_id=f"{parent_id}-p{i}", parent_id=parent_id, text=p_text,
                source_type="ecfr", collection=cfg.storage.ECFR_COLLECTION, metadata=child_meta
            ))

    logger.info(f"Part {part_number}: Extracted {len(chunks)} chunks.")
    return chunks
