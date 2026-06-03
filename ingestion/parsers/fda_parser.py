"""Parses FDA Guidance PDFs into parent (section) and child (paragraph) Chunks."""

import fitz
import re
import logging
import statistics
from pathlib import Path
from typing import List
from pipeline.state import Chunk
from config import cfg

logger = logging.getLogger(__name__)


def extract_blocks_with_formatting(pdf_path: Path):
    doc = fitz.open(pdf_path)
    blocks = []
    for page_num, page in enumerate(doc):
        page_dict = page.get_text("dict")
        for b in page_dict.get("blocks", []):
            if b.get('type') != 0:
                continue
            block_text, max_size, is_bold = "", 0, False
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    block_text += span.get("text", "") + " "
                    if span.get("size", 0) > max_size:
                        max_size = span.get("size", 0)
                    if "bold" in span.get("font", "").lower():
                        is_bold = True
            block_text = re.sub(r'\s+', ' ', block_text).strip()
            if block_text:
                blocks.append({"text": block_text, "size": round(max_size, 1), "bold": is_bold, "page": page_num + 1})
    return blocks


def parse_fda_pdf(pdf_path: Path, doc_id: str, document_title: str) -> List[Chunk]:
    if not pdf_path.exists():
        logger.error(f"File not found: {pdf_path}")
        return []

    blocks = extract_blocks_with_formatting(pdf_path)
    if not blocks:
        return []

    sizes = [b["size"] for b in blocks if len(b["text"]) > 20]
    median_size = statistics.median(sizes) if sizes else 11.0
    chunks = []
    current_heading = "Introduction"
    current_section_blocks = []
    section_counter = 0

    def process_section(heading, section_blocks):
        if not section_blocks:
            return
        nonlocal section_counter
        section_counter += 1
        parent_id = f"fda-{doc_id}-sec{section_counter}"
        parent_text = f"{heading}\n\n" + "\n\n".join([b["text"] for b in section_blocks])
        metadata = {
            "source_type": "fda_guidance", "document_title": document_title,
            "document_id": doc_id, "section_heading": heading
        }
        chunks.append(Chunk(
            chunk_id=parent_id, parent_id=None, text=parent_text,
            source_type="fda_guidance", collection=cfg.storage.GUIDANCE_COLLECTION,
            metadata={**metadata, "chunk_type": "section"}
        ))
        for i, b in enumerate(section_blocks):
            if len(b["text"].split()) < 15:
                continue
            chunks.append(Chunk(
                chunk_id=f"{parent_id}-p{i}", parent_id=parent_id, text=b["text"],
                source_type="fda_guidance", collection=cfg.storage.GUIDANCE_COLLECTION,
                metadata={**metadata, "chunk_type": "paragraph", "page_number": b["page"]}
            ))

    for block in blocks:
        is_heading = (block["size"] > median_size + 0.5 or block["bold"]) and len(block["text"].split()) < 20
        if block["size"] < median_size - 1.0 or block["text"].isnumeric():
            continue
        if is_heading:
            process_section(current_heading, current_section_blocks)
            current_heading = block["text"]
            current_section_blocks = []
        else:
            current_section_blocks.append(block)

    process_section(current_heading, current_section_blocks)
    logger.info(f"{doc_id}: Extracted {len(chunks)} chunks.")
    return chunks
