"""Chunking: outputs/ingest/*.md -> outputs/chunks/chunks.jsonl (spec section 3).

Modules, in pipeline order:
  sources   reads the ingest output (Markdown, manifest.json, ocr_cache.json)
  units     parses one Markdown file into ordered units (text, heading,
            table, screenshot, placeholder) with their character ranges
  sections  drops table-of-contents pages; assigns heading paths
            (Strategy: Markdown headings for Word, numbered headings for PDF)
  text      a unit's embed form (NFC, tables per --tables) and display form
  tables    Markdown/HTML tables -> rows -> "keep" or "flatten" rendering
  builder   packs a section's units into chunks within the token budget
            (Builder), never splitting screenshots/tables unless they alone
            exceed it, with whole-line overlap
  guards    named rules: PII (Thai national ID with checksum, phone flags),
            empty, duplicates; reports positions, never values
  pipeline  runs the above per document and writes the outputs

Tokens are counted by a callable passed in: the embedding model's own
tokenizer in real runs, tokens.approximate_tokens only in tests / `fake`.
"""
from rag_ingest.chunking.pipeline import (ChunkRun, ChunkSettings, CorpusConfig,  # noqa: F401
                                          chunk_corpus, write_run)
