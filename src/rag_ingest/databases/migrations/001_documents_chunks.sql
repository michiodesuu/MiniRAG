-- Documents and their chunks. Document id is the link to Qdrant: every point's
-- payload carries it, and a document is replaced in both stores together.
-- Access control and conversations belong to later parts; role and status are
-- stored here so those parts can use them without a rewrite.

CREATE TABLE documents (
    id            text PRIMARY KEY,          -- document stem, as in chunk ids
    name          text NOT NULL,
    source_path   text NOT NULL,             -- the ingest .md the chunks were cut from
    content_hash  text NOT NULL,             -- sha256 of that .md (chunk source_sha256)
    role          text,
    doc_type      text NOT NULL,             -- pdf / docx / unknown
    status        text NOT NULL DEFAULT 'unverified'
                  CHECK (status IN ('verified', 'unverified')),
    ingested_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
    chunk_id         text NOT NULL,          -- <doc>#p<page>#c<nn>
    index_run        text NOT NULL,          -- index folder = Qdrant collection
    document_id      text NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    section          text NOT NULL DEFAULT '',
    page_start       integer NOT NULL,
    page_end         integer NOT NULL,
    type             text NOT NULL CHECK (type IN ('text', 'table', 'screenshot')),
    text             text NOT NULL,          -- display_text: what an answer cites
    token_count      integer NOT NULL,
    embedding_model  text NOT NULL,
    PRIMARY KEY (index_run, chunk_id)
);

CREATE INDEX chunks_document ON chunks (document_id);
