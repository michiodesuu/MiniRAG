CREATE TABLE conversations (
    id          text PRIMARY KEY,
    owner_id    text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX conversations_owner ON conversations (owner_id, updated_at DESC);

CREATE TABLE conversation_messages (
    id               bigserial PRIMARY KEY,
    conversation_id   text NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    role             text NOT NULL CHECK (role IN ('user', 'assistant')),
    content          text NOT NULL,
    created_at       timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX conversation_messages_order
    ON conversation_messages (conversation_id, id);
