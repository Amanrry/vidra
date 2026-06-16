PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS videos (
    video_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL UNIQUE,
    duration_sec REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    error TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT,
    CHECK (video_id != ''),
    CHECK (filename != ''),
    CHECK (file_path != ''),
    CHECK (duration_sec >= 0),
    CHECK (status IN ('pending', 'processing', 'complete', 'error'))
);

CREATE TABLE IF NOT EXISTS memory (
    message_id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE,
    CHECK (message_id != ''),
    CHECK (role IN ('user', 'assistant', 'system')),
    CHECK (content != '')
);

CREATE TABLE IF NOT EXISTS video_context (
    context_id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    context_type TEXT NOT NULL,
    timestamp_sec REAL,
    data TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    model_name TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE,
    CHECK (context_id != ''),
    CHECK (context_type IN ('frame', 'caption', 'transcript', 'object', 'crop', 'alignment', 'metadata')),
    CHECK (timestamp_sec IS NULL OR timestamp_sec >= 0),
    CHECK (data != ''),
    CHECK (tool_name != '')
);

CREATE TABLE IF NOT EXISTS lineage (
    lineage_id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    context_id TEXT,
    operation TEXT NOT NULL,
    tool_name TEXT,
    model_name TEXT,
    parameters TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE,
    FOREIGN KEY (context_id) REFERENCES video_context(context_id) ON DELETE SET NULL,
    CHECK (lineage_id != ''),
    CHECK (operation IN ('create', 'update', 'delete', 'reprocess'))
);

CREATE INDEX IF NOT EXISTS idx_memory_video_created
    ON memory(video_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_video_context_video_type_time
    ON video_context(video_id, context_type, timestamp_sec);

CREATE INDEX IF NOT EXISTS idx_video_context_type_time
    ON video_context(context_type, timestamp_sec);

CREATE INDEX IF NOT EXISTS idx_lineage_video_created
    ON lineage(video_id, created_at DESC);

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (1, 'Initial Vidra storage schema');
