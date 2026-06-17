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
    CHECK (status IN ('pending', 'queued', 'processing', 'complete', 'partial', 'failed', 'cancelled', 'error'))
);

CREATE TABLE IF NOT EXISTS processing_jobs (
    job_id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    source_path TEXT NOT NULL,
    priority TEXT NOT NULL DEFAULT 'normal',
    status TEXT NOT NULL DEFAULT 'queued',
    stage TEXT NOT NULL DEFAULT 'pending',
    progress_percent REAL NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    queue_position INTEGER,
    frames_extracted INTEGER NOT NULL DEFAULT 0,
    captions_generated INTEGER NOT NULL DEFAULT 0,
    transcript_segments INTEGER NOT NULL DEFAULT 0,
    detections_created INTEGER NOT NULL DEFAULT 0,
    crops_created INTEGER NOT NULL DEFAULT 0,
    text_vectors_indexed INTEGER NOT NULL DEFAULT 0,
    image_vectors_indexed INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TEXT,
    completed_at TEXT,
    updated_at TEXT,
    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE,
    CHECK (job_id != ''),
    CHECK (video_id != ''),
    CHECK (run_id != ''),
    CHECK (source_path != ''),
    CHECK (priority IN ('high', 'normal', 'low')),
    CHECK (status IN ('queued', 'processing', 'complete', 'partial', 'failed', 'cancelled')),
    CHECK (stage IN ('pending', 'extracting', 'captioning', 'enriching', 'indexing', 'reporting', 'complete', 'failed')),
    CHECK (progress_percent >= 0 AND progress_percent <= 100),
    CHECK (queue_position IS NULL OR queue_position >= 1),
    CHECK (frames_extracted >= 0),
    CHECK (captions_generated >= 0),
    CHECK (transcript_segments >= 0),
    CHECK (detections_created >= 0),
    CHECK (crops_created >= 0),
    CHECK (text_vectors_indexed >= 0),
    CHECK (image_vectors_indexed >= 0)
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
    CHECK (context_type IN ('frame', 'caption', 'transcript', 'object', 'crop', 'metadata')),
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

CREATE TABLE IF NOT EXISTS processing_idempotency (
    video_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    run_id TEXT NOT NULL,
    counts TEXT NOT NULL,
    completed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (video_id, tool_name, idempotency_key),
    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE,
    CHECK (video_id != ''),
    CHECK (tool_name != ''),
    CHECK (idempotency_key != ''),
    CHECK (run_id != ''),
    CHECK (counts != '')
);

CREATE INDEX IF NOT EXISTS idx_memory_video_created
    ON memory(video_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_processing_jobs_video_created
    ON processing_jobs(video_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_processing_jobs_status_priority_created
    ON processing_jobs(status, priority, created_at);

CREATE INDEX IF NOT EXISTS idx_video_context_video_type_time
    ON video_context(video_id, context_type, timestamp_sec);

CREATE INDEX IF NOT EXISTS idx_video_context_type_time
    ON video_context(context_type, timestamp_sec);

CREATE INDEX IF NOT EXISTS idx_lineage_video_created
    ON lineage(video_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_processing_idempotency_video_completed
    ON processing_idempotency(video_id, completed_at DESC);

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (1, 'Initial Vidra storage schema');
