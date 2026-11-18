"""Integration tests for database models (requires running database)."""

import pytest


@pytest.mark.integration
class TestDatabaseModels:
    """These tests require a running PostgreSQL instance."""

    def test_tts_job_model_importable(self):
        from src.database.models import Base, TTSJob

        assert TTSJob.__tablename__ == "tts_jobs"
        assert "tts_jobs" in Base.metadata.tables

    def test_tts_usage_model_importable(self):
        from src.database.models import Base, TTSUsage

        assert TTSUsage.__tablename__ == "tts_usage"
        assert "tts_usage" in Base.metadata.tables

    def test_tts_job_columns(self):
        from src.database.models import TTSJob

        columns = {c.name for c in TTSJob.__table__.columns}
        expected = {
            "id",
            "job_id",
            "batch_id",
            "tenant_id",
            "text",
            "text_length",
            "provider",
            "voice",
            "speed",
            "output_format",
            "model",
            "extra_params",
            "status",
            "error_message",
            "audio_path",
            "audio_size_bytes",
            "audio_duration_seconds",
            "caller_item_id",
            "created_at",
            "started_at",
            "completed_at",
        }
        assert expected.issubset(columns)
