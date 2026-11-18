"""Unit tests for structured logging configuration."""

import json
import logging
from io import StringIO

from src.config import Settings, clear_settings_cache
from tests.conftest import tenant_uuid


class TestConfigureLogging:
    def setup_method(self):
        clear_settings_cache()

    def teardown_method(self):
        clear_settings_cache()
        # Restore root logger
        root = logging.getLogger()
        root.handlers.clear()
        root.setLevel(logging.WARNING)

    def test_json_logs_output(self):
        """Test that JSON logging mode produces valid JSON."""
        from src.logging_config import configure_logging

        settings = Settings(logging={"json_logs": True, "level": "INFO"})
        configure_logging(settings)

        # Capture output
        test_logger = logging.getLogger("test.json")
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        # Re-use the formatter from root
        root = logging.getLogger()
        if root.handlers:
            handler.setFormatter(root.handlers[0].formatter)
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.INFO)

        test_logger.info("test message")

        output = stream.getvalue().strip()
        assert output  # Something was logged
        # Should be valid JSON
        parsed = json.loads(output)
        assert "event" in parsed
        assert parsed["event"] == "test message"

    def test_console_logs_output(self):
        """Test that console logging mode produces readable output."""
        from src.logging_config import configure_logging

        settings = Settings(logging={"json_logs": False, "level": "DEBUG"})
        configure_logging(settings)

        test_logger = logging.getLogger("test.console")
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        root = logging.getLogger()
        if root.handlers:
            handler.setFormatter(root.handlers[0].formatter)
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.DEBUG)

        test_logger.debug("debug message")

        output = stream.getvalue()
        assert "debug message" in output

    def test_log_level_respected(self):
        """Test that the configured log level is respected."""
        from src.logging_config import configure_logging

        settings = Settings(logging={"level": "ERROR"})
        configure_logging(settings)

        root = logging.getLogger()
        assert root.level == logging.ERROR

    def test_structlog_context_binding(self):
        """Test that structlog context variables are merged into log output."""
        import structlog

        from src.logging_config import configure_logging

        settings = Settings(logging={"json_logs": True, "level": "INFO"})
        configure_logging(settings)

        structlog.contextvars.bind_contextvars(
            tenant_id=tenant_uuid("test-tenant"), request_id="abc123"
        )

        test_logger = logging.getLogger("test.context")
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        root = logging.getLogger()
        if root.handlers:
            handler.setFormatter(root.handlers[0].formatter)
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.INFO)

        test_logger.info("context test")

        output = stream.getvalue().strip()
        parsed = json.loads(output)
        assert parsed.get("tenant_id") == tenant_uuid("test-tenant")
        assert parsed.get("request_id") == "abc123"

        # Clean up context
        structlog.contextvars.clear_contextvars()
