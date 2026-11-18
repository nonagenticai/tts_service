"""Unit tests for the ASGI application module."""

import logging

from src.asgi import HealthCheckLoggingFilter, app


class TestHealthCheckLoggingFilter:
    def test_filters_health_endpoint(self):
        filt = HealthCheckLoggingFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='GET /health HTTP/1.1" 200',
            args=(),
            exc_info=None,
        )
        assert filt.filter(record) is False

    def test_filters_health_ready_endpoint(self):
        filt = HealthCheckLoggingFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='GET /health/ready HTTP/1.1" 200',
            args=(),
            exc_info=None,
        )
        assert filt.filter(record) is False

    def test_filters_health_live_endpoint(self):
        filt = HealthCheckLoggingFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='GET /health/live HTTP/1.1" 200',
            args=(),
            exc_info=None,
        )
        assert filt.filter(record) is False

    def test_passes_non_health_endpoint(self):
        filt = HealthCheckLoggingFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='POST /api/tts/generate HTTP/1.1" 200',
            args=(),
            exc_info=None,
        )
        assert filt.filter(record) is True


class TestAppCreation:
    def test_app_exists(self):
        assert app is not None

    def test_app_title(self):
        assert app.title == "TTS Service"

    def test_app_routes_include_tts(self):
        route_paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/api/tts/generate" in route_paths
        assert "/health" in route_paths
        assert "/" in route_paths

    def test_app_routes_include_usage(self):
        route_paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/api/tts/usage" in route_paths
