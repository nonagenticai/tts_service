"""Unit tests for API schemas."""

import pytest
from pydantic import ValidationError

from src.api.schemas import TTSBatchItem, TTSBatchRequest, TTSGenerateRequest


class TestTTSGenerateRequest:
    def test_defaults(self):
        req = TTSGenerateRequest(text="Hello world")
        assert req.provider == "auto"
        assert req.voice is None  # voice resolved at service layer now
        assert req.speed == 1.0
        assert req.output_format == "mp3"
        assert req.async_mode is False
        assert req.model is None

    def test_text_required(self):
        with pytest.raises(ValidationError):
            TTSGenerateRequest()

    def test_text_min_length(self):
        with pytest.raises(ValidationError):
            TTSGenerateRequest(text="")

    def test_text_max_length(self):
        with pytest.raises(ValidationError):
            TTSGenerateRequest(text="x" * 10001)

    def test_speed_range(self):
        TTSGenerateRequest(text="test", speed=0.25)  # min
        TTSGenerateRequest(text="test", speed=4.0)  # max
        with pytest.raises(ValidationError):
            TTSGenerateRequest(text="test", speed=0.1)
        with pytest.raises(ValidationError):
            TTSGenerateRequest(text="test", speed=5.0)

    def test_output_format_validation(self):
        TTSGenerateRequest(text="test", output_format="mp3")
        TTSGenerateRequest(text="test", output_format="wav")
        with pytest.raises(ValidationError):
            TTSGenerateRequest(text="test", output_format="ogg")

    def test_custom_values(self):
        req = TTSGenerateRequest(
            text="Hello",
            provider="kitten",
            voice="Bella",
            speed=1.5,
            output_format="wav",
            model="mini",
            async_mode=True,
        )
        assert req.provider == "kitten"
        assert req.voice == "Bella"
        assert req.model == "mini"
        assert req.async_mode is True

    def test_provider_params_accepts_bounded_values(self):
        req = TTSGenerateRequest(
            text="Hello",
            provider_params={"temperature": 0.7, "top_p": 1.0, "seed": 42.0},
        )
        assert req.provider_params == {"temperature": 0.7, "top_p": 1.0, "seed": 42.0}

    def test_provider_params_rejects_too_many_keys(self):
        too_many = {f"k{i}": 1.0 for i in range(33)}
        with pytest.raises(ValidationError, match="at most 32 keys"):
            TTSGenerateRequest(text="Hello", provider_params=too_many)

    def test_provider_params_rejects_out_of_range_value(self):
        with pytest.raises(ValidationError, match="provider_params"):
            TTSGenerateRequest(text="Hello", provider_params={"temperature": 1_000_000.0})

    def test_provider_params_rejects_negative_out_of_range(self):
        with pytest.raises(ValidationError, match="provider_params"):
            TTSGenerateRequest(text="Hello", provider_params={"bias": -1_000_001.0})

    def test_provider_params_rejects_nonfinite_value(self):
        with pytest.raises(ValidationError, match="provider_params"):
            TTSGenerateRequest(text="Hello", provider_params={"x": float("inf")})
        with pytest.raises(ValidationError, match="provider_params"):
            TTSGenerateRequest(text="Hello", provider_params={"x": float("nan")})

    def test_provider_params_rejects_long_key(self):
        with pytest.raises(ValidationError, match="provider_params"):
            TTSGenerateRequest(text="Hello", provider_params={"x" * 65: 0.5})

    def test_provider_params_accepts_boundary_values(self):
        req = TTSGenerateRequest(
            text="Hello",
            provider_params={"low": -1000.0, "high": 1000.0},
        )
        assert req.provider_params == {"low": -1000.0, "high": 1000.0}


class TestTTSBatchRequest:
    def test_valid_batch(self):
        req = TTSBatchRequest(
            items=[
                TTSBatchItem(id="1", text="Hello"),
                TTSBatchItem(id="2", text="World"),
            ],
            provider="kokoro",
        )
        assert len(req.items) == 2
        assert req.concatenate is False

    def test_empty_items_rejected(self):
        with pytest.raises(ValidationError):
            TTSBatchRequest(items=[])

    def test_max_items(self):
        items = [TTSBatchItem(id=str(i), text=f"text {i}") for i in range(101)]
        with pytest.raises(ValidationError):
            TTSBatchRequest(items=items)
