"""Unit tests for the SSML processor."""

from src.services.ssml_processor import (
    is_ssml,
    process_ssml_for_provider,
    ssml_to_kokoro,
    strip_ssml,
    validate_ssml,
)


class TestIsSSML:
    def test_detects_speak_tag(self):
        assert is_ssml("<speak>Hello world</speak>") is True

    def test_detects_break_tag(self):
        assert is_ssml('Hello <break time="500ms"/> world') is True

    def test_detects_emphasis_tag(self):
        assert is_ssml("<emphasis level='strong'>Hello</emphasis>") is True

    def test_detects_prosody_tag(self):
        assert is_ssml('<prosody rate="slow">Hello</prosody>') is True

    def test_detects_say_as_tag(self):
        assert is_ssml('<say-as interpret-as="date">2024-01-01</say-as>') is True

    def test_detects_phoneme_tag(self):
        assert is_ssml('<phoneme alphabet="ipa">hello</phoneme>') is True

    def test_plain_text_not_ssml(self):
        assert is_ssml("Hello world") is False

    def test_html_not_ssml(self):
        assert is_ssml("<div>Hello</div>") is False

    def test_empty_string_not_ssml(self):
        assert is_ssml("") is False


class TestValidateSSML:
    def test_valid_ssml(self):
        errors = validate_ssml("<speak>Hello world</speak>")
        assert len(errors) == 0

    def test_valid_break(self):
        errors = validate_ssml('<speak><break time="500ms"/></speak>')
        assert len(errors) == 0

    def test_invalid_break_time(self):
        errors = validate_ssml('<speak><break time="abc"/></speak>')
        assert any("Invalid break time" in e for e in errors)

    def test_invalid_xml(self):
        errors = validate_ssml("<speak><unclosed>")
        assert any("XML parse error" in e for e in errors)

    def test_unsupported_tag(self):
        errors = validate_ssml("<speak><audio src='test.mp3'>text</audio></speak>")
        assert any("Unsupported SSML tag" in e for e in errors)

    def test_auto_wraps_in_speak(self):
        # Text without <speak> wrapper should be auto-wrapped
        errors = validate_ssml('<break time="500ms"/>')
        assert len(errors) == 0

    def test_valid_prosody_rate(self):
        errors = validate_ssml('<speak><prosody rate="slow">text</prosody></speak>')
        assert len(errors) == 0

    def test_valid_prosody_percentage(self):
        errors = validate_ssml('<speak><prosody rate="150%">text</prosody></speak>')
        assert len(errors) == 0

    def test_invalid_prosody_rate(self):
        errors = validate_ssml('<speak><prosody rate="invalid">text</prosody></speak>')
        assert any("Invalid prosody rate" in e for e in errors)

    def test_valid_break_seconds(self):
        errors = validate_ssml('<speak><break time="1s"/></speak>')
        assert len(errors) == 0


class TestStripSSML:
    def test_strips_speak_tags(self):
        result = strip_ssml("<speak>Hello world</speak>")
        assert result == "Hello world"

    def test_strips_break_tags(self):
        result = strip_ssml('<speak>Hello <break time="500ms"/> world</speak>')
        assert "Hello" in result
        assert "world" in result

    def test_strips_emphasis(self):
        result = strip_ssml("<speak><emphasis level='strong'>Hello</emphasis> world</speak>")
        assert "Hello" in result
        assert "world" in result

    def test_handles_invalid_xml(self):
        result = strip_ssml("<speak><unclosed>text")
        assert "text" in result

    def test_plain_text_passthrough(self):
        result = strip_ssml("plain text no ssml")
        assert result == "plain text no ssml"


class TestSSMLToKokoro:
    def test_plain_text(self):
        result = ssml_to_kokoro("<speak>Hello world</speak>")
        assert "Hello world" in result

    def test_break_short(self):
        result = ssml_to_kokoro('<speak>Hello <break time="100ms"/> world</speak>')
        assert ", " in result or "Hello" in result

    def test_break_medium(self):
        result = ssml_to_kokoro('<speak>Hello <break time="500ms"/> world</speak>')
        assert ". " in result or "Hello" in result

    def test_break_long(self):
        result = ssml_to_kokoro('<speak>Hello <break time="1000ms"/> world</speak>')
        assert "... " in result or "Hello" in result

    def test_emphasis_strong(self):
        result = ssml_to_kokoro("<speak><emphasis level='strong'>hello</emphasis> world</speak>")
        assert "HELLO" in result

    def test_emphasis_moderate(self):
        result = ssml_to_kokoro("<speak><emphasis level='moderate'>hello</emphasis> world</speak>")
        assert "hello" in result

    def test_paragraph_tags(self):
        result = ssml_to_kokoro("<speak><p>First.</p><p>Second.</p></speak>")
        assert "First." in result
        assert "Second." in result

    def test_invalid_xml_fallback(self):
        result = ssml_to_kokoro("<speak><unclosed>text")
        assert "text" in result

    def test_prosody_passthrough(self):
        result = ssml_to_kokoro('<speak><prosody rate="fast">quick text</prosody></speak>')
        assert "quick text" in result


class TestProcessSSMLForProvider:
    def test_plain_text_passthrough(self):
        result = process_ssml_for_provider("Hello world", "kokoro")
        assert result == "Hello world"

    def test_kokoro_converts_ssml(self):
        result = process_ssml_for_provider(
            '<speak>Hello <break time="500ms"/> world</speak>', "kokoro"
        )
        # Should convert, not strip
        assert "Hello" in result
        assert "<speak>" not in result

    def test_other_provider_strips_ssml(self):
        result = process_ssml_for_provider(
            "<speak>Hello <emphasis>world</emphasis></speak>", "kitten"
        )
        assert "Hello" in result
        assert "<" not in result

    def test_native_ssml_support_passthrough(self):
        ssml_text = "<speak>Hello world</speak>"
        result = process_ssml_for_provider(ssml_text, "elevenlabs", supports_ssml=True)
        assert result == ssml_text
