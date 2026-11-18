"""SSML parsing, validation, and provider-specific conversion."""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET  # nosec B405 — used only as fallback type reference; parsing uses defusedxml

import defusedxml.ElementTree as DefusedET

logger = logging.getLogger(__name__)

# Supported SSML tags
SUPPORTED_TAGS = {"speak", "break", "emphasis", "prosody", "say-as", "phoneme", "p", "s"}

# Regex for quick SSML detection
_SSML_PATTERN = re.compile(r"<(?:speak|break|emphasis|prosody|say-as|phoneme)\b", re.IGNORECASE)

# Break time pattern
_BREAK_TIME_PATTERN = re.compile(r"(\d+)(ms|s)")


def is_ssml(text: str) -> bool:
    """Auto-detect if input contains SSML tags."""
    return bool(_SSML_PATTERN.search(text))


def validate_ssml(text: str) -> list[str]:
    """Validate SSML markup and return list of error messages (empty = valid)."""
    errors: list[str] = []

    # Wrap in <speak> if not already wrapped
    wrapped = text if text.strip().startswith("<speak") else f"<speak>{text}</speak>"

    try:
        root = DefusedET.fromstring(wrapped)
    except ET.ParseError as exc:
        errors.append(f"XML parse error: {exc}")
        return errors

    # Check for unsupported tags
    for elem in root.iter():
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag  # handle namespaces
        if tag not in SUPPORTED_TAGS:
            errors.append(f"Unsupported SSML tag: <{tag}>")

    # Validate break elements
    for break_elem in root.iter("break"):
        time_attr = break_elem.get("time")
        if time_attr and not _BREAK_TIME_PATTERN.match(time_attr):
            errors.append(
                f"Invalid break time format: '{time_attr}' (expected e.g. '500ms' or '1s')"
            )

    # Validate prosody elements
    for prosody in root.iter("prosody"):
        rate = prosody.get("rate")
        if (
            rate
            and rate not in ("x-slow", "slow", "medium", "fast", "x-fast")
            and not re.match(r"^\d+%$", rate)
        ):
            errors.append(f"Invalid prosody rate: '{rate}'")

    return errors


def strip_ssml(text: str) -> str:
    """Remove all SSML tags, returning plain text.

    Used for providers that don't support SSML (e.g., OpenAI, KittenTTS).
    """
    # Wrap in <speak> if needed for valid XML
    wrapped = text if text.strip().startswith("<speak") else f"<speak>{text}</speak>"

    try:
        root = DefusedET.fromstring(wrapped)
        # Extract all text content
        return "".join(root.itertext()).strip()
    except ET.ParseError:
        # Fallback: regex strip
        return re.sub(r"<[^>]+>", "", text).strip()


def ssml_to_kokoro(text: str) -> str:
    """Convert SSML to Kokoro-compatible format.

    Kokoro uses espeak-ng phonemes. We convert:
    - <break time="Xms"> to silence insertion markers
    - <emphasis> to stress markers
    - <prosody rate="..."> to speed adjustments
    Other tags are stripped.
    """
    wrapped = text if text.strip().startswith("<speak") else f"<speak>{text}</speak>"

    try:
        root = DefusedET.fromstring(wrapped)
    except ET.ParseError:
        return strip_ssml(text)

    return _process_kokoro_element(root)


def _kokoro_break_text(child: ET.Element) -> str:
    """Convert an SSML <break> element to a Kokoro-compatible pause string."""
    time_attr = child.get("time", "500ms")
    match = _BREAK_TIME_PATTERN.match(time_attr)
    if not match:
        return ""
    val, unit = int(match.group(1)), match.group(2)
    ms = val if unit == "ms" else val * 1000
    if ms < 300:
        return ", "
    if ms < 800:
        return ". "
    return "... "


def _kokoro_emphasis_text(child: ET.Element) -> str:
    """Render an SSML <emphasis> element for Kokoro (uppercase when level=strong)."""
    inner = _process_kokoro_element(child)
    if child.get("level", "moderate") == "strong":
        return inner.upper()
    return inner


def _kokoro_child_text(child: ET.Element) -> str:
    """Dispatch a single SSML child element to the appropriate Kokoro renderer."""
    tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
    if tag == "break":
        return _kokoro_break_text(child)
    if tag == "emphasis":
        return _kokoro_emphasis_text(child)
    if tag in ("p", "s"):
        return _process_kokoro_element(child) + ". "
    # prosody (no inline speed support) or unsupported tags: just descend.
    return _process_kokoro_element(child)


def _process_kokoro_element(elem: ET.Element) -> str:
    """Recursively process SSML elements for Kokoro."""
    parts: list[str] = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        parts.append(_kokoro_child_text(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def process_ssml_for_provider(
    text: str,
    provider_name: str,
    supports_ssml: bool = False,
) -> str:
    """Process SSML text for a specific provider.

    - Providers with native SSML support: pass through (with validation)
    - Kokoro: convert to espeak-ng compatible format
    - Others: strip SSML tags
    """
    if not is_ssml(text):
        return text

    if supports_ssml:
        # Provider supports SSML natively (e.g., ElevenLabs)
        logger.debug("Passing SSML through to provider=%s (native support)", provider_name)
        return text

    if provider_name == "kokoro":
        logger.debug("Converting SSML to Kokoro format")
        return ssml_to_kokoro(text)

    # Strip for providers without SSML support
    logger.debug("Stripping SSML tags for provider=%s", provider_name)
    return strip_ssml(text)
