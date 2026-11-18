"""Additional unit tests for provider registry."""

from unittest.mock import MagicMock

from src.providers.registry import ProviderRegistry


class TestProviderRegistryExtended:
    def test_register_duplicate(self):
        """Registering same provider name twice should update."""
        registry = ProviderRegistry()
        p1 = MagicMock()
        p1.name = "kokoro"
        p2 = MagicMock()
        p2.name = "kokoro"

        registry.register(p1)
        registry.register(p2)

        # Should have latest
        assert registry.get_provider("kokoro") is p2
        assert len(registry.provider_names) == 1

    def test_provider_names(self):
        registry = ProviderRegistry()
        p1 = MagicMock()
        p1.name = "kokoro"
        p2 = MagicMock()
        p2.name = "openai"

        registry.register(p1)
        registry.register(p2)

        names = registry.provider_names
        assert "kokoro" in names
        assert "openai" in names

    def test_list_providers_returns_instances(self):
        registry = ProviderRegistry()
        p = MagicMock()
        p.name = "test"
        registry.register(p)

        providers = registry.list_providers()
        assert len(providers) == 1
        assert providers[0] is p
