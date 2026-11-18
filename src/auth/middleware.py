"""License token authentication middleware -- delegates to w2v_common."""

from w2v_common.auth.middleware import (
    PUBLIC_ENDPOINTS as PUBLIC_ENDPOINTS,
)
from w2v_common.auth.middleware import (
    PUBLIC_PREFIXES as PUBLIC_PREFIXES,
)
from w2v_common.auth.middleware import (
    TRUSTED_NETWORKS as TRUSTED_NETWORKS,
)
from w2v_common.auth.middleware import (
    AuthMiddleware as AuthMiddleware,
)
from w2v_common.auth.middleware import (
    _get_license_cache as _get_license_cache,
)
from w2v_common.auth.middleware import (
    _introspect_license as _introspect_license,
)
from w2v_common.auth.middleware import (
    _is_trusted_network as _is_trusted_network,
)

# ``_looks_like_jwt`` physically lives in ``w2v_common.auth.keycloak`` but is
# imported into ``w2v_common.auth.middleware``'s namespace by the licence
# extraction rule that uses it. Re-exported here (not re-implemented) so the WS
# handshake in ``src.api.ws`` can apply the SAME JWT-shape test the canonical
# ``AuthMiddleware._extract_license_token`` applies -- a local copy of a
# three-segment check is exactly the kind of drift this seam exists to prevent.
from w2v_common.auth.middleware import (
    _looks_like_jwt as _looks_like_jwt,
)

__all__ = [
    "PUBLIC_ENDPOINTS",
    "PUBLIC_PREFIXES",
    "TRUSTED_NETWORKS",
    "AuthMiddleware",
    "_get_license_cache",
    "_introspect_license",
    "_is_trusted_network",
    "_looks_like_jwt",
]
