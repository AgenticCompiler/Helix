from __future__ import annotations

from fastapi import Request


async def authorize_request(request: Request) -> None:
    """Placeholder authorization hook.

    Currently allows all requests. Future versions may inspect
    the request for tokens or signatures.
    """
    pass
