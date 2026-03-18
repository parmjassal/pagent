import httpx
import logging
import sys
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_async_client: Optional[httpx.AsyncClient] = None

async def on_response_hook(response: httpx.Response):
    """
    Async hook to detect redirects and provide low-level debug logging.
    """
    logger.debug(f"HTTP Response: {response.status_code} {response.url}")
    
    # We only read and log the body if DEBUG is enabled to avoid performance overhead
    if logger.isEnabledFor(logging.DEBUG):
        try:
            await response.aread()
            logger.debug(f"HTTP Headers: {dict(response.headers)}")
            logger.debug(f"HTTP Body (first 500 chars): {response.text[:500]}...")
        except Exception as e:
            logger.debug(f"Could not read response body for debug: {e}")

    if response.is_redirect:
        redirect_url = response.headers.get("Location")
        msg = f"\n[PROXY DETECTED] Request redirected to: {redirect_url}"
        print(msg, file=sys.stderr)
        logger.warning(f"HTTP Redirect detected: {redirect_url}")


def get_platform_async_http_client() -> httpx.AsyncClient:
    """
    Returns a shared, async httpx Client configured for the platform.
    Disables automatic redirect following to capture and report them.
    """
    global _async_client
    if _async_client is None or _async_client.is_closed:
        logger.info("Creating new shared httpx.AsyncClient.")
        _async_client = httpx.AsyncClient(
            follow_redirects=False, # We want to catch and report redirects
            event_hooks={'response': [on_response_hook]}
        )
    return _async_client

async def close_platform_async_http_client():
    """Closes the shared async http client."""
    global _async_client
    if _async_client and not _async_client.is_closed:
        logger.info("Closing shared httpx.AsyncClient.")
        await _async_client.aclose()
        _async_client = None

# Keep the synchronous version for any parts of the codebase that may still need it.
def get_platform_http_client() -> httpx.Client:
    """
    Returns a synchronous httpx Client configured for the platform.
    """
    return httpx.Client(
        follow_redirects=False,
        event_hooks={'response': [on_response_hook]}
    )
