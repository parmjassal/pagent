import httpx
import logging
import sys
from typing import Any, Dict

logger = logging.getLogger(__name__)

def on_response_hook(response: httpx.Response):
    """
    Hook to detect 3xx redirects (common in corporate proxies/captive portals).
    Outputs the redirect link to stdout for user visibility.
    """
    if response.is_redirect:
        redirect_url = response.headers.get("Location")
        msg = f"\n[PROXY DETECTED] Request redirected to: {redirect_url}"
        print(msg, file=sys.stderr)
        logger.warning(f"HTTP Redirect detected: {redirect_url}")

def get_platform_http_client() -> httpx.Client:
    """
    Returns an httpx Client configured for the platform.
    Disables automatic redirect following to capture and report them.
    """
    return httpx.Client(
        follow_redirects=False, # We want to catch and report redirects
        event_hooks={'response': [on_response_hook]}
    )
