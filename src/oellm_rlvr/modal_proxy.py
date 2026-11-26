"""Route only the Modal client through an HTTP CONNECT proxy.

Leonardo compute nodes have no internet. Modal (>= 1.5) can tunnel its gRPC and blob traffic through a proxy, but it
reads the process-wide HTTP(S)_PROXY variables, which Ray, vLLM, the vLLM router and LiteLLM would also pick up for
their node-local traffic. Instead, the proxy URL is taken from ``OELLM_MODAL_PROXY`` and handed to Modal alone by
replacing ``modal._utils.proxy_support.get_proxy_url``. Modal imports that function at call time, so patching the
module attribute covers both the gRPC channel and the aiohttp blob session. Needs ``python-socks[asyncio]`` and
``aiohttp-socks`` (Modal's ``api-proxy-support`` extra).
"""

from __future__ import annotations

import importlib
import os

ENV_VAR = "OELLM_MODAL_PROXY"


def install_modal_proxy(proxy_url: str | None = None) -> str | None:
    """Make Modal use ``proxy_url`` (default: ``$OELLM_MODAL_PROXY``) for every connection. Returns the URL or None."""
    url = proxy_url if proxy_url is not None else os.environ.get(ENV_VAR, "")
    url = url.strip()
    if not url:
        return None
    if not url.startswith(("http://", "socks5://", "socks5h://", "socks4://", "socks4a://")):
        raise ValueError(f"{ENV_VAR} must be an http:// or socks URL, got {url!r}")
    proxy_support = importlib.import_module("modal._utils.proxy_support")
    for package in ("python_socks", "aiohttp_socks"):
        importlib.import_module(package)  # fail at start, not on the first sandbox

    def get_proxy_url(host: str | None, *, use_ssl: bool) -> str:
        return url

    proxy_support.get_proxy_url = get_proxy_url
    return url
