from __future__ import annotations

import sys
import types

import pytest

from oellm_rlvr.modal_proxy import install_modal_proxy


@pytest.fixture
def fake_modal(monkeypatch):
    proxy_support = types.ModuleType("modal._utils.proxy_support")
    proxy_support.get_proxy_url = lambda host, *, use_ssl: None
    for name, module in {
        "modal": types.ModuleType("modal"),
        "modal._utils": types.ModuleType("modal._utils"),
        "modal._utils.proxy_support": proxy_support,
        "python_socks": types.ModuleType("python_socks"),
        "aiohttp_socks": types.ModuleType("aiohttp_socks"),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return proxy_support


def test_unset_leaves_modal_alone(fake_modal, monkeypatch) -> None:
    monkeypatch.delenv("OELLM_MODAL_PROXY", raising=False)
    assert install_modal_proxy() is None
    assert fake_modal.get_proxy_url("api.modal.com", use_ssl=True) is None


def test_env_url_is_used_for_every_host(fake_modal, monkeypatch) -> None:
    monkeypatch.setenv("OELLM_MODAL_PROXY", "http://127.0.0.1:4321")
    assert install_modal_proxy() == "http://127.0.0.1:4321"
    assert fake_modal.get_proxy_url("api.modal.com", use_ssl=True) == "http://127.0.0.1:4321"
    assert fake_modal.get_proxy_url(None, use_ssl=True) == "http://127.0.0.1:4321"


def test_rejects_non_proxy_url(fake_modal) -> None:
    with pytest.raises(ValueError):
        install_modal_proxy("127.0.0.1:4321")


def test_missing_socks_package_fails_early(fake_modal, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "aiohttp_socks", None)
    with pytest.raises(ImportError):
        install_modal_proxy("http://127.0.0.1:4321")
