import tempfile
import unittest
from pathlib import Path
from unittest import mock

from llama_deploy import nginx


class NginxConfigTests(unittest.TestCase):
    def test_write_local_config_with_auth_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sites_available = root / "sites-available"
            sites_enabled = root / "sites-enabled"
            sites_available.mkdir(parents=True, exist_ok=True)
            sites_enabled.mkdir(parents=True, exist_ok=True)

            captured = {}

            def _fake_write_file(path: Path, content: str, *, mode=None) -> None:
                captured["path"] = path
                captured["content"] = content
                captured["mode"] = mode

            with mock.patch.object(nginx, "_SITES_AVAILABLE", sites_available), \
                 mock.patch.object(nginx, "_SITES_ENABLED", sites_enabled), \
                 mock.patch("llama_deploy.system.write_file", side_effect=_fake_write_file), \
                 mock.patch.object(nginx, "sh"), \
                 mock.patch("pathlib.Path.symlink_to", return_value=None):
                nginx.write_nginx_local_config(
                    bind_host="127.0.0.1",
                    port=8080,
                    upstream_port=8081,
                    use_auth_sidecar=True,
                    sidecar_port=9000,
                )

            self.assertIn("auth_request /auth;", captured["content"])
            self.assertIn("proxy_pass         http://127.0.0.1:8081;", captured["content"])

    def test_ensure_local_proxy_selects_free_port_when_busy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sites_available = root / "sites-available"
            sites_enabled = root / "sites-enabled"
            sites_available.mkdir(parents=True, exist_ok=True)
            sites_enabled.mkdir(parents=True, exist_ok=True)

            with mock.patch.object(nginx, "_SITES_AVAILABLE", sites_available), \
                 mock.patch.object(nginx, "_SITES_ENABLED", sites_enabled), \
                 mock.patch.object(nginx, "_pick_free_bind_port", return_value=18080), \
                 mock.patch.object(nginx, "ensure_nginx"), \
                 mock.patch.object(nginx, "write_nginx_local_config") as write_local, \
                 mock.patch.object(nginx, "log_line"), \
                 mock.patch.object(nginx, "sh"):
                chosen = nginx.ensure_local_proxy(
                    bind_host="127.0.0.1",
                    port=8080,
                    upstream_port=8081,
                    configure_ufw=False,
                    use_auth_sidecar=True,
                    sidecar_port=9000,
                )

            self.assertEqual(chosen, 18080)
            write_local.assert_called_once_with(
                "127.0.0.1",
                18080,
                8081,
                use_auth_sidecar=True,
                sidecar_port=9000,
                webui_port=0,
            )

    def test_ensure_local_proxy_reuses_existing_site_port(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sites_available = root / "sites-available"
            sites_enabled = root / "sites-enabled"
            sites_available.mkdir(parents=True, exist_ok=True)
            sites_enabled.mkdir(parents=True, exist_ok=True)
            (sites_enabled / "llama_local_8080").write_text("# enabled", encoding="utf-8")

            with mock.patch.object(nginx, "_SITES_AVAILABLE", sites_available), \
                 mock.patch.object(nginx, "_SITES_ENABLED", sites_enabled), \
                 mock.patch.object(nginx, "_pick_free_bind_port") as pick_port, \
                 mock.patch.object(nginx, "ensure_nginx"), \
                 mock.patch.object(nginx, "write_nginx_local_config") as write_local, \
                 mock.patch.object(nginx, "log_line"), \
                 mock.patch.object(nginx, "sh"):
                chosen = nginx.ensure_local_proxy(
                    bind_host="127.0.0.1",
                    port=8080,
                    upstream_port=8081,
                    configure_ufw=False,
                    use_auth_sidecar=True,
                    sidecar_port=9000,
                )

            self.assertEqual(chosen, 8080)
            pick_port.assert_not_called()
            write_local.assert_called_once_with(
                "127.0.0.1",
                8080,
                8081,
                use_auth_sidecar=True,
                sidecar_port=9000,
                webui_port=0,
            )

    def test_local_config_with_webui_adds_chat_compat_redirect(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sites_available = root / "sites-available"
            sites_enabled = root / "sites-enabled"
            sites_available.mkdir(parents=True, exist_ok=True)
            sites_enabled.mkdir(parents=True, exist_ok=True)

            captured = {}

            def _fake_write_file(path: Path, content: str, *, mode=None) -> None:
                captured["path"] = path
                captured["content"] = content
                captured["mode"] = mode

            with mock.patch.object(nginx, "_SITES_AVAILABLE", sites_available), \
                 mock.patch.object(nginx, "_SITES_ENABLED", sites_enabled), \
                 mock.patch("llama_deploy.system.write_file", side_effect=_fake_write_file), \
                 mock.patch.object(nginx, "sh"), \
                 mock.patch("pathlib.Path.symlink_to", return_value=None):
                nginx.write_nginx_local_config(
                    bind_host="127.0.0.1",
                    port=8080,
                    upstream_port=8081,
                    use_auth_sidecar=True,
                    sidecar_port=9000,
                    webui_port=3000,
                )

            self.assertIn("location = /chat", captured["content"])
            self.assertIn("return 302 /;", captured["content"])
            self.assertIn("proxy_pass         http://127.0.0.1:3000;", captured["content"])


if __name__ == "__main__":
    unittest.main()
