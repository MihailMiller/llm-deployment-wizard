import unittest
from unittest import mock

from llama_deploy.health import wait_health


class _Resp:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class HealthTests(unittest.TestCase):
    def test_wait_health_adds_bearer_header_when_provided(self) -> None:
        seen_auth = {"value": None}

        def _fake_urlopen(req, timeout=3):
            seen_auth["value"] = req.get_header("Authorization")
            return _Resp(200)

        with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen), \
             mock.patch("llama_deploy.health.log_line"):
            wait_health("http://127.0.0.1:8080/health", timeout_s=1, bearer_token="tok")

        self.assertEqual(seen_auth["value"], "Bearer tok")


if __name__ == "__main__":
    unittest.main()
