import tempfile
import unittest
from pathlib import Path
from unittest import mock

from llama_deploy.config import BackendKind, Config, ModelSpec, NetworkConfig
from llama_deploy.service import write_compose, write_models_ini


def _resolved(spec: ModelSpec, filename: str) -> ModelSpec:
    return spec.with_resolved(filename=filename, sha256="a" * 64, size=1234)


def _cfg(base_dir: Path, *, models_max: int = 1) -> Config:
    return Config(
        base_dir=base_dir,
        backend=BackendKind.CPU,
        network=NetworkConfig(bind_host="127.0.0.1", port=8080, publish=True),
        swap_gib=8,
        models_max=models_max,
        parallel=1,
        api_token=None,
        api_token_name="default",
        hf_token=None,
        skip_download=False,
        llm=ModelSpec(
            hf_repo="Qwen/Qwen3-8B-GGUF",
            candidate_patterns=["Q4_K_M"],
            ctx_len=3072,
        ),
        emb=ModelSpec(
            hf_repo="Qwen/Qwen3-Embedding-0.6B-GGUF",
            candidate_patterns=["Q8_0"],
            ctx_len=2048,
            is_embedding=True,
        ),
    )


class ServiceModelsIniTests(unittest.TestCase):
    def test_models_max_one_disables_global_and_embedding_startup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            preset_path = Path(td) / "models.ini"
            llm = _resolved(
                ModelSpec(
                    hf_repo="bartowski/Phi-3.5-mini-instruct-GGUF",
                    candidate_patterns=["Q4_K_M"],
                    ctx_len=3072,
                ),
                "phi.gguf",
            )
            emb = _resolved(
                ModelSpec(
                    hf_repo="Qwen/Qwen3-Embedding-0.6B-GGUF",
                    candidate_patterns=["Q8_0"],
                    ctx_len=2048,
                    is_embedding=True,
                ),
                "emb.gguf",
            )

            with mock.patch("llama_deploy.system.log_line"):
                write_models_ini(preset_path, llm, emb, parallel=1, models_max=1)
            content = preset_path.read_text(encoding="utf-8")

            self.assertIn("[*]", content)
            self.assertIn("load-on-startup = false", content)
            self.assertIn(f"[{llm.effective_alias}]", content)
            self.assertIn(f"[{emb.effective_alias}]", content)
            self.assertIn("model = /models/phi.gguf", content)
            self.assertIn("model = /models/emb.gguf", content)

            # With models_max=1, neither model family is preloaded at startup.
            self.assertIn(
                f"[{llm.effective_alias}]\nmodel = /models/phi.gguf\nload-on-startup = false",
                content,
            )
            self.assertIn(
                f"[{emb.effective_alias}]\nmodel = /models/emb.gguf\nload-on-startup = false",
                content,
            )

    def test_models_max_two_allows_embedding_startup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            preset_path = Path(td) / "models.ini"
            llm = _resolved(
                ModelSpec(
                    hf_repo="Qwen/Qwen3-8B-GGUF",
                    candidate_patterns=["Q4_K_M"],
                    ctx_len=3072,
                ),
                "llm.gguf",
            )
            emb = _resolved(
                ModelSpec(
                    hf_repo="Qwen/Qwen3-Embedding-0.6B-GGUF",
                    candidate_patterns=["Q8_0"],
                    ctx_len=2048,
                    is_embedding=True,
                ),
                "emb.gguf",
            )

            with mock.patch("llama_deploy.system.log_line"):
                write_models_ini(preset_path, llm, emb, parallel=2, models_max=2)
            content = preset_path.read_text(encoding="utf-8")
            self.assertIn(
                f"[{emb.effective_alias}]\nmodel = /models/emb.gguf\nload-on-startup = true",
                content,
            )

    def test_compose_does_not_pass_models_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base_dir = Path(td)
            compose_path = base_dir / "docker-compose.yml"
            cfg = _cfg(base_dir, models_max=1)

            with mock.patch("llama_deploy.system.log_line"):
                write_compose(compose_path, cfg)
            content = compose_path.read_text(encoding="utf-8")

            self.assertIn("--models-preset /presets/models.ini", content)
            self.assertNotIn("--models-dir /models", content)


if __name__ == "__main__":
    unittest.main()
