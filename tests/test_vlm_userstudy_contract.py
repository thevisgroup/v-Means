"""Repository contract for the standalone VLM user-study toolkit."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "vlm_userstudy"


class VlmUserStudyContractTests(unittest.TestCase):
    def test_standalone_module_contains_the_public_workflow(self):
        required_paths = {
            ".gitignore",
            "README.md",
            "config.py",
            "download_videos.sh",
            "push_to_sheet.py",
            "questionnaire.py",
            "requirements.txt",
            "runner.py",
            "score.py",
            "serve/glm45v.sh",
            "serve/glm46v.sh",
            "serve/internvl35_38b.sh",
            "serve/minicpmv45.sh",
            "serve/qwen35_9b.sh",
            "serve/qwen3vl_235b_fp8.sh",
            "serve/qwen3vl_8b.sh",
            "tests/test_vlm_userstudy.py",
            "videos/.gitkeep",
        }
        missing = sorted(
            path for path in required_paths if not (MODULE / path).is_file()
        )
        self.assertEqual(missing, [], f"missing standalone module files: {missing}")
        self.assertFalse((MODULE / ".git").exists(), "nested Git metadata was copied")

    def test_model_registry_is_multi_model_and_scripts_are_present(self):
        config_path = MODULE / "config.py"
        spec = importlib.util.spec_from_file_location("vlm_contract_config", config_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)

        expected_tags = {
            "glm-4.6v",
            "internvl3.5-38b",
            "minicpm-v-4.5",
            "qwen3.5-9b",
            "qwen3vl-8b",
        }
        self.assertEqual(set(config.MODELS), expected_tags)
        for tag, model in config.MODELS.items():
            with self.subTest(tag=tag):
                self.assertTrue((MODULE / model["serve_script"]).is_file())
        self.assertNotIn("glm-4.5v", config.MODELS)
        self.assertNotIn("qwen3vl-235b-fp8", config.MODELS)

    def test_docs_and_ignore_rules_keep_the_toolkit_independent(self):
        root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
        module_readme = (MODULE / "README.md").read_text(encoding="utf-8")
        module_ignore = (MODULE / ".gitignore").read_text(encoding="utf-8")
        ignore_lines = set(module_ignore.splitlines())

        self.assertIn("vlm_userstudy", root_readme)
        self.assertIn("independent", module_readme.lower())
        self.assertIn("OpenAI-compatible", module_readme)
        self.assertIn("6d62fc2fba430a02e0496fa08f4c2c4fc632bb29", module_readme)
        for ignored in ("outputs/", ".env", "*.json", "videos/*"):
            with self.subTest(ignored=ignored):
                self.assertIn(ignored, ignore_lines)


if __name__ == "__main__":
    unittest.main()
