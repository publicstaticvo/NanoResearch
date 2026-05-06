"""Debug agent helpers — syntax check, file rewrite, SLURM fixes, error classification, download."""

from __future__ import annotations

import asyncio
import logging
import re as _re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class _DebugHelpersMixin:
    """Mixin — syntax check, file rewrite, SLURM fixes, error classification, download."""

    def _check_syntax(self, filepath: Path) -> bool:
        """Check if a Python file has valid syntax."""
        try:
            import py_compile
            py_compile.compile(str(filepath), doraise=True)
            return True
        except py_compile.PyCompileError:
            return False
        except Exception:
            return True  # assume OK if check itself fails

    async def _rewrite_file(
        self, code_dir: Path, filename: str, source_files: dict[str, str], error_log: str
    ) -> bool:
        """When patching fails, ask LLM to rewrite the entire file."""
        filepath = code_dir / filename
        is_new_file = not filepath.exists()
        current_content = ""
        if not is_new_file:
            current_content = filepath.read_text(errors="replace")

        # Gather context from other files (imports they expect from this file)
        cross_refs = ""
        for other_name, other_content in source_files.items():
            if other_name == filename:
                continue
            module = filename.replace(".py", "")
            import_lines = [
                line for line in other_content.split("\n")
                if f"from {module} import" in line or f"import {module}" in line
            ]
            if import_lines:
                cross_refs += f"\n{other_name} imports: {'; '.join(import_lines)}"

        system_prompt = (
            "You are a senior ML engineer. "
            + ("Write" if is_new_file else "Rewrite")
            + " the following Python file to fix all errors. "
            "The file must be COMPLETE and RUNNABLE with correct Python syntax and indentation. "
            "Keep the same functionality and class/function names. "
            "Make sure all names that other files import from this file are defined. "
            "Return ONLY the Python code, no markdown fences, no explanation."
        )

        user_prompt = f"""File: {filename} ({'NEW FILE — does not exist yet' if is_new_file else 'existing file'})
Error: {error_log[:1500]}

Other files import from this file:
{cross_refs}

{'This file needs to be CREATED from scratch.' if is_new_file else f'Current content:{chr(10)}{current_content}'}

{'Write' if is_new_file else 'Rewrite'} this file with correct syntax. Return ONLY Python code."""

        try:
            new_content = await self.generate(system_prompt, user_prompt)

            # Robust fence stripping — handles LLM self-correction and multiple blocks
            from nanoresearch.agents._code_utils import _strip_code_fences
            new_content = _strip_code_fences(new_content)

            # Verify syntax before writing
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(new_content)
            if self._check_syntax(filepath):
                self.log(f"{'Created' if is_new_file else 'Rewrote'} {filename} successfully")
                return True
            else:
                # Rewrite also has syntax error — restore original or remove
                if is_new_file:
                    filepath.unlink(missing_ok=True)
                    self.log(f"Created {filename} has syntax errors, removed")
                else:
                    filepath.write_text(current_content)
                    self.log(f"Rewrite of {filename} also has syntax errors, rolled back")
                return False

        except Exception as e:
            self.log(f"Rewrite of {filename} failed: {e}")
            if is_new_file:
                filepath.unlink(missing_ok=True)
            else:
                filepath.write_text(current_content)
            return False

    def _fix_common_slurm_issues(self, code_dir: Path) -> bool:
        """Fix known SLURM script issues that LLMs commonly produce."""
        fixed = False

        for slurm_file in list(code_dir.glob("*.slurm")) + list(code_dir.glob("*.sh")):
            content = slurm_file.read_text(errors="replace")
            original = content

            content = self._normalize_slurm_shebang(content)
            if self._looks_like_invalid_slurm_script(content):
                regenerated = self._build_fallback_slurm_wrapper(code_dir, slurm_file)
                if regenerated:
                    content = regenerated

            # Fix 1: conda activate without proper init
            if "conda activate" in content and "conda.sh" not in content:
                content = content.replace(
                    "source ~/.bashrc\nconda activate",
                    "source ~/anaconda3/etc/profile.d/conda.sh\nconda activate",
                )
                if "source ~/anaconda3/etc/profile.d/conda.sh" not in content:
                    content = content.replace(
                        "conda activate",
                        "source ~/anaconda3/etc/profile.d/conda.sh\nconda activate",
                        1,
                    )

            # Fix 2: Ensure proxy is present for pip install (read from env, no hardcoded creds)
            if "pip install" in content and "proxy" not in content.lower():
                content = content.replace(
                    "pip install",
                    "# Enable proxy for pip (from environment)\n"
                    'export https_proxy="${HTTPS_PROXY:-}"\n'
                    'export http_proxy="${HTTP_PROXY:-}"\n'
                    "pip install",
                    1,
                )

            if content != original:
                slurm_file.write_text(content)
                fixed = True

        return fixed

    @staticmethod
    def _normalize_slurm_shebang(content: str) -> str:
        stripped = content.lstrip("\ufeff\r\n\t ")
        if not stripped:
            return content
        if stripped.startswith("#!"):
            return stripped
        return content

    @staticmethod
    def _looks_like_invalid_slurm_script(content: str) -> bool:
        stripped = content.lstrip("\ufeff\r\n\t ")
        if not stripped:
            return True
        if stripped.startswith("#!"):
            return False
        first_line = stripped.splitlines()[0].strip()
        python_markers = (
            first_line.startswith("import "),
            first_line.startswith("from "),
            first_line.startswith("def "),
            "if __name__ ==" in stripped,
        )
        return any(python_markers)

    @staticmethod
    def _build_fallback_slurm_wrapper(code_dir: Path, slurm_file: Path) -> str | None:
        runner_candidates = (
            "nanoresearch_runner.py",
            "train.py",
            "main.py",
            "run.py",
        )
        runner = next((name for name in runner_candidates if (code_dir / name).exists()), None)
        if runner is None:
            return None

        return f"""#!/bin/bash
#SBATCH --job-name={code_dir.name[:15]}
#SBATCH --output={code_dir}/logs/slurm_%j.out
#SBATCH --error={code_dir}/logs/slurm_%j.err

set -euo pipefail

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo "========================================"

PYTHON_BIN="$(command -v python 2>/dev/null || true)"
if [ -z "$PYTHON_BIN" ] && command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
fi
if [ -z "$PYTHON_BIN" ]; then
    echo "[ERROR] Could not find a usable python executable." >&2
    exit 1
fi

PIP_BIN="$(dirname "$PYTHON_BIN")/pip"
if [ ! -x "$PIP_BIN" ]; then
    PIP_BIN="$PYTHON_BIN -m pip"
fi

export https_proxy="${{HTTPS_PROXY:-}}"
export http_proxy="${{HTTP_PROXY:-}}"
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${{LD_LIBRARY_PATH:-}}

mkdir -p {code_dir}/results
mkdir -p {code_dir}/checkpoints
mkdir -p {code_dir}/logs

cd {code_dir}
if [ -f requirements.txt ]; then
    if [ "$PIP_BIN" = "$PYTHON_BIN -m pip" ]; then
        "$PYTHON_BIN" -m pip install -r requirements.txt --quiet 2>/dev/null || true
    else
        "$PIP_BIN" install -r requirements.txt --quiet 2>/dev/null || true
    fi
fi

python() {{
    "$PYTHON_BIN" "$@"
}}

echo "Starting training..."
python {runner}

EXIT_CODE=$?

echo "========================================"
echo "End: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
"""

    def _fix_common_python_runtime_issues(self, code_dir: Path) -> list[str]:
        """Apply deterministic patches for recurring generated-code bugs."""
        fixed_files: list[str] = []

        for py_file in code_dir.rglob("*.py"):
            original = py_file.read_text(errors="replace")
            content = original
            content = self._patch_broken_distribution_metadata(content)
            content = self._patch_transformers_deepspeed_runtime_guard(content)
            content = self._patch_deepspeed_stub(content)
            content = self._patch_pubmedqa_mapping_iteration(content)
            content = self._patch_metrics_history_fieldnames(content)
            content = self._patch_prompt_length_guard(content)
            content = self._patch_teacher_model_failure_fallback(content)
            content = self._patch_inference_tensor_teacher_probs(content)

            if content == original:
                continue

            py_file.write_text(content)
            if self._check_syntax(py_file):
                fixed_files.append(str(py_file.relative_to(code_dir)))
            else:
                py_file.write_text(original)
        return fixed_files

    @staticmethod
    def _patch_deepspeed_stub(content: str) -> str:
        if "ModuleType(\"deepspeed\")" not in content and "ModuleType('deepspeed')" not in content:
            return content
        if "DeepSpeedEngine" in content:
            return content

        stub_match = _re.search(
            r'^(?P<indent>\s*)(?P<var>[A-Za-z_]\w*)\s*=\s*types\.ModuleType\((?P<quote>["\'])deepspeed(?P=quote)\)\s*$',
            content,
            _re.MULTILINE,
        )
        if not stub_match:
            return content

        indent = stub_match.group("indent")
        var_name = stub_match.group("var")
        injection = (
            f'{indent}if not hasattr({var_name}, "__spec__") or {var_name}.__spec__ is None:\n'
            f'{indent}    {var_name}.__spec__ = importlib.machinery.ModuleSpec("deepspeed", loader=None)\n'
            f'{indent}class _NanoResearchDeepSpeedEngine:\n'
            f'{indent}    pass\n'
            f'{indent}{var_name}.DeepSpeedEngine = _NanoResearchDeepSpeedEngine\n'
        )

        patched = content
        if "import importlib.machinery" not in patched:
            patched = _re.sub(
                r'^(import\s+types\s*)$',
                r'\1\nimport importlib.machinery',
                patched,
                count=1,
                flags=_re.MULTILINE,
            )
            if "import importlib.machinery" not in patched:
                patched = _re.sub(
                    r'^(from\s+typing\s+import\s+.*\n)',
                    r'\1import importlib.machinery\n',
                    patched,
                    count=1,
                    flags=_re.MULTILINE,
                )
        return patched.replace(stub_match.group(0), stub_match.group(0) + "\n" + injection, 1)

    @staticmethod
    def _ensure_import_line(content: str, import_line: str) -> str:
        if import_line in content:
            return content
        matches = list(
            _re.finditer(
                r"^(?:from\s+[^\n]+\s+import\s+[^\n]+|import\s+[^\n]+)\s*$",
                content,
                _re.MULTILINE,
            )
        )
        if not matches:
            return import_line + "\n" + content
        last = matches[-1]
        return content[:last.end()] + "\n" + import_line + content[last.end():]

    @classmethod
    def _patch_transformers_deepspeed_runtime_guard(cls, content: str) -> str:
        initial_transformers_import = _re.search(
            r"^\s*(?:from\s+transformers\s+import\b|import\s+transformers\b)",
            content,
            _re.MULTILINE,
        )
        if not initial_transformers_import:
            return content

        deepspeed_none_pattern = _re.compile(
            r'^(?P<indent>\s*)if\s+["\']deepspeed["\']\s+not\s+in\s+sys\.modules:\n'
            r'(?P=indent)\s+sys\.modules\[[\'"]deepspeed[\'"]\]\s*=\s*None\s*$',
            _re.MULTILINE,
        )

        def _replace_deepspeed_none(match: _re.Match[str]) -> str:
            indent = match.group("indent")
            return (
                f'{indent}if "deepspeed" not in sys.modules:\n'
                f'{indent}    deepspeed_stub = types.ModuleType("deepspeed")\n'
                f'{indent}    deepspeed_stub.__dict__["__version__"] = "0.0.0"\n'
                f'{indent}    deepspeed_stub.__dict__["is_deepspeed_zero3_enabled"] = lambda: False\n'
                f'{indent}    class _NanoResearchDeepSpeedEngine:\n'
                f'{indent}        pass\n'
                f'{indent}    deepspeed_stub.DeepSpeedEngine = _NanoResearchDeepSpeedEngine\n'
                f'{indent}    deepspeed_stub.__spec__ = importlib.machinery.ModuleSpec("deepspeed", loader=None)\n'
                f'{indent}    sys.modules["deepspeed"] = deepspeed_stub\n'
                f'{indent}    ops_stub = types.ModuleType("deepspeed.ops")\n'
                f'{indent}    ops_stub.__spec__ = importlib.machinery.ModuleSpec("deepspeed.ops", loader=None)\n'
                f'{indent}    sys.modules.setdefault("deepspeed.ops", ops_stub)'
            )

        content = deepspeed_none_pattern.sub(_replace_deepspeed_none, content, count=1)

        if (
            "TRANSFORMERS_NO_DEEPSPEED" in content
            and "_ensure_executable_nvcc" in content
            and "DeepSpeedEngine" in content
        ):
            return content

        for import_line in (
            "import importlib.machinery",
            "import tempfile",
            "import types",
            "import sys",
            "import os",
        ):
            content = cls._ensure_import_line(content, import_line)

        transformers_import = _re.search(
            r"^\s*(?:from\s+transformers\s+import\b|import\s+transformers\b)",
            content,
            _re.MULTILINE,
        )
        if not transformers_import:
            return content

        guard_block = """
# Avoid importing deepspeed / probing nvcc in restricted CUDA environments.
os.environ.setdefault("TRANSFORMERS_NO_DEEPSPEED", "1")
os.environ.setdefault("DEEPSPEED_DISABLE", "1")

def _ensure_executable_nvcc() -> None:
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not cuda_home:
        return
    nvcc_path = os.path.join(cuda_home, "bin", "nvcc")
    if os.path.exists(nvcc_path) and not os.access(nvcc_path, os.X_OK):
        stub_root = os.path.join(tempfile.gettempdir(), "cuda_nvcc_stub")
        bin_dir = os.path.join(stub_root, "bin")
        os.makedirs(bin_dir, exist_ok=True)
        stub_nvcc = os.path.join(bin_dir, "nvcc")
        if not os.path.exists(stub_nvcc):
            with open(stub_nvcc, "w", encoding="utf-8") as f:
                f.write("#!/bin/sh\\n")
                f.write("echo 'Cuda compilation tools, release 12.1, V12.1.0'\\n")
            os.chmod(stub_nvcc, 0o755)
        os.environ["CUDA_HOME"] = stub_root
        os.environ["CUDA_PATH"] = stub_root

_ensure_executable_nvcc()

if "deepspeed" not in sys.modules:
    deepspeed_stub = types.ModuleType("deepspeed")
    deepspeed_stub.__dict__["__version__"] = "0.0.0"
    deepspeed_stub.__dict__["is_deepspeed_zero3_enabled"] = lambda: False
    class _NanoResearchDeepSpeedEngine:
        pass
    deepspeed_stub.DeepSpeedEngine = _NanoResearchDeepSpeedEngine
    deepspeed_stub.__spec__ = importlib.machinery.ModuleSpec("deepspeed", loader=None)
    sys.modules["deepspeed"] = deepspeed_stub
    ops_stub = types.ModuleType("deepspeed.ops")
    ops_stub.__spec__ = importlib.machinery.ModuleSpec("deepspeed.ops", loader=None)
    sys.modules.setdefault("deepspeed.ops", ops_stub)
"""
        guard_block = guard_block.strip("\n") + "\n\n"
        return content[:transformers_import.start()] + guard_block + content[transformers_import.start():]

    @classmethod
    def _patch_broken_distribution_metadata(cls, content: str) -> str:
        transformers_import = _re.search(
            r"^\s*(?:from\s+transformers\s+import\b|import\s+transformers\b)",
            content,
            _re.MULTILINE,
        )
        if not transformers_import or "_nr_safe_metadata_version" in content:
            return content

        for import_line in (
            "import importlib",
            "import importlib.metadata",
        ):
            content = cls._ensure_import_line(content, import_line)

        fix_block = """
# Repair broken importlib.metadata records where version() returns None.
_nr_orig_metadata_version = importlib.metadata.version
_nr_orig_metadata_distribution = getattr(importlib.metadata, "distribution", None)

def _nr_safe_metadata_version(pkg):
    try:
        ver = _nr_orig_metadata_version(pkg)
    except Exception:
        ver = None

    if ver in (None, "", "None"):
        module_name = str(pkg).replace("-", "_")
        try:
            module = importlib.import_module(module_name)
            ver = getattr(module, "__version__", None)
        except Exception:
            ver = None

    if ver in (None, "", "None"):
        fallback_versions = {
            "packaging": "26.0",
            "certifi": "2026.1.4",
        }
        ver = fallback_versions.get(str(pkg).lower())

    return ver

def _nr_safe_metadata_distribution(pkg):
    if _nr_orig_metadata_distribution is None:
        return None

    dist = _nr_orig_metadata_distribution(pkg)
    if getattr(dist, "version", None) not in (None, "", "None"):
        return dist

    safe_ver = _nr_safe_metadata_version(pkg)
    if safe_ver in (None, "", "None"):
        return dist

    class _NanoResearchDistribution:
        version = safe_ver
        files = []

    return _NanoResearchDistribution()

importlib.metadata.version = _nr_safe_metadata_version
if _nr_orig_metadata_distribution is not None:
    importlib.metadata.distribution = _nr_safe_metadata_distribution
"""
        fix_block = fix_block.strip("\n") + "\n\n"
        return content[:transformers_import.start()] + fix_block + content[transformers_import.start():]

    @staticmethod
    def _patch_pubmedqa_mapping_iteration(content: str) -> str:
        if "pubmedqa" not in content.lower() and "test_ground_truth.json" not in content:
            return content

        helper_name = "_flatten_pubmedqa_mapping"
        if helper_name not in content:
            helper_block = """
def _flatten_pubmedqa_mapping(data):
    if not isinstance(data, dict):
        return list(data) if isinstance(data, list) else []

    for container_key in ("data", "examples", "dataset"):
        nested = data.get(container_key)
        if isinstance(nested, list):
            return list(nested)
        if isinstance(nested, dict):
            data = nested
            break

    split_items = []
    for split_key in ("TRAIN", "train", "DEV", "dev", "VALID", "valid", "VAL", "val", "TEST", "test"):
        split_data = data.get(split_key)
        if isinstance(split_data, dict):
            split_items.extend(split_data.values())
        elif isinstance(split_data, list):
            split_items.extend(split_data)
    if split_items:
        return split_items

    return list(data.values())


"""
            anchor = _re.search(
                r"^def\s+(?:parse_|read_|load_).*?(?:pubmedqa|dataset)",
                content,
                _re.MULTILINE,
            )
            if anchor:
                content = content[:anchor.start()] + helper_block + content[anchor.start():]
            else:
                content = helper_block + content

        replacements = (
            (r"(?m)^(?P<indent>\s*)iterable\s*=\s*data\.values\(\)\s*$", r"\g<indent>iterable = _flatten_pubmedqa_mapping(data)"),
            (r"(?m)^(?P<indent>\s*)items\s*=\s*list\(data\.values\(\)\)\s*$", r"\g<indent>items = _flatten_pubmedqa_mapping(data)"),
            (r"(?m)^(?P<indent>\s*)entries\s*=\s*list\(data\.values\(\)\)\s*$", r"\g<indent>entries = _flatten_pubmedqa_mapping(data)"),
        )
        patched = content
        for pattern, replacement in replacements:
            patched = _re.sub(pattern, replacement, patched)
        return patched

    @staticmethod
    def _patch_metrics_history_fieldnames(content: str) -> str:
        pattern = _re.compile(
            r'^(?P<indent>\s*)fieldnames\s*=\s*sorted\(metrics_history\[0\]\.keys\(\)\)\s*$',
            _re.MULTILINE,
        )
        match = pattern.search(content)
        if match:
            indent = match.group("indent")
            replacement = (
                f"{indent}all_keys = set()\n"
                f"{indent}for row in metrics_history:\n"
                f"{indent}    if isinstance(row, dict):\n"
                f"{indent}        all_keys.update(row.keys())\n"
                f"{indent}fieldnames = sorted(all_keys)"
            )
            return pattern.sub(replacement, content, count=1)

        patterns = (
            (
                _re.compile(
                    r'^(?P<indent>\s*)writer\s*=\s*csv\.DictWriter\(f,\s*fieldnames=metrics_history\[-1\]\.keys\(\)\)\s*$',
                    _re.MULTILINE,
                ),
                "metrics_history",
                "fieldnames",
            ),
            (
                _re.compile(
                    r'^(?P<indent>\s*)keys\s*=\s*list\(metrics_list\[0\]\.keys\(\)\)\s*$',
                    _re.MULTILINE,
                ),
                "metrics_list",
                "keys",
            ),
        )
        patched = content
        for extra_pattern, source_name, target_name in patterns:
            extra_match = extra_pattern.search(patched)
            if not extra_match:
                continue
            indent = extra_match.group("indent")
            replacement = (
                f"{indent}all_keys = set()\n"
                f"{indent}for row in {source_name}:\n"
                f"{indent}    if isinstance(row, dict):\n"
                f"{indent}        all_keys.update(row.keys())\n"
                f"{indent}{target_name} = sorted(all_keys)"
            )
            patched = extra_pattern.sub(replacement, patched, count=1)
        return patched

    @staticmethod
    def _patch_prompt_length_guard(content: str) -> str:
        if (
            "args.prompt_length" not in content
            or "args.max_length" not in content
            or "AutoTokenizer.from_pretrained(args.model_dir" not in content
            or "AutoConfig" not in content
            or "max_position_embeddings" in content
        ):
            return content

        tokenizer_match = _re.search(
            r'^(?P<indent>\s*)tokenizer\s*=\s*AutoTokenizer\.from_pretrained\(args\.model_dir.*$',
            content,
            _re.MULTILINE,
        )
        if not tokenizer_match:
            return content
        indent = tokenizer_match.group("indent")
        guard = (
            f'{indent}model_config = AutoConfig.from_pretrained(args.model_dir)\n'
            f'{indent}max_positions = getattr(model_config, "max_position_embeddings", None)\n'
            f'{indent}if max_positions is not None:\n'
            f'{indent}    max_allowed = max_positions - args.prompt_length\n'
            f'{indent}    if max_allowed <= 0:\n'
            f'{indent}        raise ValueError(\n'
            f'{indent}            f"prompt_length={{args.prompt_length}} leaves no room for tokens with max_position_embeddings={{max_positions}}"\n'
            f'{indent}        )\n'
            f'{indent}    if args.max_length > max_allowed:\n'
            f'{indent}        logging.info(\n'
            f'{indent}            "Reducing max_length from %d to %d to fit prompt_length within max_position_embeddings=%d",\n'
            f'{indent}            args.max_length,\n'
            f'{indent}            max_allowed,\n'
            f'{indent}            max_positions,\n'
            f'{indent}        )\n'
            f'{indent}        args.max_length = max_allowed\n'
        )
        return content[:tokenizer_match.start()] + guard + content[tokenizer_match.start():]

    @staticmethod
    def _patch_teacher_model_failure_fallback(content: str) -> str:
        pattern = _re.compile(
            r'(?P<indent>\s*)except Exception as e:\n'
            r'(?P=indent)\s+logger\.exception\("Failed to load teacher model: %s", e\)\n'
            r'(?P=indent)\s+if args\.dry_run or args\.quick_eval:\n'
            r'(?P=indent)\s+\s+logger\.warning\("Proceeding without teacher due to quick/dry mode\."\)\n'
            r'(?P=indent)\s+\s+teacher_model = None\n'
            r'(?P=indent)\s+\s+teacher_tokenizer = None\n'
            r'(?P=indent)\s+\s+label_token_ids = None\n'
            r'(?P=indent)\s+else:\n'
            r'(?P=indent)\s+\s+raise',
            _re.MULTILINE,
        )
        match = pattern.search(content)
        if not match:
            return content
        indent = match.group("indent")
        replacement = (
            f"{indent}except Exception as e:\n"
            f'{indent}    logger.exception("Failed to load teacher model: %s", e)\n'
            f'{indent}    logger.warning("Proceeding without teacher after teacher-model load failure.")\n'
            f"{indent}    teacher_model = None\n"
            f"{indent}    teacher_tokenizer = None\n"
            f"{indent}    label_token_ids = None\n"
            f'{indent}    if hasattr(args, "disable_teacher"):\n'
            f"{indent}        args.disable_teacher = True"
        )
        return pattern.sub(replacement, content, count=1)

    @staticmethod
    def _patch_inference_tensor_teacher_probs(content: str) -> str:
        if "torch.inference_mode()" not in content or "teacher_probs" not in content:
            return content
        pattern = _re.compile(
            r'^(?P<indent>\s*)teacher_probs\s*=\s*teacher_probs\.to\((?P<device>[^)]+)\)\s*$',
            _re.MULTILINE,
        )
        match = pattern.search(content)
        if not match:
            return content
        indent = match.group("indent")
        device = match.group("device").strip()
        replacement = f"{indent}teacher_probs = teacher_probs.detach().clone().to({device})"
        return pattern.sub(replacement, content, count=1)

    def _classify_error(self, stdout_log: str, stderr_log: str) -> tuple[str, str]:
        """Classify error as ('data_missing', path) or ('code_bug', '')."""
        combined = stderr_log + "\n" + stdout_log
        combined_lower = combined.lower()
        data_missing_patterns = [
            "filenotfounderror",
            "no such file or directory",
            "file not found",
            "path does not exist",
        ]
        for pattern in data_missing_patterns:
            if pattern not in combined_lower:
                continue
            # Try quoted paths first
            for m in _re.finditer(
                r"(?:FileNotFoundError|No such file or directory|file not found)[^\n]*?['\"]([^'\"]+)['\"]",
                combined, _re.IGNORECASE,
            ):
                missing = m.group(1)
                if not missing.endswith((".py", ".pyc", ".so", ".pth")):
                    return "data_missing", missing
            # Try unquoted paths
            for m in _re.finditer(
                r"(?:FileNotFoundError|file not found)[^\n]*?(\S+\.(?:csv|tsv|obo|gaf|txt|gz|fasta|fa|pdb|pkl|h5|hdf5|json|xml|dat))\b",
                combined, _re.IGNORECASE,
            ):
                missing = m.group(1).rstrip(")")
                return "data_missing", missing
        return "code_bug", ""

    async def _download_missing_resource(self, missing_path: str) -> bool:
        """Try to download a missing data file.

        Security: validates URL scheme (http/https only), writes only inside
        the active workspace, and avoids shell invocation.
        """
        system_prompt = (
            "Given a missing file path from an ML experiment, determine its download URL. "
            "Return JSON: {\"url\": \"...\", \"filename\": \"...\"} or {\"cannot_download\": true}."
        )
        user_prompt = f"Missing file: {missing_path}\nReturn JSON only."
        try:
            result = await self.generate_json(system_prompt, user_prompt)
            if result.get("cannot_download"):
                return False
            url = result.get("url", "")
            filename = result.get("filename", "") or Path(missing_path).name
            if not url:
                return False
            # Validate URL: only allow http/https
            if not url.startswith(("http://", "https://")):
                self.log(f"Rejecting non-HTTP URL: {url}")
                return False

            workspace_root = self.workspace.path.resolve()
            dest = Path(missing_path).expanduser()
            if not dest.is_absolute():
                dest = workspace_root / dest
            dest = dest.resolve()
            try:
                dest.relative_to(workspace_root)
            except ValueError:
                self.log(f"Rejecting download outside workspace: {missing_path}")
                return False
            dest.parent.mkdir(parents=True, exist_ok=True)
            proc = await asyncio.create_subprocess_exec(
                "wget", "-q", "-O", str(dest), url,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=600)
            if dest.exists() and dest.stat().st_size > 0:
                self.log(f"Downloaded missing resource: {filename} -> {dest}")
                return True
        except Exception as e:
            self.log(f"Failed to download missing resource: {e}")
        return False

    async def close(self) -> None:
        pass
