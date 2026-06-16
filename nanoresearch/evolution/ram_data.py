"""RAM SDPO data collector — records (x, y, o) interaction triples.

Each triple captures one RAM invocation and its downstream outcome:
  x = RAM input context
  y = RAM generated output
  o = downstream feedback (execution result / user comment)

Triples are stored as JSONL for later offline SDPO training.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from nanoresearch.paths import get_ram_data_dir

logger = logging.getLogger(__name__)


class RAMTriple(BaseModel):
    """One complete (x, y, o) interaction triple."""

    triple_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    subsystem: str  # "method_gen" | "code_impl" | "paper_writing"
    stage: str  # pipeline stage name
    x_context: str  # RAM input
    y_output: str = ""  # RAM generated text
    o_feedback: str = ""  # downstream feedback
    o_quality_signal: float = 0.0  # -1.0 ~ 1.0
    session_id: str = ""
    workspace_id: str = ""


class RAMDataCollector:
    """Collects (x, y, o) triples for SDPO training.

    Lifecycle per triple:
      1. ``record_input()``   — called when RAM receives input, returns triple_id
      2. ``record_output()``  — called after RAM generates output
      3. ``complete_triple()``— called after downstream stage finishes, with feedback

    Completed triples are appended to JSONL files at
    ``{root}/triples_{subsystem}.jsonl``.
    """

    def __init__(
        self,
        root: Path | None = None,
        *,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.root = root or get_ram_data_dir()
        # Pending triples (not yet completed with feedback)
        self._pending: dict[str, RAMTriple] = {}
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _make_id(subsystem: str, stage: str, context_prefix: str) -> str:
        h = hashlib.sha1(
            f"{subsystem}:{stage}:{context_prefix[:400]}:{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]
        return f"ram-{h}"

    def record_input(
        self,
        subsystem: str,
        stage: str,
        context: str,
        *,
        session_id: str = "",
        workspace_id: str = "",
    ) -> str:
        """Record the input side of a triple. Returns a triple_id."""
        if not self.enabled:
            return ""
        triple_id = self._make_id(subsystem, stage, context)
        triple = RAMTriple(
            triple_id=triple_id,
            subsystem=subsystem,
            stage=stage,
            x_context=context,
            session_id=session_id,
            workspace_id=workspace_id,
        )
        self._pending[triple_id] = triple
        return triple_id

    def record_output(self, triple_id: str, output: str) -> None:
        """Attach RAM's generated output to a pending triple."""
        if not self.enabled or not triple_id:
            return
        if triple_id in self._pending:
            self._pending[triple_id].y_output = output

    def complete_triple(
        self,
        triple_id: str,
        feedback: str,
        quality_signal: float,
    ) -> RAMTriple | None:
        """Attach downstream feedback, write completed triple to disk."""
        if not self.enabled or not triple_id:
            return None
        triple = self._pending.pop(triple_id, None)
        if triple is None:
            logger.warning("No pending triple for id=%s", triple_id)
            return None

        triple.o_feedback = feedback
        triple.o_quality_signal = max(-1.0, min(1.0, quality_signal))

        # Append to subsystem-specific JSONL
        out_path = self.root / f"triples_{triple.subsystem}.jsonl"
        try:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(triple.model_dump_json() + "\n")
        except OSError as exc:
            logger.error("Failed to write triple %s: %s", triple_id, exc)
            return None

        logger.debug("Completed triple %s → %s", triple_id, out_path)
        return triple

    def load_triples(
        self,
        subsystem: str | None = None,
        *,
        min_quality: float | None = None,
    ) -> list[RAMTriple]:
        """Load completed triples from disk.

        Parameters
        ----------
        subsystem : str, optional
            Filter to a specific subsystem. If ``None``, load all.
        min_quality : float, optional
            Only return triples with ``o_quality_signal >= min_quality``.
        """
        if not self.enabled:
            return []

        pattern = f"triples_{subsystem}.jsonl" if subsystem else "triples_*.jsonl"
        triples: list[RAMTriple] = []
        for path in sorted(self.root.glob(pattern)):
            try:
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            t = RAMTriple.model_validate_json(line)
                            if min_quality is not None and t.o_quality_signal < min_quality:
                                continue
                            triples.append(t)
                        except Exception:
                            logger.warning("Skipping malformed triple in %s", path)
            except OSError:
                continue
        return triples

    def export_for_training(
        self,
        output_path: Path,
        subsystem: str | None = None,
    ) -> int:
        """Export triples as a single JSONL file for the SDPO trainer.

        Returns the number of triples written.
        """
        triples = self.load_triples(subsystem)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for t in triples:
                if t.y_output and t.o_feedback:  # Only export complete triples
                    f.write(t.model_dump_json() + "\n")
                    count += 1
        logger.info("Exported %d triples to %s", count, output_path)
        return count

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def discard_pending(self) -> int:
        """Drop all pending (incomplete) triples. Returns count discarded."""
        n = len(self._pending)
        self._pending.clear()
        return n
