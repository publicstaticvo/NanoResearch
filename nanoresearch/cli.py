"""CLI entry point for NanoResearch."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import sys
from pathlib import Path

# Fix Windows encoding: force UTF-8 for stdout/stderr to prevent
# UnicodeEncodeError when Rich prints non-ASCII characters (e.g. ö, é)
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace"
            )
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer, encoding="utf-8", errors="replace"
            )

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from nanoresearch import __version__
from nanoresearch.config import ExecutionProfile, ResearchConfig
from nanoresearch.pipeline.orchestrator import PipelineOrchestrator
from nanoresearch.pipeline.unified_orchestrator import UnifiedPipelineOrchestrator
from nanoresearch.pipeline.workspace import Workspace
from nanoresearch.profile import (
    ARCHETYPE_SEEDS,
    build_profile_seed,
    get_profile_json_path,
    get_profile_markdown_path,
    load_user_profile,
    render_profile_markdown,
    save_user_profile,
)
from nanoresearch.schemas.manifest import PaperMode, PipelineMode, PipelineStage
from nanoresearch.skills import UnifiedSkillMatcher

app = typer.Typer(
    name="nanoresearch",
    help="Minimal AI-driven research engine: idea → paper draft",
    add_completion=False,
)
profile_app = typer.Typer(help="Manage NanoResearch user persona/profile.")
skills_app = typer.Typer(help="Inspect NanoResearch static/evolved skill retrieval.")
app.add_typer(profile_app, name="profile")
app.add_typer(skills_app, name="skills")
console = Console()

_DEFAULT_ROOT = Path.home() / ".nanoresearch" / "workspace" / "research"


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"nanoresearch v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V", callback=_version_callback, is_eager=True,
        help="Show version and exit",
    ),
) -> None:
    """NanoResearch — AI-powered research paper generation pipeline."""
    # Auto-create ~/.nanoresearch directory structure if it doesn't exist
    _ensure_nanoresearch_home()


def _ensure_nanoresearch_home() -> None:
    """Create ~/.nanoresearch and its subdirectories if they don't exist."""
    nanoresearch_home = Path.home() / ".nanoresearch"
    subdirs = [
        "workspace/research",
        "chat_memory",
        "cache/models",
        "cache/data",
        "memory",
        "skills",
        "profile",
    ]

    nanoresearch_home.mkdir(parents=True, exist_ok=True)
    for subdir in subdirs:
        (nanoresearch_home / subdir).mkdir(parents=True, exist_ok=True)


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )


def _load_config_safe(config_path: Path | None) -> ResearchConfig:
    """Load config with user-friendly error messages."""
    try:
        cfg = ResearchConfig.load(config_path)
    except (RuntimeError, ValueError) as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(1)

    # Propagate optional third-party API keys from config.json → env vars
    _propagate_api_keys(config_path)
    return cfg


def _propagate_api_keys(config_path: Path | None) -> None:
    """Read optional API keys from config.json and set as env vars."""
    path = config_path or Path.home() / ".nanoresearch" / "config.json"
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    research = data.get("research", {})
    key_map = {
        "openalex_api_key": "OPENALEX_API_KEY",
        "s2_api_key": "S2_API_KEY",
    }
    for json_key, env_key in key_map.items():
        val = research.get(json_key, "")
        if val and not os.environ.get(env_key):
            os.environ[env_key] = str(val)


def _load_workspace_safe(path: Path) -> Workspace:
    """Load workspace with user-friendly error messages."""
    try:
        return Workspace.load(path)
    except FileNotFoundError:
        console.print(f"[red]Workspace not found:[/red] {path}")
        raise typer.Exit(1)
    except RuntimeError as exc:
        console.print(f"[red]Workspace error:[/red] {exc}")
        raise typer.Exit(1)


def _load_optional_config(config_path: Path | None) -> ResearchConfig:
    try:
        default_path = config_path or (Path.home() / ".nanoresearch" / "config.json")
        if default_path.is_file() or config_path is not None:
            return ResearchConfig.load(config_path)
    except Exception:
        pass
    return ResearchConfig()


def _resolve_skill_sources(config: ResearchConfig) -> tuple[list[Path], Path | None]:
    repo_root = Path(__file__).resolve().parents[1]
    bundled_skills_dir = repo_root / "skills"

    skill_dirs: list[Path] = []
    for value in config.static_skills_dirs:
        candidate = Path(value).expanduser()
        if candidate.is_dir() and candidate not in skill_dirs:
            skill_dirs.append(candidate)

    if config.static_skills_dir:
        legacy_dir = Path(config.static_skills_dir).expanduser()
        if legacy_dir.is_dir() and legacy_dir not in skill_dirs:
            skill_dirs.append(legacy_dir)

    if bundled_skills_dir.is_dir() and bundled_skills_dir not in skill_dirs:
        skill_dirs.append(bundled_skills_dir)

    manifest_path = Path(config.vendored_skills_manifest).expanduser() if config.vendored_skills_manifest else bundled_skills_dir / "vendor-ai-research" / "manifest.json"
    if not manifest_path.is_file():
        manifest_path = None
    return skill_dirs, manifest_path


@app.command()
def run(
    topic: str = typer.Option(..., "--topic", "-t", help="Research topic"),
    format: str = typer.Option(None, "--format", "-f", help="Paper format (auto-discovered from templates directory)"),
    config_path: Path = typer.Option(None, "--config", "-c", help="Path to config file"),
    profile: ExecutionProfile | None = typer.Option(
        None,
        "--profile",
        help="Unified execution profile",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config and exit without running"),
) -> None:
    """Run the unified research pipeline from topic to paper draft."""
    _setup_logging(verbose)

    # Validate topic
    if not topic or not topic.strip():
        console.print("[red]Error:[/red] --topic must be a non-empty string")
        raise typer.Exit(1)
    topic = topic.strip()

    # Parse paper_mode from topic prefix (e.g. "survey:short: LLM Reasoning")
    paper_mode = PaperMode.from_string(topic)
    if paper_mode.is_survey:
        # Strip the prefix from topic to get clean topic string
        for prefix in ["survey:short:", "survey:standard:", "survey:long:", "original:"]:
            if topic.lower().startswith(prefix):
                topic = topic[len(prefix):].strip()
                break

    config = _load_config_safe(config_path)
    if profile is not None:
        config.execution_profile = profile

    # Only override template_format if user explicitly passed --format
    if format is not None:
        from nanoresearch.templates import get_available_formats
        valid_formats = get_available_formats()
        if format not in valid_formats:
            console.print(f"[red]Error:[/red] --format must be one of {valid_formats}")
            raise typer.Exit(1)
        config.template_format = format

    if dry_run:
        console.print(Panel(
            f"[bold]Topic:[/bold] {topic}\n"
            f"[bold]Format:[/bold] {format}\n"
            f"[bold]Base URL:[/bold] {config.base_url}\n"
            f"[bold]Ideation model:[/bold] {config.ideation.model}\n"
            f"[bold]Writing model:[/bold] {config.writing.model}\n"
            f"[bold]Execution profile:[/bold] {config.execution_profile.value}\n"
            f"[bold]Writing mode:[/bold] {config.writing_mode.value}\n"
            f"[bold]Max retries:[/bold] {config.max_retries}\n"
            f"\n[green]Configuration is valid.[/green]",
            title="Dry Run",
            border_style="cyan",
        ))
        return

    workspace = Workspace.create(
        topic=topic,
        config_snapshot=config.snapshot(),
        pipeline_mode=PipelineMode.DEEP,
        paper_mode=paper_mode,
    )
    console.print(Panel(
        f"[bold]Topic:[/bold] {topic}\n"
        f"[bold]Pipeline:[/bold] Unified deep backbone\n"
        f"[bold]Profile:[/bold] {config.execution_profile.value}\n"
        f"[bold]Session:[/bold] {workspace.manifest.session_id}\n"
        f"[bold]Workspace:[/bold] {workspace.path}\n"
        f"[bold]Format:[/bold] {format}",
        title="NanoResearch",
        border_style="blue",
    ))

    orchestrator = UnifiedPipelineOrchestrator(
        workspace, config, progress_callback=_cli_progress,
    )
    try:
        result = asyncio.run(_run_deep_pipeline(orchestrator, topic))
        _print_result(result, workspace)
    except Exception as e:
        console.print(f"[red]Pipeline failed:[/red] {e}")
        raise typer.Exit(1)


def _cli_progress(stage: str, status: str, message: str) -> None:
    """Shared progress callback for CLI pipeline commands."""
    icons = {
        "started": "[cyan]>>>[/cyan]",
        "completed": "[green] OK[/green]",
        "skipped": "[dim] --[/dim]",
        "retrying": "[yellow] !![/yellow]",
        "failed": "[red]ERR[/red]",
    }
    console.print(f"  {icons.get(status, '   ')} {message}")


def _prompt_with_default(label: str, default: str) -> str:
    return typer.prompt(label, default=default, show_default=bool(default)).strip()


def _select_archetype() -> str:
    console.print(Panel(
        "\n".join(
            f"{idx}. {name}" for idx, name in enumerate(ARCHETYPE_SEEDS.keys(), start=1)
        ),
        title="Select Persona Archetype",
        border_style="magenta",
    ))
    names = list(ARCHETYPE_SEEDS.keys())
    choice = typer.prompt("Choose archetype number", default="4").strip()
    try:
        selected = names[max(0, min(len(names) - 1, int(choice) - 1))]
    except ValueError:
        selected = names[3]
    return selected


def _run_profile_interview(seed_name: str, existing: dict | None = None) -> dict:
    profile = build_profile_seed(seed_name)
    if existing:
        profile.update(existing)
        profile["archetype_seed"] = seed_name

    research = profile["research_profile"]
    resource = profile["resource_profile"]
    writing = profile["writing_profile"]
    publication = profile["publication_profile"]
    interaction = profile["interaction_profile"]

    console.print("[bold cyan]Nano persona interview[/bold cyan]")
    research["domain"] = _prompt_with_default("Research direction", research["domain"])
    research["method_preference"] = _prompt_with_default("Method preference", research["method_preference"])
    research["risk_preference"] = _prompt_with_default("Risk preference", research["risk_preference"])
    research["baseline_ablation_strictness"] = _prompt_with_default(
        "Baseline / ablation strictness", research["baseline_ablation_strictness"]
    )

    resource["gpu_budget"] = _prompt_with_default("GPU budget", resource["gpu_budget"])
    resource["wall_clock_budget"] = _prompt_with_default("Wall-clock budget", resource["wall_clock_budget"])
    resource["feasibility_bias"] = _prompt_with_default(
        "Feasibility / reproducibility preference", resource["feasibility_bias"]
    )

    writing["tone"] = _prompt_with_default("Writing tone", writing["tone"])
    writing["claim_strength"] = _prompt_with_default("Claim strength", writing["claim_strength"])
    writing["section_organization"] = _prompt_with_default(
        "Section organization", writing["section_organization"]
    )

    publication["venue_style"] = _prompt_with_default("Venue style", publication["venue_style"])
    publication["latex_template_preference"] = _prompt_with_default(
        "LaTeX/template preference", publication["latex_template_preference"]
    )
    publication["figure_style"] = _prompt_with_default("Figure style", publication["figure_style"])
    publication["caption_style"] = _prompt_with_default("Caption style", publication["caption_style"])

    interaction["priority_feedback"] = _prompt_with_default(
        "Most important feedback", interaction["priority_feedback"]
    )
    interaction["unacceptable_errors"] = _prompt_with_default(
        "Most unacceptable mistake", interaction["unacceptable_errors"]
    )

    return profile


def _save_profile_with_confirmation(profile: dict) -> None:
    console.print(Panel(render_profile_markdown(profile), title="Persona Summary", border_style="green"))
    if not typer.confirm("Save this profile?", default=True):
        console.print("[yellow]Profile creation cancelled.[/yellow]")
        raise typer.Exit(1)
    save_user_profile(profile)
    console.print(
        f"[green]Profile saved.[/green] JSON: {get_profile_json_path()}  MD: {get_profile_markdown_path()}"
    )


@app.command("init")
def init_profile() -> None:
    """Create or refresh the long-term NanoResearch user profile."""
    _ensure_nanoresearch_home()

    existing = load_user_profile()
    if existing is not None:
        console.print(
            Panel(render_profile_markdown(existing), title="Existing profile found", border_style="yellow")
        )
        if not typer.confirm("Refresh this profile now?", default=False):
            console.print("[cyan]Keeping current profile unchanged.[/cyan]")
            return

    archetype = _select_archetype()
    profile = _run_profile_interview(archetype, existing=existing)
    _save_profile_with_confirmation(profile)


@profile_app.command("show")
def profile_show() -> None:
    """Show the current user profile."""
    profile = load_user_profile()
    if profile is None:
        console.print("[yellow]No profile found. Run `nanoresearch init` first.[/yellow]")
        raise typer.Exit(1)
    console.print(Panel(render_profile_markdown(profile), title="NanoResearch Profile", border_style="blue"))


@profile_app.command("refresh")
def profile_refresh() -> None:
    """Refresh the current user profile via the interview flow."""
    init_profile()


@profile_app.command("export")
def profile_export(
    format: str = typer.Option("json", "--format", "-f", help="Export format: json or markdown"),
) -> None:
    """Print the current profile artifact path for downstream use."""
    profile = load_user_profile()
    if profile is None:
        console.print("[yellow]No profile found. Run `nanoresearch init` first.[/yellow]")
        raise typer.Exit(1)
    if format.lower() in {"md", "markdown"}:
        console.print(str(get_profile_markdown_path()))
        return
    console.print(str(get_profile_json_path()))


@skills_app.command("inspect")
def skills_inspect(
    stage: str = typer.Option(..., "--stage", help="Stage to inspect: planning/ideation/literature/experiment/coding/writing/review"),
    topic: str = typer.Option("", "--topic", help="Topic or task description"),
    text: str = typer.Option("", "--text", help="Free-form task text for retrieval"),
    template_format: str = typer.Option("", "--template-format", help="Template or venue format, e.g. neurips"),
    blueprint_file: Path | None = typer.Option(None, "--blueprint-file", help="JSON blueprint path for planning/experiment/coding inspection"),
    tags: str = typer.Option("", "--tags", help="Comma-separated retrieval tags"),
    config_path: Path | None = typer.Option(None, "--config", "-c", help="Optional config file path"),
) -> None:
    """Inspect which static/evolved/script skills would be injected for a stage."""
    config = _load_optional_config(config_path)
    skill_dirs, manifest_path = _resolve_skill_sources(config)
    matcher = UnifiedSkillMatcher(
        skills_dirs=skill_dirs,
        manifest_path=manifest_path,
        retrieval_top_k=config.skill_retrieval_top_k,
        autorun_policy=config.script_skill_autorun_policy,
    )

    blueprint = None
    if blueprint_file is not None:
        try:
            blueprint = json.loads(Path(blueprint_file).read_text(encoding="utf-8"))
        except Exception as exc:
            console.print(f"[red]Failed to read blueprint JSON:[/red] {exc}")
            raise typer.Exit(1)

    profile = load_user_profile()
    context = matcher.build_context(
        stage.strip().lower(),
        topic=topic,
        blueprint=blueprint,
        text=text,
        tags=[item.strip() for item in tags.split(",") if item.strip()],
        template_format=template_format,
        profile=profile,
    )

    payload = {
        "stage": stage.strip().lower(),
        "topic": topic,
        "template_format": template_format,
        "profile_loaded": profile is not None,
        "skill_dirs": [str(path) for path in skill_dirs],
        "manifest_path": str(manifest_path) if manifest_path else "",
        "candidate_static_skills": context.candidate_static_skills,
        "matched_static_skills": context.matched_static_skills,
        "matched_evolved_skills": context.matched_evolved_skills,
        "matched_script_skills": context.matched_script_skills,
        "matched_skills": context.matched_skills,
        "static_context_preview": context.static_context[:1200],
        "evolved_context_preview": context.evolved_context[:1200],
        "script_context_preview": context.script_context[:1200],
    }
    console.print_json(data=payload)


@skills_app.command("trace")
def skills_trace(
    workspace: Path = typer.Option(..., "--workspace", "-w", help="Path to a NanoResearch workspace"),
) -> None:
    """Read actual adaptive-context skill traces from a completed workspace."""
    ws = _load_workspace_safe(workspace)
    trace_files = sorted((ws.path / "logs").glob("adaptive_context_*.json"))
    stages: list[dict] = []
    for trace_file in trace_files:
        try:
            payload = json.loads(trace_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        stages.append(
            {
                "file": trace_file.name,
                "task_type": payload.get("task_type", ""),
                "topic": payload.get("topic", ""),
                "candidate_static_skills": payload.get("candidate_static_skills", []),
                "matched_static_skills": payload.get("matched_static_skills", []),
                "matched_evolved_skills": payload.get("matched_evolved_skills", []),
                "matched_script_skills": payload.get("matched_script_skills", []),
                "matched_skills": payload.get("matched_skills", []),
            }
        )

    summary = {
        "workspace": str(ws.path),
        "session_id": ws.manifest.session_id,
        "topic": ws.manifest.topic,
        "stages": stages,
    }
    console.print_json(data=summary)


@app.command()
def resume(
    workspace: Path = typer.Option(..., "--workspace", "-w", help="Path to workspace directory"),
    config_path: Path = typer.Option(None, "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Resume a pipeline from its last checkpoint."""
    _setup_logging(verbose)

    ws = _load_workspace_safe(workspace)
    manifest = ws.manifest
    config = _load_config_safe(config_path)

    console.print(Panel(
        f"[bold]Session:[/bold] {manifest.session_id}\n"
        f"[bold]Topic:[/bold] {manifest.topic}\n"
        f"[bold]Current Stage:[/bold] {manifest.current_stage.value}",
        title="Resuming NanoResearch",
        border_style="yellow",
    ))

    if manifest.current_stage in (PipelineStage.DONE, PipelineStage.FAILED):
        # Reset FAILED to last incomplete stage
        if manifest.current_stage == PipelineStage.FAILED:
            found_failed = False
            for stage_name, rec in manifest.stages.items():
                if rec.status == "failed":
                    rec.status = "pending"
                    manifest.current_stage = rec.stage
                    ws.update_manifest(
                        current_stage=manifest.current_stage,
                        stages=manifest.stages,
                    )
                    console.print(
                        f"  Resetting failed stage [yellow]{stage_name}[/yellow] to pending"
                    )
                    found_failed = True
                    break
            if not found_failed:
                console.print(
                    "[yellow]Pipeline is FAILED but no failed stage found. "
                    "Check manifest manually.[/yellow]"
                )
                raise typer.Exit(1)
        else:
            console.print("[green]Pipeline already completed.[/green]")
            return

    is_deep = manifest.pipeline_mode == PipelineMode.DEEP

    if is_deep:
        console.print("  [magenta]Detected unified/deep workspace — using UnifiedPipelineOrchestrator[/magenta]")
        orchestrator = UnifiedPipelineOrchestrator(
            ws, config, progress_callback=_cli_progress,
        )
        try:
            result = asyncio.run(_run_deep_pipeline(orchestrator, manifest.topic))
            _print_result(result, ws)
        except Exception as e:
            console.print(f"[red]Deep pipeline failed:[/red] {e}")
            raise typer.Exit(1)
    else:
        orchestrator = PipelineOrchestrator(
            ws, config, progress_callback=_cli_progress,
        )
        try:
            result = asyncio.run(_run_pipeline(orchestrator, manifest.topic))
            _print_result(result, ws)
        except Exception as e:
            console.print(f"[red]Pipeline failed:[/red] {e}")
            raise typer.Exit(1)


@app.command()
def status(
    workspace: Path = typer.Option(..., "--workspace", "-w", help="Path to workspace directory"),
) -> None:
    """Show the status of a research session."""
    ws = _load_workspace_safe(workspace)
    manifest = ws.manifest

    table = Table(title=f"Session: {manifest.session_id}")
    table.add_column("Stage", style="bold")
    table.add_column("Status")
    table.add_column("Started")
    table.add_column("Completed")
    table.add_column("Retries")

    status_colors = {
        "pending": "dim",
        "running": "yellow",
        "completed": "green",
        "failed": "red",
    }

    for stage_name, rec in manifest.stages.items():
        color = status_colors.get(rec.status, "white")
        started = rec.started_at.strftime("%H:%M:%S") if rec.started_at else "-"
        completed = rec.completed_at.strftime("%H:%M:%S") if rec.completed_at else "-"
        table.add_row(
            stage_name,
            f"[{color}]{rec.status}[/{color}]",
            started,
            completed,
            str(rec.retries),
        )

    console.print(table)
    console.print(f"\n[bold]Topic:[/bold] {manifest.topic}")
    console.print(f"[bold]Mode:[/bold] {manifest.pipeline_mode.value}")
    execution_profile = manifest.config_snapshot.get("execution_profile", "?")
    console.print(f"[bold]Profile:[/bold] {execution_profile}")
    console.print(f"[bold]Current Stage:[/bold] {manifest.current_stage.value}")
    console.print(f"[bold]Artifacts:[/bold] {len(manifest.artifacts)}")
    for art in manifest.artifacts:
        console.print(f"  - {art.name}: {art.path}")


@app.command("list")
def list_sessions(
    root: Path = typer.Option(_DEFAULT_ROOT, "--root", "-r"),
) -> None:
    """List all research sessions."""
    if not root.is_dir():
        console.print("[dim]No sessions found.[/dim]")
        return

    table = Table(title="Research Sessions")
    table.add_column("Session ID", style="bold")
    table.add_column("Topic")
    table.add_column("Stage")
    table.add_column("Created")

    for session_dir in sorted(root.iterdir()):
        manifest_path = session_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            created = str(data.get("created_at", "?"))
            table.add_row(
                data.get("session_id", "?"),
                str(data.get("topic", "?"))[:50],
                data.get("current_stage", "?"),
                created[:19] if len(created) >= 19 else created,
            )
        except (json.JSONDecodeError, OSError) as exc:
            console.print(
                f"[dim]Skipping {session_dir.name}: corrupted manifest ({exc})[/dim]"
            )
            continue

    console.print(table)


async def _run_pipeline(orchestrator: PipelineOrchestrator, topic: str) -> dict:
    try:
        return await orchestrator.run(topic)
    finally:
        await orchestrator.close()


async def _run_deep_pipeline(orchestrator, topic: str) -> dict:
    try:
        return await orchestrator.run(topic)
    finally:
        await orchestrator.close()


def _print_result(result: dict, workspace: Workspace) -> None:
    console.print("\n[bold green]Pipeline completed![/bold green]\n")

    # Auto-export to a clean output folder
    try:
        export_path = workspace.export()
        console.print(Panel(
            f"[bold]Output folder:[/bold] {export_path}\n\n"
            f"  paper.pdf        — Compiled paper\n"
            f"  paper.tex        — LaTeX source\n"
            f"  references.bib   — Bibliography\n"
            f"  figures/         — All figures\n"
            f"  code/            — Experiment code skeleton\n"
            f"  data/            — Structured research data\n"
            f"  manifest.json    — Pipeline execution record",
            title="[green]Exported[/green]",
            border_style="green",
        ))
    except Exception as e:
        console.print(f"[yellow]Export failed:[/yellow] {e}")
        console.print(f"[bold]Raw workspace:[/bold] {workspace.path}")


# Import command modules to register their @app.command() decorators
import nanoresearch.cli_commands  # noqa: F401, E402
import nanoresearch.cli_code_edit  # noqa: F401, E402
import nanoresearch.cli_paper_edit  # noqa: F401, E402


if __name__ == "__main__":
    app()
