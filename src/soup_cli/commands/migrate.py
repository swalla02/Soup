"""soup migrate — import configs from LLaMA-Factory, Axolotl, and Unsloth."""

import json
from contextlib import closing
from pathlib import Path
from typing import Iterator

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()

SUPPORTED_SOURCES = ("llamafactory", "axolotl", "unsloth")

# Suffixes whose name already says "this is a config or a notebook", so the
# structural JSONL sniff is not run on them at all (#699).
CONFIG_SUFFIXES = (".yaml", ".yml", ".ipynb")

# How many parseable top-level JSON objects, on separate lines, it takes to
# call a file JSON Lines. Two, not one: a *single* JSON object is a notebook
# or some other config and must be left alone (#699).
JSONL_MIN_OBJECTS = 2

# Per-line read bound, in bytes. `readline(N)` stops at a newline *or* at N,
# so the sniff never holds more than JSONL_MIN_OBJECTS * this in memory
# (#675 measured 240.9 MB peak on a 120 MB one-liner before it was bounded).
SNIFF_LINE_BYTES = 65536


def migrate(
    source: str = typer.Option(
        ...,
        "--from",
        help="Source tool: llamafactory, axolotl, or unsloth",
    ),
    config_file: str = typer.Argument(
        ...,
        help="Path to the source config file (.yaml or .ipynb)",
    ),
    output: str = typer.Option(
        "soup.yaml",
        "--output",
        "-o",
        help="Output path for generated soup.yaml",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print generated config without writing to file",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompts",
    ),
):
    """Import a config from LLaMA-Factory, Axolotl, or Unsloth notebook."""
    from soup_cli.migrate.common import (
        config_to_yaml,
        validate_input_path,
        validate_output_path,
    )

    # Validate source
    if source not in SUPPORTED_SOURCES:
        console.print(
            f"[red]Unknown source: {source}[/]\n"
            f"Supported: {', '.join(SUPPORTED_SOURCES)}"
        )
        raise typer.Exit(1)

    # Validate input path
    input_path = Path(config_file)
    try:
        input_path = validate_input_path(input_path)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1)

    # v0.40.1 Part D / N2 — friendly error when the user passes a JSONL
    # data file instead of a YAML config.
    #
    # `.jsonl` keeps its original fast path: suffix plus a first-character
    # sniff. Every other suffix that does not already announce itself as a
    # config or a notebook gets the *structural* sniff instead, which is what
    # catches real JSONL under a name like `data.json` (#699). Structure is a
    # stronger signal than either the suffix or `startswith("{")`: a notebook
    # is one JSON object, JSONL is two or more on separate lines.
    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        is_jsonl_data = _looks_like_jsonl(input_path)
    elif suffix in CONFIG_SUFFIXES:
        # `.ipynb` notebooks legitimately start with `{`, and `.yaml` / `.yml`
        # say config outright. Neither is ever sniffed.
        is_jsonl_data = False
    else:
        is_jsonl_data = _looks_like_jsonl_by_structure(input_path)

    if is_jsonl_data:
        console.print(
            f"[red]Expected a {source} YAML config; got JSONL "
            f"({input_path.name}) — did you pass the wrong file?[/]"
        )
        console.print(
            "[dim]Tip: `soup migrate` migrates competitor *configs*, not "
            "training data. Pass the .yaml / .ipynb file instead.[/]"
        )
        raise typer.Exit(2)

    # Validate output path
    output_path = Path(output)
    if not dry_run:
        try:
            output_path = validate_output_path(output_path)
        except ValueError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(1)

    # Run migration
    try:
        if source == "llamafactory":
            from soup_cli.migrate.llamafactory import migrate_llamafactory
            result = migrate_llamafactory(input_path)
        elif source == "axolotl":
            from soup_cli.migrate.axolotl import migrate_axolotl
            result = migrate_axolotl(input_path)
        elif source == "unsloth":
            from soup_cli.migrate.unsloth import migrate_unsloth
            result = migrate_unsloth(input_path)
    except ValueError as exc:
        console.print(f"[red]Migration failed:[/] {exc}")
        raise typer.Exit(1)

    # Show warnings (escape Rich markup from untrusted config values)
    migration_warnings = result.get("_warnings", [])
    if migration_warnings:
        from rich.markup import escape
        warning_text = "\n".join(f"  [yellow]![/] {escape(w)}" for w in migration_warnings)
        console.print(Panel(
            warning_text,
            title="[yellow]Migration Warnings[/]",
            border_style="yellow",
        ))

    # Generate YAML
    yaml_str = config_to_yaml(result)

    # Show generated config
    console.print(Panel(
        Syntax(yaml_str, "yaml", theme="monokai"),
        title=f"[bold green]Generated soup.yaml[/] (from {source})",
    ))

    if dry_run:
        console.print("[dim]Dry run -- no file written.[/]")
        return

    # Check for existing file
    if output_path.exists() and not yes:
        confirm = typer.confirm(
            f"File '{output}' already exists. Overwrite?"
        )
        if not confirm:
            console.print("[yellow]Aborted.[/]")
            raise typer.Exit(0)

    # Write output
    output_path.write_text(yaml_str, encoding="utf-8")
    console.print(f"[green]\u2713[/] Config written to [bold]{output}[/]")
    console.print(f"[dim]Next: soup train --config {output}[/]")


def _bounded_nonblank_lines(path: Path) -> Iterator[str]:
    """Yield stripped non-blank lines, reading at most 64 KiB per line.

    The single reader behind both sniffs below, so there is one place where
    the encoding and the read bound are decided (#699).

    ``utf-8-sig``, not ``utf-8``: a UTF-8 BOM decodes to U+FEFF, which
    ``str.strip()`` does not remove because it is not whitespace, so a BOM'd
    JSONL file would sniff as *not* JSONL and silently lose the friendly
    error. Windows tooling writes that BOM by default, PowerShell's
    ``Out-File`` included (#675 review).

    ``readline(65536)`` rather than ``for line in fh``: a file with no newline
    is one single line, so iteration would pull all of it into memory, and a
    120 MB one-liner measured 240.9 MB peak (#675 review).

    Consequence worth knowing, because it is a real edge: ``readline`` stops
    at a newline *or* at the byte bound, whichever comes first, so a line
    longer than 64 KiB is yielded truncated. Harmless for the first-character
    sniff, and for the structural sniff it means a JSONL record over 64 KiB
    fails to parse and the file is not classified as JSONL, which leaves the
    pre-existing raw parser error in place rather than introducing a new
    wrong answer (#699).
    """
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
            while chunk := fh.readline(SNIFF_LINE_BYTES):
                stripped = chunk.strip()
                if stripped:
                    yield stripped
    except OSError:
        return


def _looks_like_jsonl(path: Path) -> bool:
    """v0.40.1 Part D / N2 — sniff first non-blank line for `{` (JSONL).

    The `.jsonl` fast path. Only ever called when the suffix already says
    ``.jsonl``, so the first character is enough and one record is enough:
    a file named ``.jsonl`` holding a single JSON object is still data rather
    than a config. Fails if the read is unbounded (`for line in fh`) or if
    the encoding drops back to plain ``utf-8``.
    """
    with closing(_bounded_nonblank_lines(path)) as lines:
        for first in lines:
            return first.startswith("{")
    return False


def _looks_like_jsonl_by_structure(path: Path) -> bool:
    """True when the first lines are JSONL_MIN_OBJECTS top-level JSON objects.

    The sniff for a suffix that gives no hint, e.g. ``data.json`` (#699).
    Structural rather than by name: parse the first non-blank line, and only
    if that succeeds read and parse the next one. Two parseable top-level
    objects on separate lines means JSON Lines.

    Anything less is left alone, which is the whole point of the threshold:
    a single JSON object is a notebook, or a JSON-shaped config, or something
    this command has no business guessing about, and ``yaml.safe_load``
    accepts all of those. It raises ``ParserError`` only on multi-document
    input, so "two or more objects on separate lines" and "would have raised
    ParserError" are the same set of files. That is why widening the sniff
    this way cannot break an input that migrates today.

    Fails if the threshold drops to one (a lone JSON object gets flagged) or
    if the object check goes away (a YAML sequence of flow mappings, whose
    first line is ``- {"a": 1}``, must not be flagged).
    """
    parsed = 0
    with closing(_bounded_nonblank_lines(path)) as lines:
        for line in lines:
            try:
                obj = json.loads(line)
            except ValueError:
                return False
            if not isinstance(obj, dict):
                return False
            parsed += 1
            if parsed >= JSONL_MIN_OBJECTS:
                return True
    return False
