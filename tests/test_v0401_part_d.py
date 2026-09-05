"""v0.40.1 Part D — CLI UX consistency tests (highest-leverage subset).

Closes:
  - H4: Template list dynamic sync
  - M2: `soup init --force` flag
  - N6: `soup history` suggests `data registry` for dataset names
  - N2: `soup migrate` JSONL friendly error
  - G10: `soup eval custom -o` written regardless of `--attach-to-registry`
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

# --- H4: Template help is dynamically generated --------------------------


def test_init_template_help_lists_all_templates():
    from soup_cli.commands.init import _template_help_string
    from soup_cli.templates import list_templates

    help_text = _template_help_string()
    for template_name in list_templates():
        assert template_name in help_text, (
            f"template {template_name!r} missing from --template help"
        )


def test_init_template_help_includes_bco():
    """v0.40.0 added BCO; H4 must show it without a manual help-text edit."""
    from soup_cli.commands.init import _template_help_string

    assert "bco" in _template_help_string()


# --- M2: soup init --force flag ------------------------------------------


def test_init_force_flag_overwrites_without_prompt(tmp_path):
    from soup_cli.cli import app

    runner = CliRunner()
    target = tmp_path / "soup.yaml"
    target.write_text("base: existing", encoding="utf-8")

    # Without --force, prompts (we send 'n' to abort).
    result = runner.invoke(app, ["init", "--output", str(target)], input="n\n")
    assert result.exit_code == 0
    assert target.read_text(encoding="utf-8") == "base: existing", (
        "without --force the user-typed 'n' should abort and preserve file"
    )

    # With --force, overwrites silently using a registered template.
    result = runner.invoke(
        app, ["init", "--output", str(target), "--template", "chat", "--force"]
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert target.read_text(encoding="utf-8") != "base: existing"


# --- N2: soup migrate JSONL friendly error ------------------------------


def test_migrate_jsonl_input_yields_friendly_error(tmp_path: Path, monkeypatch):
    """Drives the *command*, not the helper: real JSONL named `.jsonl` is refused."""
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text('{"prompt": "hi"}\n{"prompt": "world"}\n', encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        app, ["migrate", "--from", "llamafactory", "data.jsonl", "--dry-run"]
    )
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "got JSONL" in result.output


def test_migrate_yaml_config_named_jsonl_still_migrates(tmp_path: Path, monkeypatch):
    """Control: the suffix alone must not condemn a file whose content is YAML.

    Fails if the `_looks_like_jsonl` call site is removed from the guard.
    """
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.jsonl"
    config.write_text(
        "model_name_or_path: meta-llama/Llama-3-8B\n"
        "stage: sft\n"
        "finetuning_type: lora\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        app, ["migrate", "--from", "llamafactory", "config.jsonl", "--dry-run"]
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "got JSONL" not in result.output


def test_migrate_bomd_jsonl_still_yields_friendly_error(tmp_path: Path, monkeypatch):
    """A UTF-8 BOM must not defeat the sniff (#675 review).

    Windows tooling writes UTF-8 with a BOM by default. Decoded as plain
    ``utf-8`` the BOM survives as U+FEFF, which ``str.strip()`` leaves alone
    because it is not whitespace, so ``startswith("{")`` is False and real
    JSONL sniffs as a config. Fails if the helper reads ``utf-8`` rather than
    ``utf-8-sig``.
    """
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    jsonl = tmp_path / "bom.jsonl"
    jsonl.write_bytes(
        b"\xef\xbb\xbf" + b'{"prompt": "hi"}\n{"prompt": "world"}\n'
    )
    # Guard the fixture itself: plain utf-8 must see the BOM this test is about.
    assert jsonl.read_text(encoding="utf-8").startswith("\ufeff")

    runner = CliRunner()
    result = runner.invoke(
        app, ["migrate", "--from", "llamafactory", "bom.jsonl", "--dry-run"]
    )
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "got JSONL" in result.output


def test_migrate_sniff_does_not_read_a_whole_unnewlined_file(tmp_path: Path):
    """The sniff is bounded: one very long line must not be read entire.

    ``for line in fh`` on a file with no newline pulls all of it into one
    string. Fails if the bounded ``readline`` is reverted.
    """
    import tracemalloc

    from soup_cli.commands.migrate import _looks_like_jsonl

    big = tmp_path / "oneline.jsonl"
    big.write_text('{"prompt": "' + "x" * (8 * 1024 * 1024) + '"}', encoding="utf-8")

    tracemalloc.start()
    try:
        assert _looks_like_jsonl(big) is True
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    # Bounded read touches 64 KiB; the unbounded one allocated the 8 MiB line
    # at least twice (read plus strip). 1 MiB leaves generous headroom.
    assert peak < 1024 * 1024, f"peak {peak} bytes: the read looks unbounded"


def test_migrate_helper_detects_jsonl(tmp_path: Path):
    from soup_cli.commands.migrate import _looks_like_jsonl

    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text('{"prompt": "hi"}\n{"prompt": "world"}\n', encoding="utf-8")
    assert _looks_like_jsonl(jsonl) is True


def test_migrate_yaml_does_not_look_like_jsonl(tmp_path: Path):
    from soup_cli.commands.migrate import _looks_like_jsonl

    yml = tmp_path / "config.yaml"
    yml.write_text("base: foo\ntask: sft\n", encoding="utf-8")
    assert _looks_like_jsonl(yml) is False


def test_migrate_skips_blank_lines_when_sniffing(tmp_path: Path):
    from soup_cli.commands.migrate import _looks_like_jsonl

    f = tmp_path / "blanks.jsonl"
    f.write_text('\n\n  \n{"key": 1}\n', encoding="utf-8")
    assert _looks_like_jsonl(f) is True


# --- #699: structural JSONL sniff for a suffix that gives no hint --------


def _jsonl_text(records: int = 2) -> str:
    """`records` one-line JSON objects, newline separated: real JSON Lines."""
    import json

    return "".join(
        json.dumps({"prompt": f"q{n}", "completion": f"a{n}"}) + "\n"
        for n in range(records)
    )


def _unsloth_notebook(indent: int | None = None) -> str:
    """A minimal Unsloth notebook `migrate_unsloth` can actually read.

    `indent=None` is the minified layout, which is the dangerous one: the whole
    notebook is a single line beginning with `{`, exactly the shape the suffix
    gate was written to protect.
    """
    import json

    return json.dumps(
        {
            "cells": [
                {
                    "cell_type": "code",
                    "source": [
                        "from unsloth import FastLanguageModel\n",
                        "model, tokenizer = FastLanguageModel.from_pretrained(\n",
                        "    model_name='unsloth/llama-3-8b', max_seq_length=2048\n",
                        ")\n",
                        "from trl import SFTTrainer\n",
                        "trainer = SFTTrainer(model=model, args=None)\n",
                    ],
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        },
        indent=indent,
    )


def test_migrate_jsonl_named_json_yields_friendly_error(tmp_path: Path, monkeypatch):
    """#699: real JSON Lines under a name that does not say `.jsonl`.

    On `main` this is `exit 1` and a raw `ParserError`, which is the message
    the N2 feature exists to replace. Fails if the structural sniff, or the
    branch that reaches it, is removed.
    """
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data.json").write_text(_jsonl_text(), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        app, ["migrate", "--from", "llamafactory", "data.json", "--dry-run"]
    )
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "got JSONL" in result.output


def test_migrate_single_json_object_named_json_still_migrates(
    tmp_path: Path, monkeypatch
):
    """#699: one JSON object is not JSON Lines, and must not be flagged as it.

    `yaml.safe_load` accepts a lone JSON object, so this file migrates today
    and has to keep migrating. Fails if the `two or more objects` threshold
    drops to one.
    """
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    (tmp_path / "cfg.json").write_text(
        '{"model_name_or_path": "meta-llama/Llama-3-8B", "stage": "sft",'
        ' "finetuning_type": "lora"}\n',
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        app, ["migrate", "--from", "llamafactory", "cfg.json", "--dry-run"]
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "got JSONL" not in result.output


def test_migrate_ipynb_notebook_still_migrates_via_unsloth(
    tmp_path: Path, monkeypatch
):
    """#699: the case the whole suffix gate exists to protect.

    A notebook is a single JSON object beginning with `{`, minified onto one
    line or pretty-printed, and `--from unsloth` takes both. Asserted on the
    file written rather than on the panel, so the check survives Rich
    wrapping. Fails if the sniff is ever widened to flag a lone JSON object.
    """
    import yaml

    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    for name, indent in (("minified.ipynb", None), ("pretty.ipynb", 1)):
        notebook = tmp_path / name
        notebook.write_text(_unsloth_notebook(indent), encoding="utf-8")
        # Guard the fixture: the minified layout must really be one line
        # starting with `{`, or this test stops being about anything.
        first_line = notebook.read_text(encoding="utf-8").splitlines()[0]
        assert first_line.startswith("{")
        if indent is None:
            assert len(notebook.read_text(encoding="utf-8").splitlines()) == 1

        written = tmp_path / f"{name}.soup.yaml"
        result = runner.invoke(
            app,
            [
                "migrate",
                "--from",
                "unsloth",
                name,
                "--output",
                written.name,
                "--yes",
            ],
        )
        assert result.exit_code == 0, (name, result.output, repr(result.exception))
        assert "got JSONL" not in result.output
        migrated = yaml.safe_load(written.read_text(encoding="utf-8"))
        assert isinstance(migrated, dict) and "base" in migrated, (name, migrated)


def test_migrate_single_record_jsonl_keeps_the_suffix_fast_path(
    tmp_path: Path, monkeypatch
):
    """#699: `.jsonl` still decides on the suffix plus the first character.

    A `.jsonl` file holding one record is data, not a config, and got the
    friendly error before this change. The structural sniff alone would let it
    through, because one object is not two. Fails if the `.jsonl` fast path is
    removed and every suffix goes to the structural sniff.
    """
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    (tmp_path / "one.jsonl").write_text(_jsonl_text(records=1), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        app, ["migrate", "--from", "llamafactory", "one.jsonl", "--dry-run"]
    )
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "got JSONL" in result.output


def test_migrate_yaml_sequence_of_flow_mappings_is_not_called_jsonl(
    tmp_path: Path, monkeypatch
):
    """#699: `- {"a": 1}` on consecutive lines is valid YAML, not JSON Lines.

    `yaml.safe_load` reads it as a list of two mappings, so it is not in the
    set of files that raise `ParserError` today. Fails if the sniff stops
    requiring each line to parse as a top-level JSON *object*.
    """
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    (tmp_path / "seq.json").write_text(
        '- {"a": 1}\n- {"b": 2}\n', encoding="utf-8"
    )

    runner = CliRunner()
    result = runner.invoke(
        app, ["migrate", "--from", "llamafactory", "seq.json", "--dry-run"]
    )
    assert "got JSONL" not in result.output


def test_migrate_lines_of_json_arrays_are_not_called_jsonl(
    tmp_path: Path, monkeypatch
):
    """#699 says "top-level **objects**", so a file of arrays is out of scope.

    `[1, 2]` on consecutive lines also fails `yaml.safe_load` today, with
    `found '['` rather than `found '{'`, so this is a deliberately narrower
    fix rather than a regression: the raw error it already gets is left alone.
    Fails if the sniff stops requiring each parsed line to be an object, which
    is the one thing only an array or a bare scalar can discriminate.
    """
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    (tmp_path / "arrays.json").write_text("[1, 2]\n[3, 4]\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        app, ["migrate", "--from", "llamafactory", "arrays.json", "--dry-run"]
    )
    assert "got JSONL" not in result.output
    assert result.exit_code == 1, (result.output, repr(result.exception))


def test_migrate_jsonl_record_over_the_read_bound_is_not_detected(
    tmp_path: Path, monkeypatch
):
    """#699: the disclosed limit of the bounded two-line sniff, pinned.

    `readline(65536)` stops at a newline OR at 65536 bytes, so a JSONL file
    whose first record is larger than 64 KiB yields a truncated fragment,
    `json.loads` fails, and the file is not classified as JSON Lines. The user
    gets the raw parser error, which is the pre-existing behaviour rather than
    a new wrong answer: the miss fails in the safe direction.

    The control below is the point of the test: byte-for-byte the same content
    named `.jsonl` still gets the friendly error, so this pins the read bound
    and not a claim that the file is somehow not JSONL.

    If the sniff is ever changed to stream brace depth instead of parsing two
    lines, this expectation is meant to change with it.
    """
    import json

    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    oversized = json.dumps({"prompt": "x" * (70 * 1024), "completion": "ok"})
    assert len(oversized) > 65536
    body = oversized + "\n" + json.dumps({"prompt": "second", "completion": "ok"}) + "\n"
    (tmp_path / "big.json").write_text(body, encoding="utf-8")
    (tmp_path / "big.jsonl").write_text(body, encoding="utf-8")

    runner = CliRunner()
    missed = runner.invoke(
        app, ["migrate", "--from", "llamafactory", "big.json", "--dry-run"]
    )
    assert "got JSONL" not in missed.output
    assert missed.exit_code == 1, (missed.output, repr(missed.exception))

    # Control: the same bytes under a `.jsonl` name are still caught, which is
    # what makes the miss above attributable to the 64 KiB bound.
    caught = runner.invoke(
        app, ["migrate", "--from", "llamafactory", "big.jsonl", "--dry-run"]
    )
    assert caught.exit_code == 2, (caught.output, repr(caught.exception))
    assert "got JSONL" in caught.output


def test_migrate_structural_sniff_does_not_read_a_whole_unnewlined_file(
    tmp_path: Path,
):
    """The structural sniff shares the bounded reader, so it is bounded too.

    Same property as `test_migrate_sniff_does_not_read_a_whole_unnewlined_file`
    for the new entry point. Fails if `_bounded_nonblank_lines` reverts to
    `for line in fh`, which pulls a file with no newline in whole.
    """
    import tracemalloc

    from soup_cli.commands.migrate import _looks_like_jsonl_by_structure

    big = tmp_path / "oneline.json"
    big.write_text('{"prompt": "' + "x" * (8 * 1024 * 1024) + '"}', encoding="utf-8")

    tracemalloc.start()
    try:
        # One truncated fragment, so not JSONL — see the test above for why
        # that is the safe answer rather than the right one.
        assert _looks_like_jsonl_by_structure(big) is False
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert peak < 1024 * 1024, f"peak {peak} bytes: the read looks unbounded"


def test_migrate_structural_sniff_reads_two_objects_not_one(tmp_path: Path):
    """Unit-level statement of the threshold, next to the CLI tests above."""
    from soup_cli.commands.migrate import _looks_like_jsonl_by_structure

    one = tmp_path / "one.json"
    one.write_text('{"prompt": "hi"}\n', encoding="utf-8")
    assert _looks_like_jsonl_by_structure(one) is False

    two = tmp_path / "two.json"
    two.write_text('{"prompt": "hi"}\n{"prompt": "there"}\n', encoding="utf-8")
    assert _looks_like_jsonl_by_structure(two) is True

    pretty = tmp_path / "pretty.json"
    pretty.write_text('{\n  "prompt": "hi"\n}\n', encoding="utf-8")
    assert _looks_like_jsonl_by_structure(pretty) is False


# --- N6: soup history suggests dataset registry --------------------------


def test_history_dataset_registry_helper_handles_missing():
    from soup_cli.commands.history import _name_exists_in_dataset_registry

    # Should never raise even if registry module / file is missing.
    assert isinstance(
        _name_exists_in_dataset_registry("definitely-not-a-dataset-xxx"), bool
    )


# --- G10: soup eval custom --output writes JSON without --attach-to-registry


def test_eval_custom_output_arg_described_as_independent():
    """The --output help string must mention it's honored without attach."""
    import inspect

    from soup_cli.commands.eval import custom

    src = inspect.getsource(custom)
    assert "Honored independently" in src or "G10" in src


def test_eval_custom_no_longer_shadows_output_with_response():
    """Source-level guard against the loop-variable shadow regression."""
    import inspect

    from soup_cli.commands.eval import custom

    src = inspect.getsource(custom)
    # Old buggy line: ``output = generate_fn(eval_task.prompt)``.
    # New line uses ``response`` to avoid shadowing the CLI ``output`` arg.
    assert "response = generate_fn" in src
    assert "output = generate_fn" not in src
