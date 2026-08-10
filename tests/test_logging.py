"""Logger naming and UTF-8 output.

Both of these are cosmetic until they are not. `ensure_utf8_output` exists because a print of an
em-dash raised `UnicodeEncodeError` *after* a five-hour run had written its results, leaving a
non-zero exit status that silently stopped everything chained after it with `&&`.

The naming test is smaller: `get_logger(__name__)` prefixed the root package onto a name that
already began with it, so every line of every cluster log read

    prooflens_prover.prooflens_prover.prover.vllm_policy

which is noise in the one place where a human is reading carefully, and it makes the module path
harder to find rather than easier.

Hermetic.
"""

from __future__ import annotations

import logging

from prooflens_prover.utils.logging import ensure_utf8_output, get_logger


class TestLoggerNaming:
    def test_dunder_name_is_not_prefixed_twice(self):
        """The call every module in this package makes."""
        assert get_logger("prooflens_prover.prover.vllm_policy").name == (
            "prooflens_prover.prover.vllm_policy"
        )

    def test_a_short_label_is_still_placed_under_the_root(self):
        """`get_logger("prover.search")` — the other convention in use here.

        Both must end up under the same root, or one of them escapes the configured handler and
        logs nothing at all.
        """
        assert get_logger("prover.search").name == "prooflens_prover.prover.search"

    def test_the_root_itself_is_not_doubled(self):
        assert get_logger("prooflens_prover").name == "prooflens_prover"

    def test_every_logger_sits_under_the_configured_root(self):
        """The property that actually matters: a logger outside the root emits nothing."""
        for name in ("prooflens_prover.prover.search", "prover.search", "prooflens_prover"):
            assert get_logger(name).name.startswith("prooflens_prover")

    def test_a_name_merely_starting_with_the_root_word_is_untouched(self):
        """`prooflens_proverish` is not the root package, so it must not be stripped."""
        assert get_logger("prooflens_proverish.thing").name == (
            "prooflens_prover.prooflens_proverish.thing"
        )

    def test_loggers_actually_emit_through_the_package_handler(self):
        """Captured at the package logger, not pytest's `caplog`.

        `get_logger` sets `propagate = False` on `prooflens_prover` deliberately — the package owns
        its stderr handler and must not also emit through whatever root configuration a host
        process happens to have. That is exactly why `caplog`, whose handler sits on the true root,
        sees nothing here. So this attaches a handler where the records actually go, which is also
        the only version of this test that would notice a logger escaping the root.
        """
        records: list[logging.LogRecord] = []

        class Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        package = logging.getLogger("prooflens_prover")
        handler = Capture()
        package.addHandler(handler)
        try:
            get_logger("prooflens_prover.prover.vllm_policy").warning("a message")
            get_logger("prover.search").warning("another")
        finally:
            package.removeHandler(handler)

        assert [r.getMessage() for r in records] == ["a message", "another"]


class TestEnsureUtf8Output:
    def test_it_is_idempotent_and_never_raises(self):
        # Called by get_logger and by every script's main(); a second call must be a no-op.
        ensure_utf8_output()
        ensure_utf8_output()

    def test_it_tolerates_a_stream_without_reconfigure(self, monkeypatch):
        """pytest's capture object and plain file wrappers have no `reconfigure`."""
        import sys

        monkeypatch.setattr(sys, "stdout", object())
        ensure_utf8_output()      # must not raise
