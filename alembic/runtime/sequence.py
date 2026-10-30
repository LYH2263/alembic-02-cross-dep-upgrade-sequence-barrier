# mypy: allow-untyped-defs, allow-incomplete-defs, allow-untyped-calls

"""Host-facing cross-dependency upgrade/downgrade sequence barrier.

A revision graph whose branches are additionally linked by
``depends_on`` edges has three *distinct* notions that single-shot
:func:`alembic.command.upgrade` / :func:`alembic.command.downgrade` calls
never need to tell apart:

1. script-directory heads counting version (``down_revision``) edges only
   (:meth:`.ScriptDirectory.get_versioned_heads`),
2. script-graph tips once cross-branch ``depends_on`` edges are counted
   (:meth:`.ScriptDirectory.get_dependency_heads`),
3. the rows actually present in the ``alembic_version`` table
   (:meth:`.ScriptDirectory.classify_version_rows`).

When a host chains several upgrades/downgrades -- including relative
(``+1``/``-1``) and branch-tip (``somebranch@head``) targets -- into one
job, each later step has to resolve its target against the rows actually
present *after* the previous step mutated the version table; freezing
the head snapshot at job start leaves stale row sets that cannot be
recovered with another ``upgrade heads``.

This module provides that sequencing on top of four independently
carried concerns:

* head semantics live on :class:`.RevisionMap` / :class:`.ScriptDirectory`
  (carrier A),
* per-step version table maintenance lives in
  :class:`.StepwiseVersionMaintainer` together with
  :func:`.plan_version_row_changes` (carrier B),
* per-step target re-resolution lives in the :class:`.SequenceMove`
  hierarchy of this module (carrier C),
* open/close orchestration, nesting isolation and failure rollback live
  in :class:`.MigrationSequence` (carrier D).

The barrier is opt-in; when it is not used, single-shot
upgrade/downgrade/stamp behavior is unchanged.

.. versionadded:: 1.21

"""

from __future__ import annotations

from collections.abc import Collection
from collections.abc import Iterable
import contextlib
import logging
from typing import Any
from typing import Callable
from typing import TYPE_CHECKING

from .. import util
from ..operations import Operations
from ..script.base import ScriptDirectory
from ..util import sqla_compat
from .migration import MigrationContext
from .migration import RevisionStep
from .migration import StampStep
from .migration import StepwiseVersionMaintainer
from .migration import VersionRowChange

if TYPE_CHECKING:
    from sqlalchemy.engine.base import Connection
    from sqlalchemy.engine.base import Transaction

    from ..script.revision import VersionRowState

log = logging.getLogger("alembic.runtime.sequence")


class SequenceBarrierError(util.CommandError):
    """Raised when a host sequence barrier cannot be opened or finalized
    safely.

    When raised at a nested barrier's clean close because the inner
    sequence destructively finalized rows the outer sequence still
    depends on, the inner barrier's side effects have already been rolled
    back.

    """


# ----------------------------------------------------------------------------
# Carrier C: target specs re-resolved per step against the LIVE row set
# ----------------------------------------------------------------------------


class SequenceMove:
    """One upgrade/downgrade instruction inside a host sequence.

    A move keeps only the *raw* target specification (an absolute
    revision id, a symbolic head, ``+n``/``-n``, or ``branch@head``).
    Nothing is expanded at sequence entry: :meth:`.resolve` is called
    right before execution with the rows that are actually present at
    that moment, so the same move can produce different revision steps at
    different positions of a sequence.

    """

    direction: str = ""

    def __init__(self, target: str):
        self.target_spec = target

    def resolve(
        self,
        script: ScriptDirectory,
        current_heads: Collection[str],
    ) -> list[RevisionStep]:
        """Resolve this move against ``current_heads`` -- the rows live
        in the database immediately before the move executes."""

        raise NotImplementedError()

    def __repr__(self) -> str:
        return "%s(%r)" % (self.__class__.__name__, self.target_spec)


class UpgradeMove(SequenceMove):
    """An upgrade instruction within a sequence (see :class:`.SequenceMove`)."""

    direction = "upgrade"

    def resolve(
        self,
        script: ScriptDirectory,
        current_heads: Collection[str],
    ) -> list[RevisionStep]:
        return script._upgrade_revs(
            self.target_spec, tuple(current_heads) or None
        )


class DowngradeMove(SequenceMove):
    """A downgrade instruction within a sequence (see
    :class:`.SequenceMove`)."""

    direction = "downgrade"

    def resolve(
        self,
        script: ScriptDirectory,
        current_heads: Collection[str],
    ) -> list[RevisionStep]:
        return script._downgrade_revs(
            self.target_spec, tuple(current_heads) or None
        )


# ----------------------------------------------------------------------------
# observable per-step record
# ----------------------------------------------------------------------------


class SequenceStepRecord:
    """What happened to the version table at one sequence step boundary.

    The record is created only after the migration function ran and the
    version table rows were mutated; ``heads_before``/``heads_after``
    therefore bracket a real step, and ``row_changes`` exposes the exact
    insert/delete/update row mutations (a cross-branch
    ``depends_on`` merge/unmerge yields more than one).

    """

    __slots__ = (
        "revision_id",
        "is_upgrade",
        "is_stamp",
        "heads_before",
        "heads_after",
        "row_changes",
    )

    def __init__(
        self,
        step: RevisionStep | StampStep,
        heads_before: frozenset[str],
        heads_after: frozenset[str],
        row_changes: tuple[VersionRowChange, ...],
    ) -> None:
        info = step.info
        self.revision_id: str | None = info.up_revision_id
        self.is_upgrade: bool = info.is_upgrade
        self.is_stamp: bool = info.is_stamp
        self.heads_before: frozenset[str] = heads_before
        self.heads_after: frozenset[str] = heads_after
        self.row_changes: tuple[VersionRowChange, ...] = row_changes

    def __repr__(self) -> str:
        return (
            "SequenceStepRecord(%r, is_upgrade=%r, %r -> %r, changes=%r)"
            % (
                self.revision_id,
                self.is_upgrade,
                set(self.heads_before),
                set(self.heads_after),
                self.row_changes,
            )
        )


# ----------------------------------------------------------------------------
# internal frame bookkeeping -- one open barrier on one context
# ----------------------------------------------------------------------------

_FRAMES_ATTR = "_alembic_sequence_frames"


class _BarrierFrame:
    __slots__ = (
        "sequence",
        "protected_rows",
        "anchor_transaction",
        "savepoint",
        "nested",
        "failed",
        "finalization_error",
    )

    def __init__(
        self,
        sequence: MigrationSequence,
        protected_rows: frozenset[str],
        anchor_transaction: Transaction | None,
        savepoint: Any,
        nested: bool,
    ) -> None:
        self.sequence = sequence
        # rows live at frame entry; a nested frame must not destructively
        # finalize any of these on close.
        self.protected_rows = protected_rows
        # transaction opened by this frame on a connection that had none
        self.anchor_transaction = anchor_transaction
        self.savepoint = savepoint
        self.nested = nested
        self.failed = False
        self.finalization_error: SequenceBarrierError | None = None


# ----------------------------------------------------------------------------
# Carrier D: the sequence barrier
# ----------------------------------------------------------------------------


class MigrationSequence:
    """A host-declared, recoverable upgrade/downgrade sequence barrier.

    Binds one sequence to a shared :class:`.ScriptDirectory` plus one
    online target database bind (an existing :class:`.MigrationContext`,
    or a connection / url from which a context is created).

    Use it as a context manager; within the open barrier, each
    :meth:`.upgrade` / :meth:`.downgrade` call is a *move* whose target
    is resolved against the version table rows live at that instant, the
    migration functions really execute, and only afterwards are the
    planned row mutations applied (real inserts/deletes/updates -- a
    cross-branch ``depends_on`` merge/unmerge is multi-row, never a
    single scalar UPDATE).  Every step boundary, and the barrier close
    itself, must leave an effective-head row state from which upgrading
    towards the script heads can continue.

    Semantics:

    * Each step runs in its own SAVEPOINT; the outermost barrier opens a
      transaction when the bind connection has none (a host-managed
      external transaction is respected and left alone).
    * A nested barrier may not destructively close while rows the outer
      barrier still depends on are missing: its SAVEPOINT is rolled back
      and :class:`.SequenceBarrierError` is raised.
    * Any exception inside a frame pops that frame's own side effects;
      after the outermost frame rolls back, the database is again in the
      detectable safe state observed at barrier entry.
    * All bookkeeping (frames, history, protected rows) is per
      :class:`.MigrationSequence` / :class:`.MigrationContext` instance;
      two parallel barriers never share state.
    * Offline (``as_sql=True``) contexts are rejected.

    """

    def __init__(
        self,
        script: ScriptDirectory,
        *,
        context: MigrationContext | None = None,
        connection: Connection | None = None,
        url: str | None = None,
        dialect_name: str | None = None,
        dialect_opts: dict[str, str] | None = None,
        on_version_apply: Callable[..., Any] | Iterable[Callable[..., Any]]
        | None = None,
        context_opts: dict[str, Any] | None = None,
    ) -> None:
        self.script = script

        if context is None:
            if connection is None and not url and not dialect_name:
                raise SequenceBarrierError(
                    "MigrationSequence requires an online bind: pass "
                    "'connection', 'url' or 'dialect_name', or an "
                    "existing online 'context'"
                )
            opts: dict[str, Any] = {"script": script}
            if context_opts:
                opts.update(context_opts)
            context = MigrationContext.configure(
                connection=connection,
                url=url,
                dialect_name=dialect_name,
                dialect_opts=dialect_opts,
                opts=opts,
            )
        self.context: MigrationContext = context

        self._extra_callbacks: tuple[Callable[..., Any], ...] = tuple(
            util.to_tuple(on_version_apply, default=())
        )

        # carrier B: version table rows are re-read before every step
        self._maintainer = StepwiseVersionMaintainer(context)

        self._frame: _BarrierFrame | None = None
        self._operations: Operations | None = None
        self._saved_op_proxy: Any = None
        self._poisoned = False
        self.history: list[SequenceStepRecord] = []

    # -- introspection ------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """True while this barrier's frame is open."""

        return self._frame is not None

    @property
    def applied(self) -> tuple[SequenceStepRecord, ...]:
        """Per-step records accumulated by this barrier, in order."""

        return tuple(self.history)

    @property
    def versioned_heads(self) -> list[str]:
        """Script heads counting version edges only (display answer)."""

        return self.script.get_versioned_heads()

    @property
    def dependency_heads(self) -> list[str]:
        """Script tips once ``depends_on`` edges are counted (distinct
        from both :attr:`.versioned_heads` and the live rows)."""

        return self.script.get_dependency_heads()

    def current_rows(self) -> tuple[str, ...]:
        """Read the rows live in ``alembic_version`` right now."""

        return self.context.get_current_heads()

    def row_state(self) -> VersionRowState:
        """Classify the rows live in ``alembic_version`` right now."""

        return self.script.classify_version_rows(self.current_rows())

    # -- context manager protocol ------------------------------------------

    def __enter__(self) -> MigrationSequence:
        if self._frame is not None:
            raise SequenceBarrierError(
                "this sequence barrier is already open"
            )
        if self._poisoned:
            raise SequenceBarrierError(
                "this sequence barrier has already failed; open a new "
                "MigrationSequence on the bind to continue"
            )

        ctx = self.context
        if ctx.as_sql:
            raise SequenceBarrierError(
                "the sequence barrier is not available in offline "
                "(--sql / as_sql) mode"
            )
        if ctx.connection is None:
            raise SequenceBarrierError(
                "the sequence barrier requires an online connection"
            )

        frames = self._frames()
        nested = bool(frames)

        # only the outermost frame starts migrations on the dialect;
        # a nested frame borrows the outer frame's migration run.
        if not nested:
            ctx.impl.start_migrations()

        conn = ctx.connection

        # anchor transaction only if the host did not already start one;
        # the begin_transaction semantics of MigrationContext are
        # deliberately not touched.
        anchor_transaction = None
        if not sqla_compat._get_connection_in_transaction(conn):
            anchor_transaction = conn.begin()

        savepoint = conn.begin_nested()

        frame = _BarrierFrame(
            sequence=self,
            protected_rows=frozenset(ctx.get_current_heads()),
            anchor_transaction=anchor_transaction,
            savepoint=savepoint,
            nested=nested,
        )
        frames.append(frame)
        self._frame = frame

        # revision scripts use the ``from alembic import op`` proxy.
        # The proxy slot on the ``alembic.op`` module is a single global
        # rather than a stack, so every frame saves the proxy it finds
        # (installed by an outer frame on this context, or by a parallel
        # barrier on another context) and restores it on close -- two
        # parallel barriers on different contexts can therefore interleave
        # without stealing each other's proxy.
        self._install_op_proxy()

        log.debug(
            "opened %ssequence barrier (protected rows: %s)",
            "nested " if nested else "",
            sorted(frame.protected_rows),
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        frame = self._frame
        assert frame is not None, "barrier frame missing on close"
        frames = self._frames()
        if frames[-1] is not frame:
            # an inner barrier on the same context was never closed
            self._fail(frame)
            self._rollback(frame)
            frames.remove(frame)
            self._frame = None
            self._restore_op_proxy()
            self._poisoned = True
            raise SequenceBarrierError(
                "an inner sequence barrier is still open while the outer "
                "barrier is being closed"
            )

        failure = exc_type is not None or frame.failed
        finalization_error: SequenceBarrierError | None = None

        if not failure and frame.nested:
            # nesting contract: an inner short sequence must not
            # destructively finalize rows the outer sequence still
            # depends on.
            live = frozenset(self.context.get_current_heads())
            missing = frame.protected_rows.difference(live)
            if missing:
                finalization_error = SequenceBarrierError(
                    "nested sequence destructively finalized outer "
                    "sequence rows %s; the nested sequence was rolled back"
                    % (sorted(missing),)
                )
                failure = True
        elif not failure:
            # outermost close: the remaining rows must be effective
            # tips -- upgrade heads must still be possible.
            state = self.script.classify_version_rows(
                self.context.get_current_heads()
            )
            if not state.is_effective_head_state:
                finalization_error = SequenceBarrierError(
                    "sequence ended with a non-effective version row "
                    "state: %r" % (state,)
                )
                failure = True

        frame.finalization_error = finalization_error

        if failure:
            self._rollback(frame)
        else:
            frame.savepoint.commit()
            if frame.anchor_transaction is not None:
                frame.anchor_transaction.commit()

        frames.pop()
        self._frame = None
        self._restore_op_proxy()

        log.debug(
            "closed %ssequence barrier (%s)",
            "nested " if frame.nested else "",
            "rolled back" if failure else "committed",
        )

        if failure:
            self._poisoned = True
            if exc is None and finalization_error is not None:
                raise finalization_error
        return False

    # -- moves ---------------------------------------------------------------

    def upgrade(self, target: str, **kw: Any) -> tuple[SequenceStepRecord, ...]:
        """Run one upgrade move; ``target`` is resolved against the rows
        live right now (relative / branch-tip targets included)."""

        return self._execute_move(UpgradeMove(target), kw)

    def downgrade(
        self, target: str, **kw: Any
    ) -> tuple[SequenceStepRecord, ...]:
        """Run one downgrade move; ``target`` is resolved against the
        rows live right now (relative / branch-tip targets included)."""

        return self._execute_move(DowngradeMove(target), kw)

    def run(
        self, *moves: SequenceMove, **kw: Any
    ) -> tuple[SequenceStepRecord, ...]:
        """Open the barrier, execute ``moves`` in order, and close it.

        Cannot be used while the barrier is already active; nested use
        goes through explicit ``with`` blocks.

        """
        if self._frame is not None:
            raise SequenceBarrierError(
                "run() cannot nest; use the barrier as a context manager "
                "for nested sequences"
            )
        records: list[SequenceStepRecord] = []
        with self:
            for move in moves:
                records.extend(self._execute_move(move, kw))
        return tuple(records)

    def _execute_move(
        self, move: SequenceMove, run_kwargs: dict[str, Any]
    ) -> tuple[SequenceStepRecord, ...]:
        frame = self._frame
        if frame is None:
            raise SequenceBarrierError(
                "execute moves inside the sequence barrier context "
                "manager (or use MigrationSequence.run())"
            )
        if frame.failed:
            raise SequenceBarrierError(
                "the sequence barrier has already failed; close it to "
                "roll back"
            )
        if self._frames()[-1] is not frame:
            raise SequenceBarrierError(
                "an inner sequence barrier is open on this context; "
                "close it before scheduling another move"
            )

        ctx = self.context
        conn = ctx.connection
        assert conn is not None

        # CRUCIAL: read the live rows FIRST, then resolve.  The same
        # relative/branch target must answer differently once an earlier
        # step of the sequence has mutated the version table.
        live_heads = self._maintainer.refresh()
        log.debug(
            "sequence move %r resolved against live rows %s",
            move,
            sorted(live_heads),
        )

        steps = move.resolve(self.script, live_heads)

        if steps and not ctx._has_version_table():
            ctx._ensure_version_table()

        records: list[SequenceStepRecord] = []

        for step in steps:
            heads_before = frozenset(self._maintainer.heads)
            step_savepoint = conn.begin_nested()
            try:
                log.info("Running %s", step)

                # 1) the migration function really executes first;
                # inside it, get_current_heads() still sees the
                # pre-mutation rows.
                step.migration_fn(**run_kwargs)

                # 2) only then mutate the version table, stepwise,
                # potentially with multiple row changes.
                row_changes = self._maintainer.apply(step)
                heads_after = frozenset(self._maintainer.heads)

                # 3) every step boundary must be a state from which
                # upgrading towards the script heads remains possible.
                state = self.script.classify_version_rows(heads_after)
                if not state.is_effective_head_state:
                    raise SequenceBarrierError(
                        "step %s left a non-effective version row state: "
                        "%r" % (step.short_log, state)
                    )

                # 4) callbacks observe ONLY the post-mutation rows.
                for callback in (
                    tuple(ctx.on_version_apply_callbacks)
                    + self._extra_callbacks
                ):
                    callback(
                        ctx=ctx,
                        step=step.info,
                        heads=set(heads_after),
                        run_args=run_kwargs,
                    )
            except BaseException:
                frame.failed = True
                with contextlib.suppress(Exception):
                    step_savepoint.rollback()
                raise
            else:
                step_savepoint.commit()

            record = SequenceStepRecord(
                step, heads_before, heads_after, row_changes
            )
            self.history.append(record)
            records.append(record)
            log.debug("sequence step applied: %r", record)

        return tuple(records)

    # -- internals ------------------------------------------------------------

    def _frames(self) -> list[_BarrierFrame]:
        frames = getattr(self.context, _FRAMES_ATTR, None)
        if frames is None:
            frames = []
            setattr(self.context, _FRAMES_ATTR, frames)
        return frames

    @staticmethod
    def _fail(frame: _BarrierFrame) -> None:
        frame.failed = True

    def _rollback(self, frame: _BarrierFrame) -> None:
        with contextlib.suppress(Exception):
            frame.savepoint.rollback()
        if frame.anchor_transaction is not None:
            with contextlib.suppress(Exception):
                frame.anchor_transaction.rollback()

    def _install_op_proxy(self) -> None:
        # ``alembic.op`` proxies through one module-global slot; save
        # whatever an outer/parallel frame installed, then install ours.
        from .. import op as op_module

        self._saved_op_proxy = getattr(op_module, "_proxy", None)
        self._operations = Operations(self.context)
        self._operations._install_proxy()

    def _restore_op_proxy(self) -> None:
        if self._operations is not None:
            with contextlib.suppress(Exception):
                self._operations._remove_proxy()
            if self._saved_op_proxy is not None:
                with contextlib.suppress(Exception):
                    self._saved_op_proxy._install_proxy()
            self._operations = None
            self._saved_op_proxy = None
