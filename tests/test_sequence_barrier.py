"""Acceptance tests for the host-facing cross-dependency sequence barrier.

The graph under test has two version branches plus cross-branch
``depends_on`` edges::

    a1 -> a2
     \\     ^
      b1 -> b2       b1 depends_on a1, b2 depends_on a2

so that script-directory heads (version edges only: ``a2``, ``b2``),
dependency tips (``b2`` alone once ``depends_on`` is counted) and the
live ``alembic_version`` rows are three observably different answers.

All tests run on a single shared in-memory SQLite connection
(StaticPool); no external database service is involved.

The assertions are deliberately written as time-ordered causal chains,
not as a pattern/result enumeration table.
"""

from sqlalchemy import create_engine
from sqlalchemy import inspect
from sqlalchemy.pool import StaticPool

from alembic import util
from alembic.migration import MigrationContext
from alembic.migration import plan_version_row_changes
from alembic.migration import StepwiseVersionMaintainer
from alembic.migration import VersionRowChange
from alembic.operations import Operations
from alembic.runtime.sequence import DowngradeMove
from alembic.runtime.sequence import MigrationSequence
from alembic.runtime.sequence import SequenceBarrierError
from alembic.runtime.sequence import UpgradeMove
from alembic.testing import assert_raises
from alembic.testing import assert_raises_message
from alembic.testing import eq_
from alembic.testing.env import clear_staging_env
from alembic.testing.env import staging_env
from alembic.testing.env import write_script
from alembic.testing.fixtures import TestBase


def _make_script(
    rev,
    down,
    deps,
    table,
    *,
    branch_labels=None,
    up_extra="",
    down_body=None,
):
    """Build a revision script that performs real, observable DDL and
    logs the head rows visible *during* the migration function into a
    probe table (read back by the tests)."""

    lines = [
        "from alembic import op",
        "import sqlalchemy as sa",
        "",
        "revision = %r" % rev,
        "down_revision = %r" % down,
        "depends_on = %r" % deps,
        "branch_labels = %r" % branch_labels,
        "",
        "",
        "def _log(phase):",
        "    heads = ','.join(sorted(op.get_context().get_current_heads()))",
        "    op.execute(",
        "        \"CREATE TABLE IF NOT EXISTS seq_probe (\"",
        "        \"step VARCHAR(40), heads VARCHAR(200))\")",
        "    probe = sa.table(",
        "        'seq_probe',",
        "        sa.column('step', sa.String),",
        "        sa.column('heads', sa.String))",
        "    op.execute(",
        "        sa.insert(probe).values(",
        "            step=%r + '-' + phase, heads=heads))" % rev,
        "",
        "",
        "def upgrade():",
        "    op.create_table(%r, sa.Column('x', sa.Integer))" % table,
    ]
    if up_extra:
        for line in up_extra.strip("\n").splitlines():
            lines.append("    " + line)
    lines.append("    _log('up')")
    lines.append("")
    lines.append("")
    lines.append("def downgrade():")
    if down_body is None:
        lines.append("    op.drop_table(%r)" % table)
    else:
        for line in down_body.strip("\n").splitlines():
            lines.append("    " + line)
    lines.append("    _log('down')")
    lines.append("")
    return "\n".join(lines)


def _bind():
    # one in-memory database, one shared connection for the whole test
    engine = create_engine("sqlite://", poolclass=StaticPool)
    return engine, engine.connect()


def _tables(conn):
    return set(inspect(conn).get_table_names())


def _probe(conn):
    return dict(
        conn.exec_driver_sql(
            "SELECT step, heads FROM seq_probe"
        ).fetchall()
    )


def _step_ids(steps):
    return [step.revision.revision for step in steps]


def _kinds(record):
    return [
        (change.kind, change.value, change.from_value)
        for change in record.row_changes
    ]


def _assert_merge_changes(changes, new_rev, prior_rows):
    """A cross-dep merge is exactly one DELETE plus one UPDATE.

    The pin picks which prior row is deleted and which is updated via
    set iteration, so that split is hash-seed dependent -- but the
    update always lands on ``new_rev`` and the two consumed rows are
    exactly ``prior_rows``.  Assert that invariant rather than the
    set-order accident.
    """

    eq_(len(changes), 2)
    eq_(
        sorted(change.kind for change in changes),
        ["delete", "update"],
    )
    deleted = next(
        change.value for change in changes if change.kind == "delete"
    )
    update = next(
        change for change in changes if change.kind == "update"
    )
    eq_(update.value, new_rev)
    eq_({deleted, update.from_value}, set(prior_rows))


def _assert_unmerge_changes(changes, old_rev, resulting_rows):
    """A cross-dep unmerge is exactly one INSERT plus one UPDATE.

    Which row is inserted and which the old row is updated to is
    hash-seed dependent in the pin, but the update always starts at
    ``old_rev`` and insert + update-to together equal
    ``resulting_rows``.
    """

    eq_(len(changes), 2)
    eq_(
        sorted(change.kind for change in changes),
        ["insert", "update"],
    )
    inserted = next(
        change.value for change in changes if change.kind == "insert"
    )
    update = next(
        change for change in changes if change.kind == "update"
    )
    eq_(update.from_value, old_rev)
    eq_({inserted, update.value}, set(resulting_rows))


class CrossDepSequenceTest(TestBase):
    @classmethod
    def setup_class(cls):
        cls.sd = sd = staging_env()

        sd.generate_revision("a1", "a1", head="base", refresh=True)
        sd.generate_revision("a2", "a2", head="a1", refresh=True)
        sd.generate_revision(
            "b1", "b1", head="base", depends_on="a1", refresh=True
        )
        sd.generate_revision(
            "b2", "b2", head="b1", depends_on="a2", refresh=True
        )

        #   a1 -> a2
        #    \\     ^
        #     b1 -> b2     (b1 depends_on a1; b2 depends_on a2)
        write_script(
            sd, "a1", _make_script("a1", None, None, "ta1")
        )
        write_script(
            sd, "a2", _make_script("a2", "a1", None, "ta2")
        )
        write_script(
            sd,
            "b1",
            _make_script(
                "b1", None, "a1", "tb1", branch_labels="bbranch"
            ),
        )
        write_script(
            sd,
            "b2",
            _make_script("b2", "b1", "a2", "tb2"),
        )

    @classmethod
    def teardown_class(cls):
        clear_staging_env()

    # -- carrier A --------------------------------------------------------

    def test_versioned_heads_dependency_heads_and_rows_distinct(self):
        sd = self.sd

        # heads counting version edges only (the display answer)
        eq_(sd.get_versioned_heads(), ["a2", "b2"])
        # tips once cross-branch depends_on edges are counted
        eq_(sd.get_dependency_heads(), ["b2"])
        assert sd.get_versioned_heads() != sd.get_dependency_heads()

        # and the live row set is a third, separately classified answer
        state = sd.classify_version_rows(("b2",))
        eq_(state.effective_tips, ("b2",))
        eq_(state.stale_rows, ())
        assert state.is_effective_head_state

        # an ancestor row implied by a depends_on edge is stale
        state = sd.classify_version_rows(("a1", "a2"))
        eq_(state.stale_rows, ("a1",))
        assert not state.is_effective_head_state

        # rows unknown to the script directory are neither tips nor stale
        state = sd.classify_version_rows(("nope",))
        eq_(state.unknown_rows, ("nope",))
        assert not state.is_effective_head_state

        # a fresh database is a valid (empty) effective state, and the
        # cross-dep mid-state b1 (which implies a1 through depends_on)
        # is itself an effective tip
        assert sd.classify_version_rows(()).is_effective_head_state
        eq_(sd.classify_version_rows(("b1",)).effective_tips, ("b1",))

    # -- carrier C --------------------------------------------------------

    def test_move_resolves_fresh_against_live_rows(self):
        sd = self.sd

        # the SAME relative move answers differently depending on the
        # rows live at resolution time -- nothing is expanded at entry:
        # from b1 it pulls the implicit dependency a2 along with b2...
        move = UpgradeMove("+1")
        eq_(_step_ids(move.resolve(sd, ("b1",))), ["a2", "b2"])
        # ...while at the head the same spec produces zero migrations
        # rather than replaying a frozen two-step plan
        assert_raises(util.CommandError, move.resolve, sd, ("b2",))

        # a branch-tip target resolves against the live rows as well:
        # from b1 the branch tip b2 is only reachable through the
        # not-yet-present cross-branch dependency a2 (implicit base)
        eq_(
            _step_ids(
                UpgradeMove("bbranch@head").resolve(sd, ("b1",))
            ),
            ["a2", "b2"],
        )
        eq_(
            _step_ids(
                UpgradeMove("bbranch@head").resolve(sd, ("a2", "b1"))
            ),
            ["b2"],
        )

        # a cross depends_on downgrade resolves to the unmerge step
        eq_(
            _step_ids(DowngradeMove("-1").resolve(sd, ("b2",))),
            ["b2"],
        )

    # -- carriers B + D: the required time-ordered sequence ---------------

    def test_multi_step_sequence_causal_chain(self):
        sd = self.sd
        engine, conn = _bind()

        seen = []

        seq = MigrationSequence(
            sd,
            connection=conn,
            on_version_apply=lambda **kw: seen.append(
                (
                    kw["step"].up_revision_id,
                    tuple(sorted(kw["heads"])),
                )
            ),
        )

        with seq:
            # step 1: upgrade up to the cross-dependency branch root.
            # From an empty database this really runs a1 then b1.
            first = seq.upgrade("b1")
            eq_(seq.current_rows(), ("b1",))
            assert {"ta1", "tb1"} <= _tables(conn)
            assert "ta2" not in _tables(conn)

            eq_(_kinds(first[0]), [("insert", "a1", None)])
            # b1 implies a1 via depends_on: scalar update, not a stamp
            eq_(_kinds(first[1]), [("update", "b1", "a1")])

            # step 2: a RELATIVE target, resolved against the rows the
            # previous move left behind (b1) -- an entry-time snapshot
            # of the empty database would resolve +1 to a1 instead.
            second = seq.upgrade("+1")
            eq_([r.revision_id for r in second], ["a2", "b2"])

            # a2 opens the second version branch
            eq_(_kinds(second[0]), [("insert", "a2", None)])
            # b2 merges across the depends_on edge with two concrete
            # row mutations (one delete + one update), never one scalar
            # update pretending otherwise; the two consumed rows are
            # exactly the prior {a2, b1}
            _assert_merge_changes(
                second[1].row_changes, "b2", ("a2", "b1")
            )
            eq_(seq.current_rows(), ("b2",))
            assert {"ta2", "tb2"} <= _tables(conn)

            # step 3: a downgrade ACROSS the depends_on edge; b2's
            # downgrade function really runs (tb2 disappears) before
            # the version table is unmerged into two rows.
            third = seq.downgrade("-1")
            eq_([r.revision_id for r in third], ["b2"])
            # unmerge: one insert + one update (the pin may choose
            # either prior row for each), together restoring exactly
            # {a2, b1} -- again two concrete row mutations rather than
            # one scalar UPDATE
            _assert_unmerge_changes(
                third[0].row_changes, "b2", ("a2", "b1")
            )
            eq_(tuple(sorted(seq.current_rows())), ("a2", "b1"))
            assert "tb2" not in _tables(conn)
            assert "tb1" in _tables(conn)

            # every later record starts exactly where the previous one
            # ended -- the next move saw the previous step's mutation
            records = seq.applied
            for earlier, later in zip(records, records[1:]):
                eq_(later.heads_before, earlier.heads_after)
            # including across move boundaries
            eq_(second[0].heads_before, first[-1].heads_after)
            eq_(third[0].heads_before, second[-1].heads_after)

        # the sequence closed on an effective-head state; a fresh
        # barrier can continue upgrading towards the script heads.
        cont = MigrationSequence(sd, connection=conn)
        cont.run(UpgradeMove("heads"))
        eq_(cont.current_rows(), ("b2",))
        assert "tb2" in _tables(conn)

        # callback observations, in order, all post-mutation only
        eq_(
            seen,
            [
                ("a1", ("a1",)),
                ("b1", ("b1",)),
                ("a2", ("a2", "b1")),
                ("b2", ("b2",)),
                ("b2", ("a2", "b1")),
            ],
        )

    def test_fn_heads_fork_from_callback_heads(self):
        sd = self.sd
        engine, conn = _bind()

        seen = []
        seq = MigrationSequence(
            sd,
            connection=conn,
            on_version_apply=lambda **kw: seen.append(
                (
                    kw["step"].up_revision_id,
                    tuple(sorted(kw["heads"])),
                )
            ),
        )
        with seq:
            seq.upgrade("heads")
            seq.downgrade("-1")

        probe = _probe(conn)

        # during the b2 UPGRADE function the pre-merge rows are still
        # live; the upgrade callback only sees the post-merge single row
        eq_(probe["b2-up"], "a2,b1")

        # during the b2 DOWNGRADE function the single row is still
        # live; the callbacks are upgrade-then-downgrade, so the second
        # b2 observation is the post-unmerge two-row state
        eq_(probe["b2-down"], "b2")
        b2_callbacks = [heads for rev, heads in seen if rev == "b2"]
        eq_(b2_callbacks, [("b2",), ("a2", "b1")])

        # and a pure insert step: empty during the fn, populated in cb
        eq_(probe["a1-up"], "")
        eq_(seen[0], ("a1", ("a1",)))

    def test_real_downgrade_forks_from_stamp_only(self):
        sd = self.sd

        # real path: migration functions execute, cross-dep unmerge
        engine_a, conn_a = _bind()
        real = MigrationSequence(sd, connection=conn_a)
        with real:
            real.upgrade("heads")
            real.downgrade("-1")
        real_rows = set(real.current_rows())
        real_tables = _tables(conn_a)

        # stamp-only path: no downgrade function, scalar version patch
        engine_b, conn_b = _bind()
        stamped = MigrationSequence(sd, connection=conn_b)
        with stamped:
            stamped.upgrade("heads")
            stamped.context.stamp(sd, "b1")
        stamp_rows = set(stamped.current_rows())
        stamp_tables = _tables(conn_b)

        # the row sets measurably differ: the real downgrade keeps the
        # a2 dependency row, stamp-only does not
        eq_(real_rows, {"a2", "b1"})
        eq_(stamp_rows, {"b1"})
        assert real_rows != stamp_rows

        # and the DDL fork proves the migration function really ran
        assert "tb2" not in real_tables
        assert "tb2" in stamp_tables

    # -- nesting ----------------------------------------------------------

    def test_nested_destructive_close_rolled_back(self):
        sd = self.sd
        engine, conn = _bind()

        outer = MigrationSequence(sd, connection=conn)
        with outer:
            outer.upgrade("heads")
            eq_(outer.current_rows(), ("b2",))

            inner = MigrationSequence(sd, context=outer.context)
            assert_raises_message(
                SequenceBarrierError,
                "destructively finalized",
                self._run_inner_downgrade,
                inner,
            )

            # the inner side effects were popped: rows AND tables of
            # the outer sequence are intact
            eq_(outer.current_rows(), ("b2",))
            assert "tb2" in _tables(conn)

            # the outer sequence simply continues
            outer.downgrade("-1")
            eq_(
                tuple(sorted(outer.current_rows())), ("a2", "b1")
            )
            outer.upgrade("b2")
            eq_(outer.current_rows(), ("b2",))

        eq_(outer.current_rows(), ("b2",))
        assert outer.row_state().is_effective_head_state

    @staticmethod
    def _run_inner_downgrade(inner):
        with inner:
            inner.downgrade("-1")

    def test_nested_clean_roundtrip_is_allowed(self):
        sd = self.sd
        engine, conn = _bind()

        outer = MigrationSequence(sd, connection=conn)
        with outer:
            outer.upgrade("heads")

            inner = MigrationSequence(sd, context=outer.context)
            with inner:
                inner.downgrade("-1")
                eq_(
                    tuple(sorted(inner.current_rows())),
                    ("a2", "b1"),
                )
                inner.upgrade("b2")
                eq_(inner.current_rows(), ("b2",))

            # protected rows restored: inner changes committed, outer
            # is free to continue
            eq_(outer.current_rows(), ("b2",))
            outer.downgrade("-1")
            outer.upgrade("heads")
            eq_(outer.current_rows(), ("b2",))

    def test_inner_exception_pops_its_own_side_effects(self):
        sd = self.sd
        engine, conn = _bind()

        def fail_on_a2(**kw):
            if kw["step"].up_revision_id == "a2":
                raise RuntimeError("boom from inner callback")

        outer = MigrationSequence(sd, connection=conn)
        with outer:
            outer.upgrade("b1")
            eq_(outer.current_rows(), ("b1",))

            inner = MigrationSequence(
                sd,
                context=outer.context,
                on_version_apply=fail_on_a2,
            )
            assert_raises(
                RuntimeError,
                self._run_inner_upgrade,
                inner,
            )

            # a2's DDL and the inserted a2 row were both popped
            eq_(outer.current_rows(), ("b1",))
            assert "ta2" not in _tables(conn)
            assert "tb2" not in _tables(conn)

            # the outer sequence (which never had that callback)
            # continues and completes the same upgrade
            outer.upgrade("+1")
            eq_(outer.current_rows(), ("b2",))

    @staticmethod
    def _run_inner_upgrade(inner):
        with inner:
            inner.upgrade("+1")

    # -- parallel isolation -----------------------------------------------

    def test_parallel_barriers_do_not_share_state(self):
        sd = self.sd
        engine_a, conn_a = _bind()
        engine_b, conn_b = _bind()

        barrier_a = MigrationSequence(sd, connection=conn_a)
        barrier_b = MigrationSequence(sd, connection=conn_b)

        with barrier_a:
            barrier_a.upgrade("a1")

            with barrier_b:
                barrier_b.upgrade("b1")

                # mid-flight interleave: each barrier reads its own
                # database, its own frame stack, its own history
                eq_(barrier_a.current_rows(), ("a1",))
                eq_(barrier_b.current_rows(), ("b1",))
                frames_a = getattr(
                    barrier_a.context, "_alembic_sequence_frames"
                )
                frames_b = getattr(
                    barrier_b.context, "_alembic_sequence_frames"
                )
                assert frames_a is not frames_b
                assert len(barrier_a.applied) == 1
                assert len(barrier_b.applied) == 2

            # B closed and committed; A's open frame is undisturbed
            eq_(barrier_a.current_rows(), ("a1",))
            assert "tb1" not in _tables(conn_a)

            barrier_a.upgrade("heads")
            eq_(barrier_a.current_rows(), ("b2",))

        # two independent end states
        eq_(barrier_a.current_rows(), ("b2",))
        eq_(barrier_b.current_rows(), ("b1",))
        assert {"ta2", "tb2"} <= _tables(conn_a)
        assert {"ta2", "tb2"}.isdisjoint(_tables(conn_b))

    def test_two_maintainers_on_two_contexts(self):
        sd = self.sd
        engine_a, conn_a = _bind()
        engine_b, conn_b = _bind()

        MigrationSequence(sd, connection=conn_a).run(UpgradeMove("b1"))
        MigrationSequence(sd, connection=conn_b).run(UpgradeMove("a1"))

        ctx_a = MigrationContext.configure(
            connection=conn_a, opts={"script": sd}
        )
        ctx_b = MigrationContext.configure(
            connection=conn_b, opts={"script": sd}
        )
        maint_a = StepwiseVersionMaintainer(ctx_a)
        maint_b = StepwiseVersionMaintainer(ctx_b)

        eq_(maint_a.refresh(), {"b1"})
        eq_(maint_b.refresh(), {"a1"})

        # advancing A must not touch B's rows
        step_a = sd._upgrade_revs("b2", ("b1",))
        for step in step_a:
            maint_a.apply(step)
        eq_(maint_a.heads, {"b2"})
        eq_(maint_b.refresh(), {"a1"})

    # -- guards -----------------------------------------------------------

    def test_offline_and_unbound_rejected(self):
        sd = self.sd

        offline = MigrationContext.configure(
            dialect_name="sqlite",
            opts={"as_sql": True, "script": sd},
        )
        assert_raises_message(
            SequenceBarrierError,
            "offline",
            self._enter,
            MigrationSequence(sd, context=offline),
        )

        assert_raises_message(
            SequenceBarrierError,
            "online bind",
            MigrationSequence,
            sd,
        )

    @staticmethod
    def _enter(seq):
        with seq:
            pass

    # -- pin compatibility ------------------------------------------------

    def test_single_shot_pin_behavior_unchanged(self):
        sd = self.sd

        # pin: MigrationContext.run_migrations directly
        engine_a, conn_a = _bind()
        ctx_a = MigrationContext.configure(
            connection=conn_a,
            opts={
                "script": sd,
                "fn": lambda rev, context: sd._upgrade_revs(
                    "heads", rev
                ),
            },
        )
        with Operations.context(ctx_a):
            ctx_a.run_migrations()
        pin_rows = tuple(sorted(ctx_a.get_current_heads()))
        pin_tables = _tables(conn_a)

        # barrier: upgrade heads
        engine_b, conn_b = _bind()
        seq = MigrationSequence(sd, connection=conn_b)
        with seq:
            seq.upgrade("heads")
        barrier_rows = tuple(sorted(seq.current_rows()))
        barrier_tables = _tables(conn_b)

        eq_(pin_rows, barrier_rows)
        eq_(pin_rows, ("b2",))
        eq_(pin_tables, barrier_tables)

        # pin downgrade across the depends_on edge
        ctx_a2 = MigrationContext.configure(
            connection=conn_a,
            opts={
                "script": sd,
                "fn": lambda rev, context: sd._downgrade_revs(
                    "-1", rev
                ),
            },
        )
        with Operations.context(ctx_a2):
            ctx_a2.run_migrations()
        eq_(tuple(sorted(ctx_a.get_current_heads())), ("a2", "b1"))

        with seq:
            seq.downgrade("-1")
        eq_(tuple(sorted(seq.current_rows())), ("a2", "b1"))

    def test_planned_row_changes_match_pin_dispatch(self):
        sd = self.sd

        eq_(
            plan_version_row_changes(
                sd._upgrade_revs("a1", None)[0], set()
            ),
            (VersionRowChange("insert", "a1"),),
        )
        eq_(
            plan_version_row_changes(
                sd._upgrade_revs("b1", ("a1",))[0], {"a1"}
            ),
            (VersionRowChange("update", "b1", "a1"),),
        )
        _assert_merge_changes(
            plan_version_row_changes(
                sd._upgrade_revs("b2", ("a2", "b1"))[0],
                {"a2", "b1"},
            ),
            "b2",
            ("a2", "b1"),
        )
        _assert_unmerge_changes(
            plan_version_row_changes(
                sd._downgrade_revs("-1", ("b2",))[0], {"b2"}
            ),
            "b2",
            ("a2", "b1"),
        )
        eq_(
            plan_version_row_changes(
                sd._downgrade_revs("base", ("a1",))[0], {"a1"}
            ),
            (VersionRowChange("delete", "a1"),),
        )


class SequenceFailureTest(TestBase):
    """Safe-state recovery on a graph that also contains a revision
    whose upgrade function fails."""

    @classmethod
    def setup_class(cls):
        cls.sd = sd = staging_env()

        sd.generate_revision("a1", "a1", head="base", refresh=True)
        sd.generate_revision("a2", "a2", head="a1", refresh=True)
        sd.generate_revision(
            "b1", "b1", head="base", depends_on="a1", refresh=True
        )
        sd.generate_revision(
            "b2", "b2", head="b1", depends_on="a2", refresh=True
        )
        sd.generate_revision("c1", "c1", head="base", refresh=True)
        sd.generate_revision("c2", "c2", head="c1", refresh=True)

        write_script(sd, "a1", _make_script("a1", None, None, "ta1"))
        write_script(sd, "a2", _make_script("a2", "a1", None, "ta2"))
        write_script(
            sd, "b1", _make_script("b1", None, "a1", "tb1")
        )
        write_script(
            sd, "b2", _make_script("b2", "b1", "a2", "tb2")
        )
        write_script(sd, "c1", _make_script("c1", None, None, "tc1"))
        write_script(
            sd,
            "c2",
            _make_script(
                "c2",
                "c1",
                None,
                "tc2",
                up_extra="raise RuntimeError('boom c2')",
                down_body="pass",
            ),
        )

    @classmethod
    def teardown_class(cls):
        clear_staging_env()

    def test_failure_returns_to_detectable_safe_state(self):
        sd = self.sd
        engine, conn = _bind()

        seq = MigrationSequence(sd, connection=conn)
        assert_raises(
            RuntimeError,
            self._fail_inside_sequence,
            seq,
        )

        # the outermost frame rolled back to its entry state: no rows,
        # no tables, and the row state is classifiable as safe
        fresh = MigrationSequence(sd, connection=conn)
        eq_(fresh.current_rows(), ())
        assert fresh.row_state().is_effective_head_state
        assert {"ta1", "tc1", "tc2"}.isdisjoint(_tables(conn))

        # the poisoned object cannot be reopened ...
        assert_raises_message(
            SequenceBarrierError,
            "already failed",
            self._enter,
            seq,
        )

        # ... but a NEW barrier on the same bind proceeds normally on
        # an unrelated branch
        fresh.run(UpgradeMove("a1"))
        eq_(fresh.current_rows(), ("a1",))
        assert "ta1" in _tables(conn)

    @staticmethod
    def _enter(seq):
        with seq:
            pass

    @staticmethod
    def _fail_inside_sequence(seq):
        with seq:
            seq.upgrade("c1")
            seq.upgrade("c2")
