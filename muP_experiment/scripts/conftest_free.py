"""``one_batch`` of ``tests/conftest.py`` for scripts (no pytest import)."""
import dataclasses


def one_batch(runs):
    """Give the ungated runs the gated runs' κ and T so all five readouts share one vmap batch.

    κ does not enter an ungated run's dynamics (pN = N, g = 1); it only sets T and the age schedule.
    """
    g = next(r for r in runs if r.gated == 1.0)
    return [r if r.gated == 1.0 else dataclasses.replace(r, kappa=g.kappa, kappa_label=g.kappa_label, kappa_sched=g.kappa_sched,
                                                         T_burn=g.T_burn, T_meas=g.T_meas) for r in runs]
