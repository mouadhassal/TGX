"""
WS4: quantum resource analysis. Every formula here is tagged; nothing is
invented (Rule 1). Two real, fetched sources ground this module:

[FACT: An, Onwunta, Yang, "Fast-forwarding quantum algorithms for linear
dissipative differential equations," Quantum 10, 1986 (2026),
arXiv:2410.13189] -- for a GENERIC dissipative linear ODE (does not
assume the generator is diagonalizable by an efficient unitary), the
best known complexity is:
    history-state prep:  O~( log(T) * (log(1/eps))^2 )   -- exp. speedup in T
    final-state prep:    O~( sqrt(T) * (log(1/eps))^2 )  -- poly speedup in T
This is the general benchmark D2 must be compared against, not "no
speedup at all."

[FACT: Jennings, Korzekwa, Lostaglio, Ashworth, Marsili, Rolston, "An
end-to-end quantum algorithm for nonlinear fluid dynamics with bounded
quantum advantage," arXiv:2512.03758] -- their QLBM algorithm's cost
including data-extraction overhead q_M ~ O(Re^{3/8}) (for drag-type
observables) gives an analytic LOWER bound O(Re^{3/4*(1+D/2)} * q_M):
at D=2 this is Re^{1.5} * Re^{0.375} = Re^{1.875} TOTAL -- the two
factors multiply, they do not collapse to Re^{3D/8} (=Re^{0.75} at
D=2). An earlier draft of this module stated that false collapsed
form, sourced from an imprecise first-pass summary rather than the
paper itself; re-fetched and corrected 2026-08-27 (see AUDIT.md B3).
Their own NUMERICAL/empirical estimate (Sec 4.4, "Estimating query
complexity", from simulating actual Carleman matrices up to dimension
~10^8) is lower than the analytic bound: O(Re^1.936 * q_M) for D=2.
D = spatial dimension (2 here).

[FACT: Yang, Onwunta, An, "Fast-forwarding quantum algorithms for
weakly nonlinear dissipative differential equations and beyond,"
arXiv:2608.25822] -- extends the linear fast-forwarding result above to
WEAKLY NONLINEAR dissipative ODEs -- exactly K1's regime (a small
perturbation eps breaking D1's exact linearity). Their headline result:
complexity does NOT explicitly depend on T at all -- O(1), improving
the linear case's own previous state-of-the-art Õ(sqrt(T)) -- with
"any remaining time dependence enter[ing] through the output norm and
forcing parameters" (i.e. through eps, not T). This directly grounds
K1's correction-term cost as T-independent, replacing what would
otherwise be an invented scaling assumption.

Everything else (our own algorithm's structural advantage, the QLSA/
Carleman reference formula, the definitional time-marching baseline
used for G3) is [DERIVED] from standard, named results, tagged inline.
"""
import numpy as np


# ---------------------------------------------------------------------
# D2 / our algorithm (Sec 3.1a): structural comparison against the
# general fast-forwarding literature bound.
# ---------------------------------------------------------------------

def general_fastforward_query_complexity(T: float, eps: float, mode: str = "final_state"):
    """
    [FACT: An-Onwunta-Yang 2026] Best known GENERAL (non-structure-
    exploiting) complexity for preparing the solution of a dissipative
    linear ODE at time T to precision eps. Returns the tilde-O exponent
    structure as a dict (T-scaling and log(1/eps)-scaling), NOT a raw
    gate count (the hidden polylog/dimension factors are not stated in
    the source we fetched -- reporting a bare number here would violate
    Rule 1).
    """
    unit = ("dimensionless scaling proxy (tilde-O exponent structure only -- "
            "the source's hidden polylog/dimension constants are NOT known, so "
            "this is NOT in gate units and must never be compared numerically "
            "against our_algorithm_query_complexity's total_gates_proxy or any "
            "other function's gate-unit output; AUDIT.md minor)")
    if mode == "history_state":
        return {"T_scaling": "log(T)", "eps_scaling": "(log(1/eps))^2",
                "value_proxy": np.log(max(T, 1.0)) * (np.log(1.0 / eps)) ** 2,
                "unit": unit}
    elif mode == "final_state":
        return {"T_scaling": "sqrt(T)", "eps_scaling": "(log(1/eps))^2",
                "value_proxy": np.sqrt(max(T, 0.0)) * (np.log(1.0 / eps)) ** 2,
                "unit": unit}
    raise ValueError(mode)


def our_algorithm_query_complexity(n_qubits: int, T: float, eps: float, k1_eps: float = 0.0):
    """
    Our algorithm (Sec 3.1a), two regimes:

    k1_eps = 0 (base TGV, D1/D2 hold exactly): the generator is exactly
    circulant, hence exactly diagonalized by a QFT -- reaching ANY time
    T costs a SINGLE QFT + diagonal phase/decay + inverse-QFT
    application, cost INDEPENDENT of T. [DERIVED from D2] -- strictly
    better than even the general dissipative-ODE bound
    (Õ(sqrt(T)), An-Onwunta-Yang 2026), because that bound doesn't
    assume an efficient diagonalizing unitary is known; ours does (D3:
    the generator's own structure IS the diagonalizing unitary).

    k1_eps > 0 (K1 knob, weakly nonlinear): [FACT: Yang, Onwunta, An,
    arXiv:2608.25822] proves complexity for weakly nonlinear dissipative
    ODEs is ALSO O(1) in T (improving their own linear-case Õ(sqrt(T))),
    with remaining dependence entering through the forcing strength
    (here, k1_eps) and output norm, not T. We therefore model the LCU
    correction cost as T-independent and monotonic in k1_eps, per that
    result -- not an invented T-scaling assumption.

    QFT gate count on n qubits to precision eps: O(n*log(n/eps))
    [FACT: approximate QFT, Coppersmith 1994 / standard result].
    """
    qft_gates = n_qubits * np.log(max(n_qubits / eps, 1.0))
    # [FACT: arXiv:2608.25822] grounds T-INDEPENDENCE only, not this
    # specific linear form or its coefficient=1 -- AUDIT.md minor: an
    # earlier version's inline tag claimed more than the source
    # supports. The functional form below (linear in k1_eps, unit
    # coefficient) is [ASSUMPTION: monotone increasing in k1_eps,
    # vanishing at k1_eps=0; exact form/coefficient not established by
    # the cited source].
    lcu_correction_gates = 0.0 if k1_eps == 0.0 else k1_eps * qft_gates
    return {
        "T_scaling": "O(1) -- exact (k1_eps=0, D2) or per arXiv:2608.25822 (k1_eps>0)",
        "qft_gates": float(qft_gates),
        "lcu_correction_gates": float(lcu_correction_gates),
        "total_gates_proxy": float(2 * qft_gates + lcu_correction_gates),
        "unit": "gates (state-prep only -- NOT end-to-end; see "
                "our_algorithm_end_to_end_query_complexity for readout-inclusive cost)",
    }


def our_algorithm_end_to_end_query_complexity(n_qubits: int, T: float, eps_state: float,
                                               eps_readout: float, k1_eps: float = 0.0):
    """
    [DERIVED] End-to-end cost = state-prep gates x number of shots
    needed for readout, per Q5 (state prep AND readout always count).
    The readout model reuses D6's global-functional sample complexity
    (extraction.py): O(1/eps_readout^2) shots, independent of N -- valid
    for a global observable like mean KE, which is what the challenge's
    required curves report. AUDIT.md condition (i): the earlier
    `our_algorithm_query_complexity` alone was state-prep only and was
    NOT the quantity G3 asks about; this function is.
    """
    from .extraction import local_observable_samples_needed

    prep = our_algorithm_query_complexity(n_qubits, T, eps_state, k1_eps)
    n_shots = local_observable_samples_needed(n_outputs=1, eps=eps_readout)
    total = n_shots * prep["total_gates_proxy"]
    return {
        "state_prep_gates_per_shot": prep["total_gates_proxy"],
        "n_shots_for_readout": n_shots,
        "total_gates_proxy": float(total),
        "unit": "gates, end-to-end (state prep x shots, global-observable readout)",
    }


def lbm_acoustic_scaling_dt(dx: float, lattice_velocity: float = 1.0) -> float:
    """
    [FACT: standard LBM result] LBM's own timestep is set by ACOUSTIC
    scaling, dt = dx/lambda with a fixed lattice velocity lambda --
    LINEAR in dx, not quadratic. An earlier version of this module
    anchored the G3 reference dt to an explicit-finite-difference
    diffusive stability limit (dt ~ dx^2/(4*nu)); that is a category
    substitution -- QLBM's dt is not set by an FD von Neumann limit at
    all (AUDIT-2 §1). lambda ~ O(1), comparable to the flow's own
    characteristic velocity (Uc=1 here), is the GENEROUS choice for a
    lower-bound argument: a smaller lambda gives a LARGER dt, i.e. a
    CHEAPER (more favorable-to-the-reference) time-marching cost.

    COHERENCE CHECK (AUDIT-3 §0, verified): under LBM acoustic scaling
    the physical viscosity is nu = nu_lattice*dx (a standard LBM
    relation), so Re = U*L/nu ~ 1/dx ~ N -- the SAME N ~ Re relation
    the pre-registered Q6 cell-Reynolds rule (config.py's N_of_Re)
    gives from a completely independent derivation (Re_Delta =
    |u|*dx/nu <= 2). Two unrelated routes landing on the same scaling
    is a nontrivial cross-check that the resolution rule and the LBM
    reference's own timestep rule are not accidentally in tension.
    """
    return dx / lattice_velocity


def time_marching_query_complexity(T: float, dt: float, gates_per_step: float = 1.0):
    """
    [DERIVED, definitional] ANY genuine EXPLICIT time-marching
    implementation (apply a fixed short-time-step unitary/channel
    repeatedly) costs at least n_steps = T/dt applications by
    definition -- no citation needed for this structural fact, only for
    what dt and gates_per_step should be for a specific reference
    algorithm (which we do not have precise numbers for; see the G3
    test's honesty note). Returns total gate proxy = n_steps *
    gates_per_step, growing linearly in T for any fixed dt,
    gates_per_step > 0.

    SCOPE (AUDIT-2 §1): this bound applies to EXPLICIT references only.
    QLBM (the reference this is used against) is explicit by
    construction (lattice Boltzmann is an explicit streaming-collision
    scheme), so the scope matches the actual comparison being made --
    but the T/dt argument does NOT apply to an implicit or global-in-
    time reference, which has no stability-derived upper bound on dt at
    all; defeating such a reference would need a different argument
    (per-step/per-solve cost, e.g. the condition number of an implicit
    linear solve), which this function does not make and which is not
    attempted here.
    """
    n_steps = T / dt
    return {"n_steps": float(n_steps),
            "total_gates_proxy": float(n_steps * gates_per_step),
            "unit": "gates (state-prep/time-marching only -- NOT end-to-end); "
                    "EXPLICIT references only, see docstring scope note"}


def time_marching_end_to_end_query_complexity(T: float, dt: float, eps_readout: float,
                                               gates_per_step: float = 1.0):
    """
    [DERIVED] End-to-end reference cost, mirroring
    our_algorithm_end_to_end_query_complexity's readout model exactly
    (same D6 global-functional shot count) so the SAME multiplier
    applies to both sides of the G3 comparison -- AUDIT.md condition
    (ii): stated explicitly here (and cancels in the ratio) rather than
    silently assumed.
    """
    from .extraction import local_observable_samples_needed

    march = time_marching_query_complexity(T, dt, gates_per_step)
    n_shots = local_observable_samples_needed(n_outputs=1, eps=eps_readout)
    total = n_shots * march["total_gates_proxy"]
    return {
        "march_gates_per_shot": march["total_gates_proxy"],
        "n_shots_for_readout": n_shots,
        "total_gates_proxy": float(total),
        "unit": "gates, end-to-end (time-marching x shots, global-observable readout)",
    }


# ---------------------------------------------------------------------
# Reference method: QLBM (Jennings et al.)
# ---------------------------------------------------------------------

def qlbm_reference_cost(Re: float, D: int = 2, empirical: bool = True):
    """
    [FACT: Jennings et al. arXiv:2512.03758, re-verified 2026-08-27]
    Their end-to-end QLBM cost, including the data-extraction overhead
    q_M ~ Re^{3/8} for drag-type observables:
      analytic bound:   O(Re^{3/4*(1+D/2)} * q_M)  -- the two factors
                        MULTIPLY (exponents add: 1.875 at D=2), they do
                        NOT collapse to Re^{3D/8} -- see module docstring.
      empirical (2D):   O(Re^1.936 * q_M)  -- their Sec 4.4, lower than
                        the analytic bound (numerical Carleman-matrix
                        simulation up to dimension ~10^8)
    Returns the exponent used and the resulting proxy value (NOT a gate
    count in absolute units -- the paper's stated result is a scaling
    law, and reporting an absolute gate count from a scaling law alone
    would be inventing a constant, which Rule 1 forbids).
    """
    q_M = Re ** 0.375  # Re^{3/8}
    if empirical:
        exponent = 1.936
        scaling = (Re ** exponent) * q_M
        label = "empirical 2D: Re^1.936 * q_M"
    else:
        exponent = 0.75 * (1 + D / 2.0)
        scaling = (Re ** exponent) * q_M
        label = f"analytic bound: Re^(3/4*(1+D/2)) * q_M, D={D}"
    return {"label": label, "q_M": float(q_M), "scaling_proxy": float(scaling)}


# ---------------------------------------------------------------------
# Reference method: Carleman + QLSA
# ---------------------------------------------------------------------

def carleman_qlsa_reference_cost(kappa_lifted: float, eps: float, carleman_order: int):
    """
    [FACT: Childs-Kothari-Somma / HHL-family result -- QLSA query
    complexity O(kappa * polylog(1/eps))], applied to the
    CARLEMAN-LIFTED linear system at the given truncation order.

    Rule 1 / AUDIT.md B4: `kappa_lifted` MUST be the condition number of
    the ACTUAL LIFTED operator at this order, not the base (order-1)
    system's kappa. Carleman truncation at order k lifts an
    n-dimensional system to dimension ~ sum_{j<=k} n^j ~ n^k [FACT:
    standard Carleman embedding fact] -- the lifted operator's
    conditioning is NOT, in general, kappa_base * k; an earlier version
    of this function computed exactly that linear-in-order model, which
    understates cost by orders of magnitude for k>1 and was flagged as
    wrong (not merely under-tagged) in the audit. We do not have a
    derived or cited formula for how kappa grows with order for THIS
    specific problem, so this function REFUSES to compute that growth
    internally -- the caller must already have derived/cited
    kappa_lifted for the order in question. Passing a naive
    kappa_base*order estimate here would silently reintroduce the same
    bug; carleman_order is accepted only for bookkeeping/labeling, not
    used in the formula.

    Does not include the convergence-failure regime (R>=1, D1's
    consequence): this formula is only meaningful when Carleman
    convergence holds, which per D1(2) is optimistic-by-construction on
    this benchmark and must be flagged wherever this function is used.
    """
    query_count = kappa_lifted * np.log(1.0 / eps)
    return {"query_count_proxy": float(query_count),
            "carleman_order": carleman_order,
            "caveat": "kappa_lifted must be the LIFTED system's own condition number "
                      "at this order (not kappa_base*order -- see AUDIT.md B4); "
                      "valid only if Carleman ratio R<1, and D1 shows R on this "
                      "benchmark is suppressed by O(dx^p), optimistic by construction"}
