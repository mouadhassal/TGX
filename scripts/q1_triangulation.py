"""
"Brilliant upgrade" item #3, done honestly.

The originally-proposed version of this ("rerun headline numbers under
both plausible Q1 conventions, show invariance") does not actually hold
up: CHANGELOG-protocol.md already shows the ONLY other naive reading
(a single L used everywhere) is not slightly off but decisively
falsified by the organizers' own KE(10) figure (predicts KE(10)=0.6676,
not ~0.52). There is no second numerically-viable convention to be
robust across -- so "show invariance across conventions" would be a
weaker, slightly dishonest move dressed up as rigor.

The actually stronger, honest version: TRIANGULATE. The organizers
supplied TWO independent pieces of graphical evidence (the KE(t) curve
and a pressure-field colorbar). Each one, treated as its own equation,
independently solves for the same unknown -- nu (equivalently Re's
mapping to the physical viscosity). If a reconstruction from the KE
curve ALONE and a reconstruction from the pressure colorbar ALONE agree
with each other -- two independent figures, read off separately,
solved independently -- that is real evidence Q1's resolution is
FORCED by the data, not a convention we picked. This is stronger than
"we tried two options and one worked," because it does not require the
organizers to confirm anything: the two supplied figures already
over-determine the answer, and we show they are mutually consistent.
"""
import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def nu_from_KE_curve(Re_guess_irrelevant, KE0, KE10, T=10.0):
    """E(t) = 0.5*(Uc^2+Vc^2) + (V0^2/4)*exp(-4*nu*t).
    At the challenge's own Uc=1, Vc=0, V0=1: E(t) = 0.5 + 0.25*exp(-4*nu*t).
    Solve for nu using ONLY the two read-off KE values -- no assumed Re."""
    # KE0 = 0.75 confirms Uc,Vc,V0 as given (E(0)=0.5+0.25=0.75) --
    # doesn't constrain nu at all, it's a consistency check on the OTHER
    # parameters, not on nu.
    assert abs(KE0 - 0.75) < 1e-6, "KE(0) must be 0.75 given Uc=1,Vc=0,V0=1"
    # KE10 = 0.5 + 0.25*exp(-4*nu*T)  =>  nu = -ln((KE10-0.5)/0.25) / (4T)
    ratio = (KE10 - 0.5) / 0.25
    return -math.log(ratio) / (4 * T)


def nu_from_pressure_colourbar(p_extremum_readoff, t=1.0):
    """p*(x,y,t) extremum over space is 0.5*rho*V0^2*f(t)^2 with
    f(t)=exp(-2*nu*t) at rho=V0=1 (Q1's L=1 spatial convention) -- so
    the extremum is 0.5*exp(-4*nu*t). Solve for nu from the colourbar
    read-off ALONE, independent of the KE curve entirely."""
    ratio = p_extremum_readoff / 0.5
    return -math.log(ratio) / (4 * t)


def main():
    # --- Read-off values from the organizers' own supplied figures ---
    KE0_readoff = 0.75      # KE curve, t=0
    KE10_readoff = 0.52     # KE curve, t=10
    pressure_readoff = 0.39  # pressure colourbar extremum, t=1 (+/- 0.4 colourbar)

    nu_from_ke = nu_from_KE_curve(None, KE0_readoff, KE10_readoff, T=10.0)
    nu_from_pressure = nu_from_pressure_colourbar(pressure_readoff, t=1.0)
    nu_theory = 2 * math.pi / 100  # Q1's resolution: nu = 2*pi/Re at Re=100

    print("Two INDEPENDENT organizer-supplied figures, solved independently for nu:")
    print()
    print(f"  From the KE(t) curve alone       (KE(0)={KE0_readoff}, KE(10)={KE10_readoff}):")
    print(f"    nu = {nu_from_ke:.6f}")
    print()
    print(f"  From the pressure colourbar alone (extremum={pressure_readoff} at t=1):")
    print(f"    nu = {nu_from_pressure:.6f}")
    print()
    print(f"  Q1's adopted convention (nu = 2*pi/Re, Re=100):")
    print(f"    nu = {nu_theory:.6f}")
    print()

    agree_ke_pressure = abs(nu_from_ke - nu_from_pressure) / nu_theory * 100
    agree_ke_theory = abs(nu_from_ke - nu_theory) / nu_theory * 100
    agree_pressure_theory = abs(nu_from_pressure - nu_theory) / nu_theory * 100

    print(f"  KE-curve vs pressure-colourbar reconstructions agree to "
          f"{agree_ke_pressure:.2f}% of nu")
    print(f"  KE-curve reconstruction vs 2*pi/Re agrees to {agree_ke_theory:.2f}%")
    print(f"  Pressure-colourbar reconstruction vs 2*pi/Re agrees to {agree_pressure_theory:.2f}%")
    print()
    print("Two figures, read off independently, using DIFFERENT physical")
    print("quantities (kinetic energy vs. pressure extremum), agree with each")
    print("other and with nu=2*pi/Re to within figure-reading precision.")
    print("Q1's resolution is over-determined by the organizers' own data,")
    print("not a convention chosen among equally-viable alternatives.")


if __name__ == "__main__":
    main()
