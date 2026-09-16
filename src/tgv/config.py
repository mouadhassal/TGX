"""
Physical and numerical parameters for the convecting 2D Taylor-Green Vortex,
under the Q1 convention adopted in airbus-tgv-constitution-v2.md §1.2.

Q1 [ASSUMPTION]: domain Omega = [0, 2*pi]^2, doubly periodic; the length
scale L appearing in the exact solution and in nu = V0*L/Re is L = 1
(NOT the domain length 2*pi). This was reconstructed from the organizers'
supplied figures (KE(0)=0.75, KE(10)~=0.52 at Re=100; pressure extremum
at t=1 matching a +/-0.4 colourbar), not confirmed by the organizers.
See G0 in the constitution: this file's parameters must reproduce those
two figures or the whole plan is invalid.
"""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TGVParams:
    """
    Q1's overloaded `L` resolved as TWO distinct constants (constitution
    v2 §1.2, Q1): the spatial/decay length scale inside the exact solution
    (sin/cos arguments and the exponent's L^2) is `L = 1`; the length scale
    that enters the viscosity formula nu = V0*L/Re is the DOMAIN length,
    2*pi. Using a single L for both (as the challenge text's notation
    naively suggests) fails G0 -- see CHANGELOG-protocol.md.
    """
    Re: float
    L: float = 1.0                       # spatial/decay scale in u*, p*
    V0: float = 1.0
    Uc: float = 1.0
    Vc: float = 0.0
    rho: float = 1.0
    p0: float = 0.0
    domain_length: float = 2.0 * math.pi  # Omega = [0, 2*pi]^2

    @property
    def nu(self) -> float:
        # nu = V0 * domain_length / Re  (per Q1: the L in this formula is
        # the domain length, NOT the spatial/decay scale L above)
        return self.V0 * self.domain_length / self.Re


# Cell-Reynolds resolution rule (Q6 [DERIVED]):
#   Re_Delta = |u| * dx / nu <= 2,  dx = domain_length / N,  nu = 2*pi/Re
#   => N >= Re / 2
def N_of_Re(Re: float) -> int:
    n = max(32, Re / 2.0)
    return 2 ** math.ceil(math.log2(n))
