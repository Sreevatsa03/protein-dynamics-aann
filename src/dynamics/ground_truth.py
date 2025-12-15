import math
import torch
from torch.distributions import LogNormal


def _torch_rand_complex(rng: torch.Generator, shape, device, dtype_real):
    """
    Sample a complex tensor with real/imag ~ N(0,1).

    :param rng: torch random number generator
    :type rng: torch.Generator
    :param shape: shape of the output tensor
    :type shape: tuple
    :param device: torch device
    :type device: torch.device
    :param dtype_real: real dtype (torch.float32 or torch.float64)
    :type dtype_real: torch.dtype
    :return: complex tensor of given shape
    :rtype: torch.Tensor
    """
    re = torch.randn(*shape, generator=rng, device=device, dtype=dtype_real)
    im = torch.randn(*shape, generator=rng, device=device, dtype=dtype_real)
    return re + 1j * im


def build_hebbian_W_from_phase_patterns_torch(
    phase_patterns: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Build a real-valued coupling matrix W (N,N) that promotes phase-locked attractors
    corresponding to K phase patterns.

    :param phase_patterns: (K,N) tensor of phases in radians
    :type phase_patterns: torch.Tensor
    :param scale: scaling factor for W, defaults to 1.0
    :type scale: float, optional
    :return: coupling matrix W
    :rtype: torch.Tensor
    """
    if phase_patterns.ndim != 2:
        raise ValueError("phase_patterns must have shape (K, N)")

    K, N = phase_patterns.shape
    device = phase_patterns.device
    dtype_real = phase_patterns.dtype

    # unit complex vectors for each pattern
    P = torch.exp(1j * phase_patterns.to(dtype_real))  # (K, N) complex

    # complex hebbian sum: sum_k p_k p_k^H
    Wc = torch.zeros((N, N), device=device, dtype=torch.complex64 if dtype_real ==
                     torch.float32 else torch.complex128)
    for k in range(K):
        pk = P[k].reshape(N, 1)  # (N,1)
        Wc = Wc + pk @ torch.conj(pk).T

    W = torch.real(Wc).to(dtype_real)
    W.fill_diagonal_(0.0)

    max_abs = torch.max(torch.abs(W))
    if max_abs < 1e-8:
        raise ValueError("Hebbian W is near-zero; check phase patterns.")

    W = W / max_abs
    return W * scale


def sample_random_phase_patterns(
    K: int,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 0,
) -> torch.Tensor:
    """
    Returns (K,N) phases uniform on [0, 2pi).

    :param K: number of patterns
    :type K: int
    :param N: dimensionality of each pattern
    :type N: int
    :param device: torch device
    :type device: torch.device
    :param dtype: torch data type
    :type dtype: torch.dtype
    :param seed: random seed for reproducibility, defaults to 0
    :type seed: int, optional
    :return: (K,N) tensor of phases in radians
    :rtype: torch.Tensor
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    phases = (2.0 * math.pi) * torch.rand((K, N), generator=g, dtype=dtype)
    return phases.to(device=device)


class StuartLandauNetwork:
    """
    Coupled Stuart–Landau (Hopf) oscillator network.\\
    Observed state is x = Re(z) mapped optionally to [0,1] via sigmoid.

    Dynamics per node i:
        - z_i = x_i + i y_i
        - dz/dt = (μ - |z|^2) z + i ω z + κ (W @ z)

    This guarantees stable limit cycles for μ>0, and with coupling can produce
    phase-locked attractors (multiple "memories") when W is constructed via Hebbian phases.

    Notes:
        - W is real (N,N), applied to complex z.
        - ω is real (N,), per-node intrinsic frequencies.

    :param W: Coupling weight matrix (N,N)
    :type W: torch.Tensor
    :param mu: Intrinsic growth parameter, defaults to 1.0
    :type mu: float, optional
    :param kappa: Coupling strength, defaults to 0.6
    :type kappa: float, optional
    :param omega: Intrinsic frequencies (N,), defaults to None (randomly sampled)
    :type omega: torch.Tensor, optional
    :param omega_range: Range (low, high) for uniform sampling of omega if omega is None, defaults to (1.5, 2.5)
    :type omega_range: tuple[float, float], optional
    :param device: torch device, defaults to None (CPU)
    :type device: torch.device, optional
    :param dtype: torch data type, defaults to torch.float32
    :type dtype: torch.dtype, optional
    :param seed: Random seed for reproducibility, defaults to 0
    :type seed: int, optional
    """

    def __init__(
        self,
        W: torch.Tensor,
        mu: float = 1.0,
        kappa: float = 0.6,
        omega: torch.Tensor | None = None,
        omega_range: tuple[float, float] = (1.5, 2.5),
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
    ):
        if device is None:
            device = torch.device("cpu")

        self.device = device
        self.dtype = dtype

        W = W.to(device=device, dtype=dtype)
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError("W must be square (N,N)")
        self.N = W.shape[0]

        # store real coupling
        self.W = W

        # mu handling
        if isinstance(mu, (float, int)):
            self.mu = torch.full((self.N,), float(
                mu), device=device, dtype=dtype)
        else:
            mu = mu.to(device=device, dtype=dtype)
            if mu.shape != (self.N,):
                raise ValueError(f"mu must be scalar or shape ({self.N},)")
            self.mu = mu

        self.kappa = float(kappa)

        # omega handling
        if omega is None:
            g = torch.Generator(device="cpu")
            g.manual_seed(int(seed))
            lo, hi = omega_range
            omega = (lo + (hi - lo) * torch.rand((self.N,),
                     generator=g, dtype=dtype)).to(device=device)
        else:
            omega = omega.to(device=device, dtype=dtype)
            if omega.shape != (self.N,):
                raise ValueError(f"omega must have shape ({self.N},)")

        self.omega = omega

        # complex dtype that matches dtype
        self.cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128

        # deterministic complex RNG for initial conditions
        self._ic_gen = torch.Generator(device="cpu")
        self._ic_gen.manual_seed(int(seed))

    def initial_state(self, scale: float = 0.5) -> torch.Tensor:
        """
        Returns complex initial state z (N,) as torch.complex.

        :param scale: scaling factor for initial state magnitude, defaults to 0.5
        :type scale: float, optional
        :return: complex initial state z
        :rtype: torch.Tensor
        """
        z = _torch_rand_complex(
            self._ic_gen, (self.N,), device=self.device, dtype_real=self.dtype) * float(scale)
        return z.to(dtype=self.cdtype)

    def rhs(self, z: torch.Tensor) -> torch.Tensor:
        """
        Complex RHS: dz/dt

        :param z: complex state (N,)
        :type z: torch.Tensor
        :return: complex derivative dz/dt
        :rtype: torch.Tensor
        """
        if z.dtype != self.cdtype:
            z = z.to(self.cdtype)

        r2 = torch.abs(z) ** 2
        mu = self.mu.to(self.cdtype)
        intrinsic = (mu - r2) * z + 1j * self.omega.to(self.cdtype) * z

        coupling = self.kappa * (self.W.to(self.cdtype) @ z)
        return intrinsic + coupling

    def observe(self, z: torch.Tensor, squash_to_01: bool = True) -> torch.Tensor:
        """
        Observation: x = Re(z), optionally sigmoid -> [0,1].
        Returns real tensor (N,)

        :param z: complex state (N,)
        :type z: torch.Tensor
        :param squash_to_01: whether to squash output to [0,1] using sigmoid, defaults to True
        :type squash_to_01: bool, optional
        :return: observed real state (N,)
        :rtype: torch.Tensor
        """
        x = torch.real(z).to(self.dtype)
        if squash_to_01:
            x = torch.sigmoid(x)
        return x


class ReducedAbroudiODE:
    """
    Reduced cell-cycle ODE model based on Abroudi et al. (2017).\\
    12-state nonlinear regulatory system with explicit production/degradation gates.\\
    Includes one hidden delayed feedback state apc_act to generate limit-cycle oscillations.\\
    State is clamped to [0, 1] in the integrator.

    :param device: torch device to run computations on
    :type device: torch.device
    :param dtype: torch data type, defaults to torch.float32
    :type dtype: torch.dtype, optional
    """
    STATE_NAMES = [
        "Myc", "Cdh1", "p27", "Rb",
        "CycD", "E2F", "SCF", "CycE",
        "CycA", "NFY", "CycB", "Cdc20",
        "APC_act",
    ]

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
        self.device = device
        self.dtype = dtype

        # hill params (global)
        self.K = 0.35
        self.n = 4
        self.K_apc = 0.40
        self.n_apc = 8

        # delayed apc activation params
        self.k_apc_on = 3.0
        self.k_apc_off = 0.6
        self.tau_apc = 2.5

        # production rates
        self.k = {
            "Myc": 1.2,
            "CycD": 1.4,
            "E2F": 1.3,
            "CycE": 1.6,
            "CycA": 1.4,
            "NFY": 1.1,
            "CycB": 1.3,
            "Cdc20": 1.6,
            "Cdh1": 1.1,
            "SCF": 1.0,
            "p27": 1.0,
            "Rb": 0.9,
        }

        # basal productions
        self.basal = {
            "Myc": 0.05,
            "CycD": 0.02,
            "E2F": 0.02,
            "CycE": 0.01,
            "CycA": 0.01,
            "NFY": 0.01,
            "CycB": 0.01,
            "Cdc20": 0.01,
            "Cdh1": 0.02,
            "SCF": 0.02,
            "p27": 0.10,
            "Rb": 0.15,
        }

        # degradation strengths
        self.d = {
            "Myc": 0.35,
            "CycD": 0.55,
            "E2F": 0.35,
            "CycE": 0.70,
            "CycA": 0.55,
            "NFY": 0.35,
            "CycB": 0.65,
            "Cdc20": 0.75,
            "Cdh1": 0.35,
            "SCF": 0.40,
            "p27": 0.25,
            "Rb": 0.20,
        }

    def _hpos(self, x, K=None, n=None):
        """
        Hill positive function.

        :param x: Input variable
        :type x: torch.Tensor
        :param K: Hill constant, defaults to None
        :type K: float, optional
        :param n: Hill coefficient, defaults to None
        :type n: float, optional
        :return: Hill positive function output
        :rtype: torch.Tensor
        """
        K = self.K if K is None else K
        n = self.n if n is None else n
        x = torch.clamp(x, 0.0, 1.0)
        return (x**n) / (K**n + x**n + 1e-12)

    def _hneg(self, x, K=None, n=None):
        """
        Hill negative function.

        :param x: Input variable
        :type x: torch.Tensor
        :param K: Hill constant, defaults to None
        :type K: float, optional
        :param n: Hill coefficient, defaults to None
        :type n: float, optional
        :return: Hill negative function output
        :rtype: torch.Tensor
        """
        K = self.K if K is None else K
        n = self.n if n is None else n
        x = torch.clamp(x, 0.0, 1.0)
        return (K**n) / (K**n + x**n + 1e-12)

    def initial_state(self):
        """
        Initial state vector.

        :return: Initial state vector
        :rtype: torch.Tensor
        """
        # g1-like: high p27, high cdh1, low cyclins, low apc
        x = torch.tensor(
            [0.15, 0.80, 0.75, 0.80,
             0.10, 0.10, 0.20, 0.10,
             0.10, 0.10, 0.10, 0.10,
             0.05],
            device=self.device,
            dtype=self.dtype,
        )
        return x

    def rhs(self, x):
        """
        Right-hand side of the ODE system.

        :param x: Current state vector
        :type x: torch.Tensor
        :return: Time derivatives of the state vector
        :rtype: torch.Tensor
        """
        Myc, Cdh1, p27, Rb, CycD, E2F, SCF, CycE, CycA, NFY, CycB, Cdc20, APC_act = x

        if not hasattr(self, "_dbg_printed"):
            self._dbg_printed = 0

        if self._dbg_printed < 3:
            apc_cdh1 = self._hpos(Cdh1, K=self.K_apc, n=self.n_apc)
            apc_cdc20 = self._hpos(Cdc20, K=self.K_apc, n=self.n_apc)

            g_cycb_old = self._hpos(
                NFY) * self._hneg(apc_cdh1) * self._hneg(apc_cdc20)
            g_cycb_new = self._hpos(NFY)

            print("dbg gates:",
                  "Cdh1", float(Cdh1),
                  "NFY", float(NFY),
                  "apc_cdh1", float(apc_cdh1),
                  "g_cycb_old", float(g_cycb_old),
                  "g_cycb_new", float(g_cycb_new))
            self._dbg_printed += 1

        # apc proxies for gating (kept, but now the true delayed switch is apc_act)
        apc_cdh1 = self._hpos(Cdh1, K=self.K_apc, n=self.n_apc)
        apc_cdc20 = self._hpos(Cdc20, K=self.K_apc, n=self.n_apc)

        # regulatory production gates
        g_myc = 0.35 + 0.65 * (self._hpos(E2F) * self._hneg(Rb))
        g_cycd = self._hpos(Myc) * self._hneg(p27)
        g_e2f = self._hpos(Myc) * self._hneg(Rb) * self._hpos(CycD)
        g_cyce = self._hpos(E2F) * self._hneg(p27) * self._hneg(SCF)
        g_cyca = self._hpos(E2F) * self._hneg(p27) * \
            self._hneg(apc_cdc20) * self._hneg(apc_cdh1)
        g_nfy = 1.0 - (1.0 - self._hpos(CycA)) * (1.0 - self._hpos(CycE))
        g_cycb = self._hpos(NFY)
        g_cdc20 = self._hpos(CycB) * self._hneg(apc_cdh1)
        g_cdh1 = self._hneg(CycE, K=self.K_apc, n=self.n_apc) * self._hneg(
            CycA, K=self.K_apc, n=self.n_apc) * self._hneg(CycB, K=self.K_apc, n=self.n_apc) * self._hneg(APC_act, K=0.3, n=6)
        g_scf = self._hpos(CycE) * self._hneg(Cdh1)
        g_p27 = self._hneg(Myc) * self._hneg(CycD) * self._hneg(CycE)
        g_rb = self._hneg(CycD, K=0.25, n=6) * \
            self._hneg(CycE) * self._hneg(CycA)

        # define dynamics

        def dyn(xi, prod_gate, k, basal, deg_base, deg_extra=0.0):
            prod = basal + k * prod_gate
            deg = deg_base + deg_extra
            return prod * (1.0 - xi) - deg * xi

        # compute derivatives
        # delayed apc activation (slow rise, slow fall)
        dAPC_act = (
            self.k_apc_on * self._hpos(CycB, K=0.40, n=6) * (1.0 - APC_act)
            - self.k_apc_off * APC_act
        ) / self.tau_apc

        dMyc = dyn(Myc, g_myc, self.k["Myc"], self.basal["Myc"], self.d["Myc"])
        dCycD = dyn(
            CycD, g_cycd, self.k["CycD"], self.basal["CycD"], self.d["CycD"],
            deg_extra=0.35 * self._hpos(SCF) + 0.35 * apc_cdc20
        )
        dE2F = dyn(E2F, g_e2f, self.k["E2F"], self.basal["E2F"], self.d["E2F"])
        dCycE = dyn(
            CycE, g_cyce, self.k["CycE"], self.basal["CycE"], self.d["CycE"],
            deg_extra=0.65 * self._hpos(SCF)
        )
        dCycA = dyn(
            CycA, g_cyca, self.k["CycA"], self.basal["CycA"], self.d["CycA"],
            deg_extra=0.60 * apc_cdc20 + 0.35 * apc_cdh1
        )
        dNFY = dyn(NFY, g_nfy, self.k["NFY"], self.basal["NFY"], self.d["NFY"])
        dCycB = dyn(
            CycB, g_cycb, self.k["CycB"], self.basal["CycB"], self.d["CycB"],
            deg_extra=1.40 * APC_act
        )
        dCdc20 = dyn(
            Cdc20, g_cdc20, self.k["Cdc20"], self.basal["Cdc20"], self.d["Cdc20"],
            deg_extra=0.65 * apc_cdh1
        )
        dCdh1 = dyn(
            Cdh1, g_cdh1, self.k["Cdh1"], self.basal["Cdh1"], self.d["Cdh1"],
            deg_extra=0.25 * self._hpos(CycE, K=self.K_apc, n=self.n_apc)
        )
        dSCF = dyn(SCF, g_scf, self.k["SCF"], self.basal["SCF"], self.d["SCF"])
        dp27 = dyn(
            p27, g_p27, self.k["p27"], self.basal["p27"], self.d["p27"],
            deg_extra=0.55 * self._hpos(SCF)
        )
        dRb = dyn(Rb, g_rb, self.k["Rb"], self.basal["Rb"], self.d["Rb"])

        dx = torch.stack([
            dMyc, dCdh1, dp27, dRb,
            dCycD, dE2F, dSCF, dCycE,
            dCycA, dNFY, dCycB, dCdc20,
            dAPC_act,
        ])
        return dx


def make_cell_cycle_mu(
    proteins: list[str],
    cyclin_mu: float = 1.8,
    regulator_mu: float = 1.2,
    input_mu: float = 0.8,
    device=None,
    dtype=torch.float32,
):
    """Assign per-node Hopf amplitudes :math:`\mu_i` using a three-tier hierarchy.

    cyclins are modeled as strong oscillators, regulators as moderate oscillators,
    and inputs/modulators as weak oscillators. this hierarchy is phenomenological
    and intended to reflect relative dynamical influence, not biochemical kinetics.

    :param proteins: ordered list of protein names
    :type proteins: list[str]
    :param cyclin_mu: Hopf amplitude for cyclin nodes, defaults to 1.8
    :type cyclin_mu: float, optional
    :param regulator_mu: Hopf amplitude for regulator nodes, defaults to 1.2
    :type regulator_mu: float, optional
    :param input_mu: Hopf amplitude for input/modulator nodes, defaults to 0.8
    :type input_mu: float, optional
    :param device: optional device to place the returned tensor on, defaults to None
    :type device: torch.device | None, optional
    :param dtype: data type for the returned tensor, defaults to torch.float32
    :type dtype: torch.dtype, optional
    :return: per-node Hopf amplitudes of shape (len(proteins),)
    :rtype: torch.Tensor
    """

    cyclins = {"CycD", "CycE", "CycA", "CycB"}
    regulators = {"E2F", "Rb", "Cdh1", "Cdc20", "NFY", "SCF"}
    inputs = {"Myc", "p27"}

    mu = torch.empty(len(proteins), dtype=dtype)

    for i, p in enumerate(proteins):
        if p in cyclins:
            mu[i] = cyclin_mu
        elif p in regulators:
            mu[i] = regulator_mu
        elif p in inputs:
            mu[i] = input_mu
        else:
            # fallback (should not happen, but keeps it safe)
            mu[i] = regulator_mu

    if device is not None:
        mu = mu.to(device)

    return mu


def build_gt_W(
    S: torch.Tensor,
    mu: float = -1.2,
    sigma: float = 0.6,
    target_radius: float = 0.9,
    gain: float = 1.0
) -> torch.Tensor:
    """
    Construct the effective interaction weight matrix W_eff.

    :param S: signed adjacency matrix in {-1, 0, +1}^{d x d}
    :type S: torch.Tensor
    :param mu: mean of the underlying normal distribution for log-normal sampling, defaults to -1.2
    :type mu: float, optional
    :param sigma: standard deviation of the underlying normal distribution for log-normal sampling, defaults to 0.6
    :type sigma: float, optional
    :param target_radius: desired spectral radius of W_eff, defaults to 0.9
    :type target_radius: float, optional
    :param gain: gain factor to scale W_eff, defaults to 1.0
    :type gain: float, optional
    :return: effective interaction weight matrix W_eff
    :rtype: torch.Tensor
    """
    d = S.shape[0]

    # sample positive magnitudes
    dist = LogNormal(mu, sigma)
    W = dist.sample((d, d))

    # apply signs from S
    W_eff = S * W

    # compute current spectral radius
    eigenvalues = torch.linalg.eigvals(gain * W_eff)
    rho = eigenvalues.abs().max().real

    if rho < 1e-8:
        raise ValueError("Spectral radius too small, mask S may be empty.")

    # rescale to target_radius
    W_eff = W_eff * (target_radius / rho)
    return W_eff
