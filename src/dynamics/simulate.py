from typing import Literal

import math
import torch
import torch.nn.functional as F

from src.dynamics.ground_truth import (
    StuartLandauNetwork,
    build_hebbian_W_from_phase_patterns_torch,
    sample_random_phase_patterns,
    ReducedAbroudiODE
)


def simulate_wilson_cowan_with_adaptation(
    W_eff,
    n_seqs,
    x0,
    T_obs,
    dt,
    steps_per_obs,
    gain,
    tau=1.0,
    tau_a=20.0,
    beta=2.0,
    b=None,
    noise_std=0.0,
    add_skew=0.0,
    return_u=False,
):
    """
    Simulate the Wilson-Cowan model with adaptation. Same dynamics as simulate_custom_ct_ode_dynamics, but with slow adaptation variable.

    :param W_eff: _effective interaction weight matrix W_eff in R^{d x d}_
    :type W_eff: torch.Tensor
    :param n_seqs: number of sequences to simulate
    :type n_seqs: int
    :param x0: initial state vector
    :type x0: torch.Tensor
    :param T_obs: number of observation time steps
    :type T_obs: int
    :param dt: integration time step
    :type dt: float
    :param steps_per_obs: number of integration steps per observation
    :type steps_per_obs: int
    :param gain: gain factor for the activation function
    :type gain: float
    :param tau: time constant for the Wilson-Cowan dynamics, defaults to 1.0
    :type tau: float, optional
    :param tau_a: time constant for the adaptation variable, defaults to 20.0
    :type tau_a: float, optional
    :param beta: adaptation strength, defaults to 2.0
    :type beta: float, optional
    :param b: bias vector, defaults to None
    :type b: torch.Tensor, optional
    :param noise_std: standard deviation of Gaussian noise added to the dynamics, defaults to 0.0
    :type noise_std: float, optional
    :param add_skew: magnitude of skew-symmetric perturbation to W_eff, defaults to 0.0
    :type add_skew: float, optional
    :param return_u: whether to return pre-sigmoid activations, defaults to False
    :type return_u: bool, optional
    :return: simulated sequences and optionally pre-sigmoid activations
    :rtype: torch.Tensor or tuple[torch.Tensor, torch.Tensor]
    """
    d = W_eff.shape[0]
    u_traj = [] if return_u else None

    # tau vector
    if isinstance(tau, (int, float)):
        tau_vec = torch.full((d,), float(tau), dtype=W_eff.dtype)
    else:
        tau_vec = tau.to(dtype=W_eff.dtype)

    # skew option
    mask = (W_eff != 0).to(dtype=W_eff.dtype)
    if add_skew != 0:
        A = torch.randn(d, d, dtype=W_eff.dtype)
        skew = A - A.T
        W_use = W_eff + add_skew * skew * mask
    else:
        W_use = W_eff

    # bias:
    # don't force a stable fixed point at mu. use either:
    # - provided b
    # - or small random bias around 0
    if b is None:
        b = 0.2 * torch.randn(d, dtype=W_eff.dtype)

    def f(state):
        x = state[:d]
        a = state[d:]

        z = gain * (W_use @ x + b - beta * a)
        y = torch.sigmoid(z)

        dx = (-x + y) / tau_vec
        da = (x - a) / float(tau_a)
        return torch.cat([dx, da], dim=0)

    sequences = []
    for _ in range(n_seqs):
        if x0 is None:
            x = torch.distributions.Beta(2.0, 2.0).sample(
                (d,)).to(dtype=W_eff.dtype)
        else:
            x = x0.clone()

        a = x.clone()  # start adapted to x
        state = torch.cat([x, a], dim=0)

        traj = [x]

        for _ in range(T_obs):
            for _ in range(steps_per_obs):
                state = _rk4_step(state, f, dt)

                if noise_std > 0:
                    state[:d] = state[:d] + noise_std * \
                        math.sqrt(dt) * torch.randn(d, dtype=W_eff.dtype)

                state[:d] = torch.clamp(state[:d], 0.0, 1.0)

            x = state[:d]
            if return_u:
                u = gain * (W_use @ x + b - beta * state[d:])
                u_traj.append(u.detach().cpu())

            traj.append(x)

        sequences.append(torch.stack(traj))

    if return_u:
        return torch.stack(sequences), torch.stack(u_traj)
    return torch.stack(sequences)


def _rk4_step(x, f, dt):
    """Perform a single Runge–Kutta 4 integration step.

    :param x: current state
    :type x: torch.Tensor
    :param f: right-hand-side function mapping state to derivative
    :type f: callable
    :param dt: integration time step
    :type dt: float
    :return: updated state after one RK4 step
    :rtype: torch.Tensor
    """
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_hopf_oscillator_network(
    W: torch.Tensor | None = None,
    *,
    # core dataset shape
    n_seqs: int = 10,
    T_obs: int = 350,
    total_time: float = 200.0,
    dt: float = 0.01,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,

    # Hopf network knobs
    mu: float = 1.0,
    kappa: float = 0.6,
    omega: torch.Tensor | None = None,
    omega_range: tuple[float, float] = (1.5, 2.5),
    tau_p: float | None = None,   # protein lifetime (None = disabled)

    # multistability / "associative memory" knobs
    use_hebbian_W: bool = True,
    n_memories: int = 4,
    hebbian_scale: float = 1.0,
    phase_patterns: torch.Tensor | None = None,
    base_W: torch.Tensor | None = None,  # incorporate signed wiring

    # noise + observation
    noise_std: float = 0.0,
    squash_to_01: bool = True,

    # slow adaptive regulation (true multiscale dynamics)
    use_slow_mu: bool = False,
    slow_mu_rate: float = 0.01,   # ε
    slow_mu_target: float | None = None,  # μ̄

    # slow gating variable (single scalar per trajectory)
    use_gate: bool = False,
    gate_tau: float = 25.0,          # slow timescale (>> dt)
    gate_slope: float = 12.0,        # sigmoid sharpness
    gate_theta: float = 0.52,        # threshold in observed space
    gate_beta: float = 0.8,          # how strongly g modulates kappa
    gate_idx: list[int] | None = None,  # which proteins drive the gate

    # reproducibility
    seed: int = 0,
):
    """
    Simulate a coupled Hopf (Stuart–Landau) oscillator network and return observed trajectories

    :param W: optional coupling matrix W (N,N); if None and use_hebbian_W=True, builds Hebbian W from phase patterns, defaults to None
    :type W: torch.Tensor, optional
    :param n_seqs: number of independent sequences to simulate, defaults to 10
    :type n_seqs: int, optional
    :param T_obs: number of observation time points, defaults to 350
    :type T_obs: int, optional
    :param total_time: total simulation time, defaults to 200.0
    :type total_time: float, optional
    :param dt: integration time step, defaults to 0.01
    :type dt: float, optional
    :param device: torch device to run computations on, defaults to None
    :type device: torch.device, optional
    :param dtype: torch data type, defaults to torch.float32
    :type dtype: torch.dtype, optional
    :param mu: Hopf bifurcation parameter, defaults to 1.0
    :type mu: float, optional
    :param kappa: coupling strength, defaults to 0.6
    :type kappa: float, optional
    :param omega: optional intrinsic frequencies (N,); if None, sampled uniformly from omega_range, defaults to None
    :type omega: torch.Tensor, optional
    :param omega_range: range for sampling intrinsic frequencies if omega is None, defaults to (1.5, 2.5)
    :type omega_range: tuple[float, float], optional
    :param tau_p: optional protein lifetime (damping); if None, no damping is applied, defaults to None
    :type tau_p: float | None, optional
    :param use_hebbian_W: whether to build W using Hebbian phase patterns, defaults to True
    :type use_hebbian_W: bool, optional
    :param n_memories: number of phase patterns / attractors to store if use_hebbian_W=True, defaults to 4
    :type n_memories: int, optional
    :param hebbian_scale: scaling factor for Hebbian W, defaults to 1.0
    :type hebbian_scale: float, optional
    :param phase_patterns: optional (K,N) tensor of phases in radians to build Hebbian W, defaults to None
    :type phase_patterns: torch.Tensor, optional
    :param base_W: optional coupling matrix (N,N) to blend with Hebbian W, defaults to None
    :type base_W: torch.Tensor, optional
    :param noise_std: standard deviation for additive observation noise, defaults to 0.0
    :type noise_std: float, optional
    :param squash_to_01: whether to squash observed values to [0,1] via sigmoid, defaults to True
    :type squash_to_01: bool, optional
    :param seed: random seed for reproducibility, defaults to 0
    :type seed: int, optional
    :return: simulated protein expression trajectories of shape (n_seqs, T_obs, 12)
    :rtype: torch.Tensor
    """
    if device is None:
        device = torch.device("cpu")

    N = 12

    # build coupling matrix W if not provided
    if W is None:
        if use_hebbian_W:
            if phase_patterns is None:
                phase_patterns = sample_random_phase_patterns(
                    K=n_memories,
                    N=N,
                    device=device,
                    dtype=dtype,
                    seed=seed,
                )
            else:
                phase_patterns = phase_patterns.to(device=device, dtype=dtype)

            W = build_hebbian_W_from_phase_patterns_torch(
                phase_patterns,
                scale=hebbian_scale,
            )

            if base_W is not None:
                # blend hebbian structure with your signed/weighted wiring
                # keep it stable by normalizing base_W to max|.| = 1 first
                base = base_W.to(device=device, dtype=dtype)
                base = base.clone()
                base.fill_diagonal_(0.0)
                m = torch.max(torch.abs(base))
                if m > 1e-8:
                    base = base / m
                # 70% hebbian, 30% base wiring by default
                W = 0.7 * W + 0.3 * base
                W.fill_diagonal_(0.0)
        else:
            # if no W and no hebbian, default to small random coupling
            g = torch.Generator(device="cpu")
            g.manual_seed(int(seed))
            W = 0.2 * torch.randn((N, N), generator=g,
                                  dtype=dtype).to(device=device)
            W.fill_diagonal_(0.0)
    else:
        W = W.to(device=device, dtype=dtype)
        if W.shape != (N, N):
            raise ValueError(f"W must have shape ({N},{N})")

    # make Hopf network
    net = StuartLandauNetwork(
        W=W,
        mu=mu,
        kappa=kappa,
        omega=omega,
        omega_range=omega_range,
        device=device,
        dtype=dtype,
        seed=seed,
    )

    def _gate_drive(x_obs: torch.Tensor) -> torch.Tensor:
        # x_obs is (N,)
        if gate_idx is None:
            # default: cyclins in ordering from make_cell_cycle_mask
            idx = torch.tensor([4, 7, 8, 10], device=x_obs.device)
        else:
            idx = torch.tensor(gate_idx, device=x_obs.device)

        m = x_obs.index_select(0, idx).mean()  # scalar
        # target in (0,1): high when cyclins are high
        return torch.sigmoid(gate_slope * (m - gate_theta))  # scalar

    # protein-level observation state (optional low-pass)
    use_protein_filter = tau_p is not None
    if use_protein_filter:
        tau_p = float(tau_p)

    # RK4 and resample to T_obs
    obs_dt = total_time / (T_obs - 1)
    steps_per_obs = max(1, int(round(obs_dt / dt)))
    dt_use = obs_dt / steps_per_obs

    xs = []
    for s in range(n_seqs):

        # slow adaptive mu variable (one per node)
        if use_slow_mu:
            a = torch.zeros((N,), device=device, dtype=dtype)
            mu_target = mu if slow_mu_target is None else float(slow_mu_target)
            eps_mu = float(slow_mu_rate)

        z = net.initial_state(scale=0.5)
        x = net.observe(z, squash_to_01=squash_to_01)

        if use_protein_filter:
            p = x.clone()

        # slow gate state
        if use_gate:
            g = torch.tensor(0.0, device=device, dtype=dtype)  # start "off"
            gate_tau_use = float(gate_tau)
            gate_beta_use = float(gate_beta)
            base_kappa = float(kappa)

        traj = []

        for t in range(T_obs):
            x = net.observe(z, squash_to_01=squash_to_01)

            if use_protein_filter:
                # dp/dt = (-p + x) / tau_p
                p = p + dt_use * (-p + x) / tau_p
                traj.append(p.clone())
            else:
                traj.append(x.clone())

            for _ in range(steps_per_obs):
                # observe current x for gate/slow_mu updates
                if use_gate or use_slow_mu:
                    x_inner = net.observe(z, squash_to_01=squash_to_01)

                # slow gate update: dg/dt = (drive(x) - g) / gate_tau
                if use_gate:
                    drive = _gate_drive(x_inner)  # scalar in (0,1)
                    g = g + dt_use * (drive - g) / gate_tau_use
                    g = torch.clamp(g, 0.0, 1.0)

                    # modulate kappa smoothly around base_kappa
                    # kappa_eff in [base*(1-beta), base*(1+beta)] if beta<=1
                    kappa_eff = base_kappa * \
                        (1.0 + gate_beta_use * (2.0 * g - 1.0))  # tensor scalar
                    net.kappa = float(torch.clamp(kappa_eff, min=0.0).item())

                # slow_mu
                if use_slow_mu:
                    r2 = (z.real ** 2 + z.imag ** 2)
                    a = a + dt_use * eps_mu * (mu_target - r2)
                    a = torch.clamp(a, -0.5 * mu, 0.5 * mu)
                    net.mu = mu + a

                z = _rk4_step(z, net.rhs, dt_use)

                if noise_std > 0:
                    eps = noise_std * math.sqrt(dt_use) * \
                        torch.randn((N,), device=device, dtype=dtype)
                    z = z + eps.to(net.cdtype)

        xs.append(torch.stack(traj, dim=0))

    return torch.stack(xs, dim=0)  # (n_seqs, T_obs, 12)


def simulate_reduced_abroudi_ode(
    n_seqs: int = 10,
    T_obs: int = 350,
    total_time: float = 4500.0,
    dt: float = 0.25,
    x0: torch.Tensor = None,
    noise_std: float = 0.003,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
):
    """
    Simulate reduced Abroudi ODE with one hidden apc_act state.

    :param n_seqs: number of independent sequences to simulate, defaults to 10
    :type n_seqs: int, optional
    :param T_obs: number of observation time points, defaults to 350
    :type T_obs: int, optional
    :type total_time: total simulation time, defaults to 4500.0
    :param total_time: float, optional
    :param dt: integration time step, defaults to 0.01
    :type dt: float, optional
    :param x0: optional initial condition of shape (d,), defaults to None
    :type x0: torch.Tensor, optional
    :param noise_std: standard deviation for noise, defaults to 0.0
    :type noise_std: float, optional
    :param device: torch device to run computations on, defaults to None
    :type device: torch.device, optional
    :param dtype: torch data type, defaults to torch.float32
    :type dtype: torch.dtype, optional
    :return: simulated protein expression trajectories of shape (n_seqs, T_obs, d)
    :rtype: torch.Tensor
    """
    if device is None:
        device = torch.device("cpu")

    ode = ReducedAbroudiODE(device=device, dtype=dtype)

    obs_dt = total_time / (T_obs - 1)
    steps_per_obs = max(1, int(round(obs_dt / dt)))
    dt_use = obs_dt / steps_per_obs

    xs = []
    for _ in range(n_seqs):
        if x0 is None:
            # random initial condition around initial_state
            x = ode.initial_state().clone()
            x = x + 0.02 * torch.randn_like(x)
            x = torch.clamp(x, 0.0, 1.0)
        else:
            x = x0.to(device=device, dtype=dtype).clone()
            if x.shape != (13,):
                raise ValueError(
                    f"x0 must have shape (13,), got {tuple(x.shape)}")

        # simulate trajectory
        traj = []
        for _ in range(T_obs):
            # record current state (only first 12 are proteins)
            traj.append(x[:12].clone())

            for _ in range(steps_per_obs):
                x = _rk4_step(x, ode.rhs, dt_use)

                # add noise
                if noise_std > 0:
                    x = x + noise_std * math.sqrt(dt_use) * torch.randn_like(x)
                x = torch.clamp(x, 0.0, 1.0)

        xs.append(torch.stack(traj, dim=0))

    return torch.stack(xs, dim=0)  # shape (n_seqs, T_obs, d)


def simulate_custom_ct_ode_dynamics(
    W_eff: torch.Tensor,
    n_seqs: int = 10,
    x0: torch.Tensor = None,
    T_obs: int = 350,
    dt: float = 0.01,
    steps_per_obs: int = 20,
    gain: float = 8.0,
    tau: torch.Tensor | float = 1.0,
    mu: float = 0.5,
    noise_std: float = 0.0,
    add_skew: float = 0.15,
    return_u: bool = False
) -> torch.Tensor:
    """
    Continuous-time simulator with sigmoid activation dynamics using RK4 integration.

    Attempt to simulate similar dynamics as ODE model in Abroudi et al. for protein interactions.\\
    We are not considering specific biochemical rate constants or explicit cyclin/CDK kinetics here,\\
    but rather a simplified model that tries to capture key qualitative features of protein expression dynamics.

    Dynamics are given by the ODE:
        x_dot = (-x + σ(gain * (W_eff @ x) + b)) / τ + ϵ(t)

        where b is chosen such that x = μ is a fixed point when noise is zero,

    :param W_eff: effective interaction weight matrix W_eff in R^{d x d}
    :type W_eff: torch.Tensor
    :param n_seqs: number of independent sequences to simulate, defaults to 10
    :type n_seqs: int, optional
    :param x0: optional initial condition of shape (d,), defaults to random sample from Β(2, 2)
    :type x0: torch.Tensor, optional
    :param T_obs: number of observation time points, defaults to 350
    :type T_obs: int, optional
    :param dt: integration time step, defaults to 0.01
    :type dt: float, optional
    :param steps_per_obs: number of integration steps per observation, defaults to 20
    :type steps_per_obs: int, optional
    :param gain: gain factor for sigmoid activation, defaults to 8.0
    :type gain: float, optional
    :param tau: time constant for protein dynamics, defaults to 1.0
    :type tau: torch.Tensor | float, optional
    :param mu: baseline expression level, defaults to 0.5
    :type mu: float, optional
    :param noise_std: standard deviation for noise, defaults to 0.0
    :type noise_std: float, optional
    :param add_skew: skewness adjustment for W_eff, defaults to 0.15
    :type add_skew: float, optional
    :param return_u: whether to return unactivated values, defaults to False
    :type return_u: bool, optional
    :return: simulated protein expression trajectories of shape (n_seqs, T_obs+1, d) with values in [0, 1]
    :rtype: torch.Tensor
    """
    d = W_eff.shape[0]
    u_traj = [] if return_u else None

    if isinstance(tau, (int, float)):
        tau_vec = torch.full((d,), float(tau), dtype=W_eff.dtype)
    else:
        tau_vec = tau.to(dtype=W_eff.dtype)
        if tau_vec.shape != (d,):
            raise ValueError(
                f"τ must be scalar or shape ({d},), got {tuple(tau_vec.shape)}")

    # adjust W_eff to add skewness
    mask = (W_eff != 0).to(dtype=W_eff.dtype)
    if add_skew != 0:
        A = torch.randn(d, d, dtype=W_eff.dtype)
        skew = A - A.T
        W_use = W_eff + add_skew * skew * mask
    else:
        W_use = W_eff

    # set ode parameters
    mu_vec = torch.full((d,), float(mu), dtype=W_eff.dtype)
    eps = 1e-6
    mu_vec = torch.clamp(mu_vec, eps, 1 - eps)
    b = torch.log(mu_vec / (1 - mu_vec)) - (W_use @ mu_vec)

    # ode function
    def f(x):
        z = gain * (W_use @ x + b)
        y = torch.sigmoid(z)
        return (-x + y) / tau_vec

    # simulate sequences
    sequences = []
    for _ in range(n_seqs):
        if x0 is None:
            x = torch.distributions.Beta(2.0, 2.0).sample((d,))
        else:
            x = x0.clone()

        traj = [x]

        for _ in range(T_obs):
            for _ in range(steps_per_obs):
                x = _rk4_step(x, f, dt)

                # add noise
                if noise_std > 0:
                    x = x + noise_std * \
                        math.sqrt(dt) * torch.randn(d, dtype=W_eff.dtype)

                x = torch.clamp(x, 0.0, 1.0)

            # log unactivated values
            if return_u:
                u = gain * (W_use @ x + b)
                u_traj.append(u.detach().cpu())

            traj.append(x)

        sequences.append(torch.stack(traj))

    if return_u:
        return torch.stack(sequences), torch.stack(u_traj)
    return torch.stack(sequences)


def simulate_dt_sigmoid_dynamics(
    W_eff: torch.Tensor,
    T: int = 400,
    n_seqs: int = 5,
    alpha: float = 0.9,
    noise_std: float = 0.02
) -> list[torch.Tensor]:
    """
    Simulate discrete-time protein dynamics using a linear update rule with Gaussian noise.

    x_{t+1} = σ(α * (W_eff @ x_t) + ϵ), ϵ ~ N(0, noise_std^2)

    :param W_eff: effective interaction weight matrix W_eff in R^{d x d}
    :type W_eff: torch.Tensor
    :param T: number of time steps per sequence, defaults to 400
    :type T: int, optional
    :param n_seqs: number of independent trajectories, defaults to 5
    :type n_seqs: int, optional
    :param alpha: update weight parameter, defaults to 0.9
    :type alpha: float, optional
    :param noise_std: standard deviation of Gaussian noise, defaults to 0.02
    :type noise_std: float, optional
    :return: list of simulated protein expression trajectories with shape (T+1, d)
    :rtype: list[torch.Tensor]
    """
    d = W_eff.shape[0]
    sequences = []
    b = torch.zeros(d)  # no bias term

    for _ in range(n_seqs):
        # initialize x ~ Β(2, 2)
        x = torch.distributions.Beta(2., 2.).sample((d,))
        trajectories = [x]

        for _ in range(T):
            noise = torch.randn(d) * noise_std
            z = F.relu(alpha * (W_eff @ x) + b + noise)
            x = torch.sigmoid(z)
            trajectories.append(x)

        sequences.append(torch.stack(trajectories))  # shape (T+1, d)

    return sequences


def get_cell_cycle_phase_states(
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """
    Return canonical cell-cycle phase states for 12 proteins.

    Protein ordering (from make_cell_cycle_mask):
        0: Myc, 1: Cdh1, 2: p27, 3: Rb, 4: CycD, 5: E2F,
        6: SCF, 7: CycE, 8: CycA, 9: NFY, 10: CycB, 11: Cdc20

    Phase definitions based on cell-cycle biology:
        - G1: high Cdh1, p27, Rb; low cyclins; Myc rising
        - S:  high E2F, CycE, CycA rising; SCF active; low Cdh1/p27
        - G2: high CycA, NFY; CycB rising; preparing for mitosis
        - M:  high CycB, Cdc20; APC active; cyclins being degraded

    :param device: torch device, defaults to None (CPU)
    :type device: torch.device, optional
    :param dtype: torch data type, defaults to torch.float32
    :type dtype: torch.dtype, optional
    :return: dict mapping phase names to 12-dimensional state vectors
    :rtype: dict[str, torch.Tensor]
    """
    if device is None:
        device = torch.device("cpu")

    # Protein indices for reference:
    # 0:Myc, 1:Cdh1, 2:p27, 3:Rb, 4:CycD, 5:E2F, 6:SCF, 7:CycE, 8:CycA, 9:NFY, 10:CycB, 11:Cdc20

    phases = {
        "G1": torch.tensor(
            #  Myc   Cdh1   p27    Rb    CycD   E2F    SCF   CycE   CycA   NFY   CycB  Cdc20
            [0.30,  0.85,  0.80,  0.85,  0.15,  0.10,
                0.20,  0.10,  0.05,  0.15,  0.05,  0.10],
            device=device, dtype=dtype
        ),
        "S": torch.tensor(
            #  Myc   Cdh1   p27    Rb    CycD   E2F    SCF   CycE   CycA   NFY   CycB  Cdc20
            [0.60,  0.15,  0.15,  0.20,  0.70,  0.85,
                0.75,  0.90,  0.50,  0.40,  0.10,  0.10],
            device=device, dtype=dtype
        ),
        "G2": torch.tensor(
            #  Myc   Cdh1   p27    Rb    CycD   E2F    SCF   CycE   CycA   NFY   CycB  Cdc20
            [0.45,  0.10,  0.10,  0.15,  0.40,  0.50,
                0.50,  0.30,  0.85,  0.80,  0.60,  0.15],
            device=device, dtype=dtype
        ),
        "M": torch.tensor(
            #  Myc   Cdh1   p27    Rb    CycD   E2F    SCF   CycE   CycA   NFY   CycB  Cdc20
            [0.25,  0.20,  0.10,  0.10,  0.20,  0.25,
                0.30,  0.10,  0.40,  0.50,  0.90,  0.85],
            device=device, dtype=dtype
        ),
    }

    return phases


def simulate_phase_dynamics_with_basins(
    n_seqs: int = 24,
    T_obs: int = 400,
    n_cycles: int = 4,
    steps_per_phase: int = 80,
    basin_strength: float = 0.5,
    noise_std: float = 0.02,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> torch.Tensor:
    """
    Simulate cell-cycle dynamics with smooth attractor basins.

    This version creates continuous flow toward phase attractors with
    configurable basin strength. Higher basin_strength creates sharper
    attractor basins that should enable better learning.

    :param n_seqs: number of independent sequences, defaults to 24
    :type n_seqs: int, optional
    :param T_obs: total observation time points, defaults to 400
    :type T_obs: int, optional
    :param n_cycles: number of complete cell cycles, defaults to 4
    :type n_cycles: int, optional
    :param steps_per_phase: steps to dwell in each phase, defaults to 80
    :type steps_per_phase: int, optional
    :param basin_strength: strength of attractor basin (0=random walk, 1=deterministic), defaults to 0.5
    :type basin_strength: float, optional
    :param noise_std: standard deviation of Gaussian noise, defaults to 0.02
    :type noise_std: float, optional
    :param device: torch device, defaults to None (CPU)
    :type device: torch.device, optional
    :param dtype: torch data type, defaults to torch.float32
    :type dtype: torch.dtype, optional
    :param seed: random seed, defaults to 42
    :type seed: int, optional
    :return: simulated trajectories (n_seqs, T_obs, 12)
    :rtype: torch.Tensor
    """
    if device is None:
        device = torch.device("cpu")

    torch.manual_seed(seed)

    phases = get_cell_cycle_phase_states(device=device, dtype=dtype)
    phase_order = ["G1", "S", "G2", "M"]
    n_phases = len(phase_order)
    d = 12

    sequences = []

    for seq_idx in range(n_seqs):
        traj = []

        # start in random phase with noise
        start_phase_idx = seq_idx % n_phases
        x = phases[phase_order[start_phase_idx]].clone()
        x = x + 0.1 * torch.randn(d, device=device, dtype=dtype)
        x = x.clamp(0.0, 1.0)

        # track which phase we're targeting
        current_phase_idx = start_phase_idx
        steps_in_current = 0

        for t in range(T_obs):
            target = phases[phase_order[current_phase_idx]]

            # flow toward target with basin strength
            gradient = target - x
            x = x + basin_strength * gradient + noise_std * \
                torch.randn(d, device=device, dtype=dtype)
            x = x.clamp(0.0, 1.0)

            traj.append(x.clone())
            steps_in_current += 1

            # transition to next phase after dwelling
            if steps_in_current >= steps_per_phase:
                current_phase_idx = (current_phase_idx + 1) % n_phases
                steps_in_current = 0

        sequences.append(torch.stack(traj, dim=0))

    return torch.stack(sequences, dim=0)


def simulate_phase_dynamics(
    n_seqs: int = 24,
    T_obs: int = 400,
    dwell_time: int = 80,
    jump_prob: float = 0.02,
    relaxation_rate: float = 0.3,
    noise_std: float = 0.015,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> torch.Tensor:
    """
    Simulate cell-cycle dynamics with stable phase attractors and stochastic jumps.

    Each phase (G1, S, G2, M) is a stable attractor. The system dwells in each
    phase with fast relaxation dynamics, occasionally jumping to the next phase
    with probability `jump_prob` after a minimum dwell time.

    This produces multi-attractor data where:
        - Each phase is a TRUE stable attractor (system returns if perturbed)
        - Transitions are stochastic jumps, not smooth relaxations
        - One-step dynamics strongly pull toward current attractor

    :param n_seqs: number of independent sequences to simulate, defaults to 24
    :type n_seqs: int, optional
    :param T_obs: total number of observation time points, defaults to 400
    :type T_obs: int, optional
    :param dwell_time: minimum steps before allowing phase transition, defaults to 80
    :type dwell_time: int, optional
    :param jump_prob: probability of jumping to next phase after dwell_time, defaults to 0.02
    :type jump_prob: float, optional
    :param relaxation_rate: rate of exponential relaxation toward phase attractor, defaults to 0.3
    :type relaxation_rate: float, optional
    :param noise_std: standard deviation of per-step Gaussian noise, defaults to 0.015
    :type noise_std: float, optional
    :param device: torch device, defaults to None (CPU)
    :type device: torch.device, optional
    :param dtype: torch data type, defaults to torch.float32
    :type dtype: torch.dtype, optional
    :param seed: random seed for reproducibility, defaults to 42
    :type seed: int, optional
    :return: simulated protein expression trajectories of shape (n_seqs, T_obs, 12)
    :rtype: torch.Tensor
    """
    if device is None:
        device = torch.device("cpu")

    torch.manual_seed(seed)

    phases = get_cell_cycle_phase_states(device=device, dtype=dtype)
    phase_order = ["G1", "S", "G2", "M"]
    n_phases = len(phase_order)
    d = 12  # number of proteins

    sequences = []

    for seq_idx in range(n_seqs):
        # randomize starting phase for diversity
        current_phase_idx = seq_idx % n_phases
        target = phases[phase_order[current_phase_idx]].clone()

        # initialize near the starting phase attractor with some noise
        x = target.clone()
        x = x + 0.1 * torch.randn(d, device=device, dtype=dtype)
        x = torch.clamp(x, 0.0, 1.0)

        traj = []
        steps_in_phase = 0

        for t in range(T_obs):
            # fast relaxation toward current phase attractor
            x = x + relaxation_rate * (target - x)

            # add small noise
            noise = noise_std * torch.randn(d, device=device, dtype=dtype)
            x = x + noise
            x = torch.clamp(x, 0.0, 1.0)

            traj.append(x.clone())
            steps_in_phase += 1

            # check for phase transition (only after minimum dwell time)
            if steps_in_phase >= dwell_time:
                if torch.rand(1).item() < jump_prob:
                    # transition to next phase
                    current_phase_idx = (current_phase_idx + 1) % n_phases
                    target = phases[phase_order[current_phase_idx]].clone()
                    steps_in_phase = 0

        sequences.append(torch.stack(traj, dim=0))

    return torch.stack(sequences, dim=0)  # (n_seqs, T_obs, 12)


def make_transition_dataset(sequences: list[torch.Tensor], train_p: float = 0.7, val_p: float = 0.15):
    """
    Flatten sequences into transition pairs (x_t -> x_{t+1}) and split into train, val, test sets.

    :param sequences: list of simulated protein expression trajectories with shape (T+1, d)
    :type sequences: list[torch.Tensor]
    :param train_p: proportion of data for training set, defaults to 0.7
    :type train_p: float, optional
    :param val_p: proportion of data for validation set, defaults to 0.15
    :type val_p: float, optional
    :return: train, val, test datasets as tuples of (X, Y)
    :rtype: tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
    """
    X_list, Y_list = [], []

    for seq in sequences:
        X_list.append(seq[:-1])  # shape (T, d)
        Y_list.append(seq[1:])   # shape (T, d)

    X = torch.cat(X_list, dim=0)  # shape (n_seqs * T, d)
    Y = torch.cat(Y_list, dim=0)  # shape (n_seqs * T, d)

    N = X.shape[0]
    idx = torch.randperm(N)  # shuffle
    X, Y = X[idx], Y[idx]

    n_train = int(N * train_p)
    n_val = int(N * val_p)

    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:n_train + n_val], Y[n_train:n_train + n_val]
    X_test, Y_test = X[n_train + n_val:], Y[n_train + n_val:]

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)
