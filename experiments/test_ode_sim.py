import json
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import json
import itertools
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.dynamics.ground_truth import ReducedAbroudiODE, build_gt_W, make_cell_cycle_mu
from src.dynamics.simulate import (
    simulate_wilson_cowan_with_adaptation,
    simulate_hopf_oscillator_network,
    simulate_reduced_abroudi_ode,
    simulate_custom_ct_ode_dynamics
)
from src.dynamics.masks import make_cell_cycle_mask
from src.evaluation.dynamics_diagnostics import evaluate_dynamics, summarize_diagnostics
from src.utils.seed import set_seed


def test_wilson_cowan_adaptation_sim():
    # config
    set_seed(0)

    outdir = Path("experiments/results/wilson_cowan_adaptation_test")
    outdir.mkdir(parents=True, exist_ok=True)

    d = 12
    n_seqs = 1
    T_obs = 350
    steps_per_obs = 20
    dt = 0.01

    # key model parameters
    tau = 1.0
    tau_a = 20.0          # slow adaptation
    beta = 2.0            # adaptation strength
    gain = 10.0           # stronger than vanilla WC
    noise_std = 0.0
    add_skew = 0.05

    # wiring + bias
    S, proteins = make_cell_cycle_mask()
    W_eff = build_gt_W(S, target_radius=2.5, gain=gain)

    print("rho(gW) =", torch.linalg.eigvals(gain * W_eff).abs().max().item())

    # single trajectory plotting
    print("Simulating single trajectory with adaptation...")

    x0 = torch.rand(d)
    xs, us = simulate_wilson_cowan_with_adaptation(
        W_eff=W_eff,
        n_seqs=n_seqs,
        x0=x0,
        T_obs=T_obs,
        dt=dt,
        steps_per_obs=steps_per_obs,
        gain=gain,
        tau=tau,
        tau_a=tau_a,
        beta=beta,
        noise_std=noise_std,
        add_skew=add_skew,
        return_u=True,
    )

    X = xs[0]  # (T_obs+1, d)
    t = torch.arange(X.shape[0])

    # time series of multiple nodes
    plt.figure(figsize=(10, 5))
    for i in [4, 7, 8, 10]:  # CycD, CycE, CycA, CycB indices
        plt.plot(t, X[:, i], label=f"{proteins[i]}")
    plt.legend()
    plt.xlabel("Time step")
    plt.ylabel("Expression")
    plt.title("Wilson–Cowan + adaptation: time series")
    plt.tight_layout()
    plt.savefig(outdir / "timeseries.png")
    plt.close()

    # phase portrait
    plt.figure(figsize=(5, 5))
    plt.plot(X[:, 8], X[:, 10], alpha=0.8)  # CycA vs CycB
    plt.xlabel("CycA")
    plt.ylabel("CycB")
    plt.title("Phase portrait: CycA vs CycB")
    plt.tight_layout()
    plt.savefig(outdir / "phase_portrait.png")
    plt.close()

    # FFT
    print("Computing FFT...")

    x = X[:, 10] - X[:, 10].mean()
    fft = torch.fft.rfft(x)
    freqs = torch.fft.rfftfreq(len(x), d=1.0)

    plt.figure(figsize=(8, 4))
    plt.plot(freqs[1:], fft.abs()[1:])
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.title("FFT magnitude (CycB)")
    plt.tight_layout()
    plt.savefig(outdir / "fft_cycb.png")
    plt.close()

    # pre-sigmoid activations
    print("Analyzing pre-sigmoid activations...")

    u_flat = us.reshape(-1).numpy()
    stats = {
        "u_min": float(u_flat.min()),
        "u_median": float(torch.median(us).item()),
        "u_max": float(u_flat.max()),
    }

    with open(outdir / "u_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    plt.figure(figsize=(6, 4))
    plt.hist(u_flat, bins=100)
    plt.xlabel(r"$u = g(Wx + b - \beta a)$")
    plt.ylabel("Count")
    plt.title("Distribution of pre-sigmoid activations")
    plt.tight_layout()
    plt.savefig(outdir / "u_hist.png")
    plt.close()

    # multi-IC behavior
    print("Simulating multiple initial conditions...")

    plt.figure(figsize=(10, 5))
    for k in range(5):
        x0 = torch.rand(d)
        Xk = simulate_wilson_cowan_with_adaptation(
            W_eff=W_eff,
            n_seqs=1,
            x0=x0,
            T_obs=T_obs,
            dt=dt,
            steps_per_obs=steps_per_obs,
            gain=gain,
            tau=tau,
            tau_a=tau_a,
            beta=beta,
            noise_std=noise_std,
            add_skew=add_skew,
            return_u=False,
        )[0]

        plt.plot(Xk[:, 10], alpha=0.8)

    plt.xlabel("Time step")
    plt.ylabel("CycB")
    plt.title("Multi-IC behavior (CycB)")
    plt.tight_layout()
    plt.savefig(outdir / "multi_ic_cycb.png")
    plt.close()

    # numerical consistency dt vs dt/2
    print("Checking numerical consistency...")

    X_dt = simulate_wilson_cowan_with_adaptation(
        W_eff=W_eff,
        n_seqs=1,
        x0=x0,
        T_obs=T_obs,
        dt=dt,
        steps_per_obs=steps_per_obs,
        gain=gain,
        tau=tau,
        tau_a=tau_a,
        beta=beta,
        noise_std=noise_std,
        add_skew=add_skew,
        return_u=False,
    )[0]

    X_dt2 = simulate_wilson_cowan_with_adaptation(
        W_eff=W_eff,
        n_seqs=1,
        x0=x0,
        T_obs=T_obs,
        dt=dt / 2,
        steps_per_obs=steps_per_obs * 2,
        gain=gain,
        tau=tau,
        tau_a=tau_a,
        beta=beta,
        noise_std=noise_std,
        add_skew=add_skew,
        return_u=False,
    )[0]

    plt.figure(figsize=(10, 5))
    plt.plot(X_dt[:, 10], label="dt")
    plt.plot(X_dt2[:, 10], "--", label="dt/2")
    plt.legend()
    plt.title("Step size consistency (CycB)")
    plt.tight_layout()
    plt.savefig(outdir / "dt_comparison.png")
    plt.close()

    print("Wilson–Cowan + adaptation plots saved to:", outdir)


def test_hopf_oscillator_sim():
    set_seed(0)

    outdir = Path("experiments/results/hopf_oscillator_test")
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    dtype = torch.float32

    # dataset parameters
    n_seqs = 1
    T_obs = 350
    total_time = 300.0
    dt = 0.02

    # Hopf parameters
    kappa = 0.85
    n_memories = 4
    noise_std = 0.002
    omega_range = (1.0, 3.0)
    hebbian_scale = 0.8

    # gating parameters
    use_gate = True
    gate_tau = 50.0
    gate_slope = 10.0
    gate_theta = 0.5
    gate_beta = 0.5
    gate_idx = [4, 7, 8, 10]  # CycD, CycE, CycA, CycB

    # protein observation (biological smoothing)
    tau_p = None
    use_slow_mu = True
    slow_mu_rate = 0.1

    # signed biological wiring + mu hierarchy
    S, proteins = make_cell_cycle_mask()
    signed_W = build_gt_W(S, target_radius=1.0, gain=1.0)

    # heterogeneous mu
    mu_vec = make_cell_cycle_mu(
        proteins=proteins,
        cyclin_mu=1.8,
        regulator_mu=1.2,
        input_mu=0.8,
        device=device,
        dtype=dtype,
    )

    # fixed plotting indices (labels only, not mechanistic)
    IDX = {p: i for i, p in enumerate(proteins)}

    print("Simulating Hopf oscillator trajectory...")

    # simulate single trajectory
    X = simulate_hopf_oscillator_network(
        n_seqs=n_seqs,
        T_obs=T_obs,
        total_time=total_time,
        dt=dt,
        mu=mu_vec,
        kappa=kappa,
        omega_range=omega_range,
        n_memories=n_memories,
        noise_std=noise_std,
        tau_p=tau_p,
        hebbian_scale=hebbian_scale,
        use_slow_mu=use_slow_mu,
        slow_mu_rate=slow_mu_rate,
        use_gate=use_gate,
        gate_tau=gate_tau,
        gate_slope=gate_slope,
        gate_theta=gate_theta,
        gate_beta=gate_beta,
        gate_idx=gate_idx,
        base_W=signed_W,
        squash_to_01=True,
        seed=0,
        device=device,
        dtype=dtype,
    )[0]  # (T_obs, 12)

    t = torch.arange(X.shape[0])

    print("Generating oscillator verification plots...")

    # sustained oscillations time series
    plt.figure(figsize=(10, 5))
    for name in ["CycD", "CycE", "CycA", "CycB"]:
        plt.plot(t, X[:, IDX[name]], label=name)
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Expression")
    plt.title("Sustained oscillations (cyclins, heterogeneous μ)")
    plt.tight_layout()
    plt.savefig(outdir / "oscillatory_timeseries.png")
    plt.close()

    # drop initial transient for phase portrait and FFT
    burn = int(0.1 * len(X))
    X2 = X[burn:]

    # phase portrait (limit cycle)
    plt.figure(figsize=(5, 5))
    plt.plot(
        X2[:, IDX["CycA"]],
        X2[:, IDX["CycB"]],
        alpha=0.8
    )
    plt.xlabel("CycA")
    plt.ylabel("CycB")
    plt.title("Phase portrait: CycA vs CycB")
    plt.tight_layout()
    plt.savefig(outdir / "phase_portrait_cyca_cycb.png")
    plt.close()

    # FFT
    print("Computing frequency content...")

    x = X2[:, IDX["CycB"]] - X2[:, IDX["CycB"]].mean()
    fft = torch.fft.rfft(x)
    freqs = torch.fft.rfftfreq(len(x), d=1.0)

    plt.figure(figsize=(8, 4))
    plt.plot(freqs[1:], fft.abs()[1:])
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.title("FFT magnitude (CycB)")
    plt.tight_layout()
    plt.savefig(outdir / "fft_cycb.png")
    plt.close()

    # multi-IC basin of attraction
    print("Simulating multi-initial-condition robustness...")

    plt.figure(figsize=(10, 5))
    for k in range(5):
        Xk = simulate_hopf_oscillator_network(
            n_seqs=n_seqs,
            T_obs=T_obs,
            total_time=total_time,
            dt=dt,
            mu=mu_vec,
            kappa=kappa,
            omega_range=omega_range,
            n_memories=n_memories,
            noise_std=noise_std,
            tau_p=tau_p,
            hebbian_scale=hebbian_scale,
            use_slow_mu=use_slow_mu,
            slow_mu_rate=slow_mu_rate,
            use_gate=use_gate,
            gate_tau=gate_tau,
            gate_slope=gate_slope,
            gate_theta=gate_theta,
            gate_beta=gate_beta,
            gate_idx=gate_idx,
            base_W=signed_W,
            squash_to_01=True,
            seed=100 + k,
            device=device,
            dtype=dtype,
        )[0]

        plt.plot(Xk[:, IDX["CycB"]], alpha=0.8)

    plt.xlabel("Time")
    plt.ylabel("CycB")
    plt.title("Multi-IC behavior (CycB)")
    plt.tight_layout()
    plt.savefig(outdir / "multi_ic_cycb.png")
    plt.close()

    # μ-hierarchy plot
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(mu_vec)), mu_vec.cpu().numpy())
    plt.xticks(range(len(mu_vec)), proteins, rotation=45, ha="right")
    plt.ylabel("μᵢ (Hopf amplitude)")
    plt.title("Per-node oscillation strength (μ hierarchy)")
    plt.tight_layout()
    plt.savefig(outdir / "mu_hierarchy.png")
    plt.close()

    # build one-step dataset
    X_in = X[:-1]          # (T-1, d)
    Y = X[1:]              # (T-1, d)
    d = X.shape[1]

    # fit linear model
    X_aug = torch.cat(
        # (T-1, d+1)
        [X_in, torch.ones((X_in.shape[0], 1), dtype=dtype)], dim=1)
    Theta = torch.linalg.lstsq(X_aug, Y).solution  # (d+1, d)
    A = Theta[:d, :].T                            # (d, d)
    b = Theta[d, :].unsqueeze(0)                  # (1, d)

    def linear_step(x):
        y = x @ A.T + b
        return torch.clamp(y, 0.0, 1.0)

    # fit small sigmoid
    model = torch.nn.Sequential(
        torch.nn.Linear(d, 64),
        torch.nn.Sigmoid(),
        torch.nn.Linear(64, d),
        torch.nn.Sigmoid(),  # keep output in (0,1)
    ).to(device=device, dtype=dtype)

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = torch.nn.MSELoss()

    print("Training sigmoid predictor...")
    for it in range(2000):
        opt.zero_grad()
        pred = model(X_in)
        loss = loss_fn(pred, Y)
        loss.backward()
        opt.step()

    def sigmoid_step(x):
        return model(x)

    # open loop rollout
    T_roll = T_obs
    x0 = X[0:1]  # (1,d)

    X_lin = [x0]
    X_sig = [x0]
    for _ in range(T_roll - 1):
        X_lin.append(linear_step(X_lin[-1]))
        X_sig.append(sigmoid_step(X_sig[-1]))

    X_lin = torch.cat(X_lin, dim=0).detach()  # (T,d)
    X_sig = torch.cat(X_sig, dim=0).detach()  # (T,d)

    # rollout MSE vs time (cumulative or per-step)
    err_lin = ((X_lin - X) ** 2).mean(dim=1).detach()
    err_sig = ((X_sig - X) ** 2).mean(dim=1).detach()

    plt.figure(figsize=(10, 5))
    plt.plot(t, err_lin, label="Linear Rollout MSE(t)")
    plt.plot(t, err_sig, label="Sigmoid Rollout MSE(t)")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("MSE")
    plt.title("Open-loop rollout error vs time")
    plt.tight_layout()
    plt.savefig(outdir / "rollout_mse_vs_time.png")
    plt.close()

    # overlay on CycB
    name = "CycB"
    i = IDX[name]
    plt.figure(figsize=(10, 5))
    plt.plot(t, X[:, i], label="True")
    plt.plot(t, X_lin[:, i], "--", label="Linear Rollout")
    plt.plot(t, X_sig[:, i], "--", label="Sigmoid Rollout")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel(name)
    plt.title(f"Rollout overlay ({name})")
    plt.tight_layout()
    plt.savefig(outdir / f"overlay_{name.lower()}.png")
    plt.close()

    # phase portrait true vs linear vs sigmoid (CycA vs CycB)
    plt.figure(figsize=(6, 6))
    plt.plot(X[:, IDX["CycA"]], X[:, IDX["CycB"]], alpha=0.7, label="true")
    plt.plot(X_lin[:, IDX["CycA"]], X_lin[:, IDX["CycB"]],
             alpha=0.7, label="linear")
    plt.plot(X_sig[:, IDX["CycA"]], X_sig[:, IDX["CycB"]],
             alpha=0.7, label="sigmoid")
    plt.xlabel("CycA")
    plt.ylabel("CycB")
    plt.title("Phase portrait: CycA vs CycB (rollouts)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "phase_portrait_rollouts.png")
    plt.close()

    # summary numbers
    mse_lin = float(((X_lin - X) ** 2).mean().item())
    mse_sig = float(((X_sig - X) ** 2).mean().item())

    stats = {"mse_linear_rollout": mse_lin, "mse_sigmoid_rollout": mse_sig}
    with open(outdir / "rollout_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # evaluate dynamics
    X = simulate_hopf_oscillator_network(
        n_seqs=24,
        T_obs=T_obs,
        total_time=total_time,
        dt=dt,
        mu=mu_vec,
        kappa=kappa,
        omega_range=omega_range,
        n_memories=n_memories,
        noise_std=noise_std,
        tau_p=tau_p,
        hebbian_scale=hebbian_scale,
        use_slow_mu=use_slow_mu,
        slow_mu_rate=slow_mu_rate,
        use_gate=use_gate,
        gate_tau=gate_tau,
        gate_slope=gate_slope,
        gate_theta=gate_theta,
        gate_beta=gate_beta,
        gate_idx=gate_idx,
        base_W=signed_W,
        squash_to_01=True,
        seed=0,
        device=device,
        dtype=dtype,
    )

    print("Evaluating learned dynamics...")
    results = evaluate_dynamics(
        X,
        dt,
        n_memories=n_memories,
        seed=0
    )
    print(summarize_diagnostics(results))

    print("Hopf oscillator plots saved to:", outdir)
    print("Rollout MSE:", stats)


def test_bio_ode_model_sim():
    # config
    set_seed(0)

    outdir = Path("experiments/results/model_bio_ode_test")
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")

    # simulate baseline trajectory
    print("Simulating baseline trajectory...")

    x0 = torch.rand(13)
    X = simulate_reduced_abroudi_ode(
        n_seqs=1,
        T_obs=350,
        total_time=4500.0,
        dt=0.25,
        x0=x0,
        noise_std=0.0,
        device=device,
    )[0]  # shape (T_obs, 12)

    print("first timepoint:", X[0].tolist())
    print("last  timepoint:", X[-1].tolist())

    # show a few named values using the assumed correct order
    print("named:",
          "Cdh1", float(X[0, 1]),
          "NFY", float(X[0, 9]),
          "CycB", float(X[0, 10]),
          "Cdc20", float(X[0, 11]))

    # unpack state indices
    IDX = {
        "Myc": 0,
        "Cdh1": 1,
        "p27": 2,
        "Rb": 3,
        "CycD": 4,
        "E2F": 5,
        "SCF": 6,
        "CycE": 7,
        "CycA": 8,
        "NFY": 9,
        "CycB": 10,
        "Cdc20": 11,
    }

    t = torch.arange(X.shape[0])

    print("Generating verification plots...")

    # NFY vs CycB sanity check plot
    plt.figure(figsize=(10, 5))
    plt.plot(t, X[:, IDX["NFY"]], label="NFY")
    plt.plot(t, X[:, IDX["CycB"]], label="CycB")
    plt.legend()
    plt.title("NFY vs CycB (sanity check)")
    plt.tight_layout()
    plt.savefig(outdir / "nfy_vs_cycb.png")
    plt.close()

    # cyclin ordering plot
    plt.figure(figsize=(10, 5))
    plt.plot(t, X[:, IDX["CycD"]], label="CycD")
    plt.plot(t, X[:, IDX["CycE"]], label="CycE")
    plt.plot(t, X[:, IDX["CycA"]], label="CycA")
    plt.plot(t, X[:, IDX["CycB"]], label="CycB")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Expression")
    plt.title("Cyclin ordering: D → E → A → B")
    plt.tight_layout()
    plt.savefig(outdir / "cyclin_ordering.png")
    plt.close()

    # APC switch plot
    plt.figure(figsize=(10, 5))
    plt.plot(t, X[:, IDX["Cdh1"]], label="Cdh1")
    plt.plot(t, X[:, IDX["Cdc20"]], label="Cdc20")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Expression")
    plt.title("APC switch: Cdh1 vs Cdc20")
    plt.tight_layout()
    plt.savefig(outdir / "apc_switch.png")
    plt.close()

    # E2F-Rb antagonism plot
    plt.figure(figsize=(10, 5))
    plt.plot(t, X[:, IDX["E2F"]], label="E2F")
    plt.plot(t, X[:, IDX["Rb"]], label="Rb")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Expression")
    plt.title("E2F–Rb antagonism")
    plt.tight_layout()
    plt.savefig(outdir / "e2f_rb.png")
    plt.close()

    # multi-IC robustness plot
    print("Simulating multi-IC robustness...")

    plt.figure(figsize=(10, 5))
    for k in range(5):
        x0 = ReducedAbroudiODE(device=device).initial_state()
        x0 = x0 + 0.02 * torch.randn_like(x0)
        x0 = torch.clamp(x0, 0.0, 1.0)

        Xk = simulate_reduced_abroudi_ode(
            n_seqs=1,
            T_obs=350,
            total_time=4500.0,
            dt=0.25,
            x0=x0,
            noise_std=0.0,
            device=device,
        )[0]

        plt.plot(Xk[:, IDX["CycB"]], alpha=0.8)

    plt.xlabel("Time")
    plt.ylabel("CycB")
    plt.title("Multi-IC robustness (CycB)")
    plt.tight_layout()
    plt.savefig(outdir / "multi_ic.png")
    plt.close()

    print("Plots saved to:", outdir)


def test_custom_ode_sim():
    # config
    set_seed(0)

    outdir = Path(f"experiments/results/custom_ode_test")
    outdir.mkdir(parents=True, exist_ok=True)

    d = 12
    n_seqs = 1  # single trajectory
    T = 3000
    steps_per_obs = 20
    tau = 1.0
    mu = 0.5
    gain = 8.0
    dt = 0.01
    noise_std = 0.0
    add_skew = 0.0

    # wiring + bias
    S, proteins = make_cell_cycle_mask()
    W_eff = build_gt_W(S, target_radius=1.5, gain=gain)
    print("rho(gW) =", torch.linalg.eigvals(gain * W_eff).abs().max().item())

    # single trajectory plotting
    print("Simulating single trajectory...")
    x0 = torch.rand(d)
    xs, us = simulate_custom_ct_ode_dynamics(
        W_eff, n_seqs, x0, T, dt, steps_per_obs, gain, tau, mu, noise_std, add_skew, return_u=True
    )

    plt.figure(figsize=(10, 5))
    for i in [0, 1, 2, 3]:
        plt.plot(xs[0, :, i], label=f"Protein {i}")
    plt.legend()
    plt.title("Time Series (No Noise)")
    plt.xlabel("Time step")
    plt.ylabel("Expression")
    plt.tight_layout()
    plt.savefig(outdir / "timeseries.png")
    plt.close()

    # dt vs dt/2 comparison
    print("Comparing dt vs dt/2...")
    xs_dt = simulate_custom_ct_ode_dynamics(
        W_eff, n_seqs, x0, T, dt, steps_per_obs, gain, tau, mu, noise_std, add_skew, return_u=False
    )
    xs_dt2 = simulate_custom_ct_ode_dynamics(
        W_eff, n_seqs, x0, T * 2, dt / 2, steps_per_obs, gain, tau, mu, noise_std, add_skew, return_u=False
    )

    plt.figure(figsize=(10, 5))
    for i in [0, 1]:
        plt.plot(xs_dt[0, :, i], label=f"dt Protein {i}")
        plt.plot(xs_dt2[0, :, i], "--", label=f"dt/2 Protein {i}")
    plt.legend()
    plt.title("Step Size Consistency")
    plt.tight_layout()
    plt.savefig(outdir / "dt_comparison.png")
    plt.close()

    # nonlinearity regime
    print("Analyzing pre-sigmoid activations...")
    u_flat = us.reshape(-1).numpy()

    stats = {
        "u_min": float(u_flat.min()),
        "u_median": float(torch.median(us).item()),
        "u_max": float(u_flat.max()),
    }

    with open(outdir / "u_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    plt.figure()
    plt.hist(u_flat, bins=100)
    plt.title("Distribution of pre-sigmoid activations")
    plt.xlabel("$u = g(Wx+b)$")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outdir / "u_hist.png")
    plt.close()

    # multi-initial-conditions
    print("Simulating multiple initial conditions...")
    plt.figure(figsize=(10, 5))

    x0s = []
    for k in range(5):
        x0 = torch.rand(d)
        x0s.append(x0)
        xs = simulate_custom_ct_ode_dynamics(
            W_eff, n_seqs, x0, T, dt, steps_per_obs, gain, tau, mu, noise_std, add_skew, return_u=False
        )
        plt.plot(xs[0, :, 0], alpha=0.8)

    plt.title("Multiple Initial Conditions (Protein 0)")
    plt.xlabel("Time Step")
    plt.ylabel("Expression")
    plt.tight_layout()
    plt.savefig(outdir / "multi_ic.png")
    plt.close()

    print("Plots saved to", outdir)


def sweep_dynamics_and_score(
    base_params: dict | None = None,
    n_rollouts: int = 24,
    T: int = 400,
    dt: float = 0.05,
    transient_frac: float = 0.25,
    seed: int = 0,
) -> list[dict]:
    """
    Sweep parameter space and score dynamical regimes for AANN experiments.

    This function explores meaningful parameter combinations to identify
    2-3 calibrated regimes with rich, protein-like dynamics suitable for
    testing associative memory models.

    :param base_params: base configuration dict to merge with grid, defaults to None
    :param n_rollouts: number of initial conditions per parameter set, defaults to 24
    :param T: number of observation time steps, defaults to 400
    :param dt: integration time step, defaults to 0.05
    :param transient_frac: fraction of initial timesteps to discard, defaults to 0.25
    :param seed: random seed for reproducibility, defaults to 0
    :return: sorted list of configurations with scores
    """
    set_seed(seed)

    if base_params is None:
        base_params = {}

    # construct default parameter grid
    param_grid = {
        "kappa": [1.0, 1.4],
        "hebbian_scale": [0.8, 1.2, 1.6],
        "gate_tau_mult": [10.0],  # multiplier of T*dt
        "gate_beta": [0.5],
        "noise_std": [0.0],
    }

    # generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    # setup base configuration
    device = torch.device("cpu")
    dtype = torch.float32

    S, proteins = make_cell_cycle_mask()
    signed_W = build_gt_W(S, target_radius=1.0, gain=1.0)

    mu_vec = make_cell_cycle_mu(
        proteins=proteins,
        cyclin_mu=1.8,
        regulator_mu=1.2,
        input_mu=0.8,
        device=device,
        dtype=dtype,
    )

    total_time = T * dt
    T_burn = int(transient_frac * T)
    T_eff = T - T_burn

    results = []

    for combo in combinations:
        params_dict = dict(zip(keys, combo))

        # compute derived parameters
        gate_tau = params_dict["gate_tau_mult"] * total_time
        gate_beta = params_dict["gate_beta"]
        kappa = params_dict["kappa"]
        hebbian_scale = params_dict["hebbian_scale"]
        noise_std = params_dict["noise_std"]

        # merge with base params
        sim_params = {
            "n_seqs": n_rollouts,
            "T_obs": T,
            "total_time": total_time,
            "dt": dt,
            "mu": mu_vec,
            "kappa": kappa,
            "omega_range": (1.0, 3.0),
            "n_memories": 4,
            "hebbian_scale": hebbian_scale,
            "noise_std": noise_std,
            "tau_p": None,
            "use_slow_mu": False,
            "use_gate": True,
            "gate_tau": gate_tau,
            "gate_slope": 10.0,
            "gate_theta": 0.5,
            "gate_beta": gate_beta,
            "gate_idx": [4, 7, 8, 10],
            "base_W": signed_W,
            "squash_to_01": True,
            "seed": seed,
            "device": device,
            "dtype": dtype,
        }
        sim_params.update(base_params)

        # simulate trajectories
        try:
            X = simulate_hopf_oscillator_network(
                **sim_params)  # (n_rollouts, T, 12)

            # drop transient
            X_post = X[:, T_burn:, :]  # (n_rollouts, T_eff, 12)

            # evaluate dynamics
            diagnostics = evaluate_dynamics(
                X_post,
                dt=dt,
                n_memories=4,
                burn_frac=0.0,  # already burned in
                seed=seed,
            )

            # compute weighted total score
            # emphasize: attractor diversity, geometry, phase order, saturation
            total_score = (
                2.0 * diagnostics["n_attractors"]
                + 3.0 * diagnostics["attractor_separation"]
                + 2.0 * diagnostics["geometry_score"]
                + 1.5 * diagnostics["phase_order_score"]
                + 1.0 * diagnostics["mean_saturation"]
                + 1.0 * diagnostics["spectral_sharpness"]
                + 0.5 * diagnostics["coupling_structure_score"]
            )

            # generate summary for logging and storage
            summary = summarize_diagnostics(diagnostics)

            # collect results
            result = {
                "params": params_dict,
                "total_score": float(total_score),
                "summary": summary,
            }
            result.update({k: v for k, v in diagnostics.items()
                          # exclude large arrays
                           if not k.endswith("_matrix")})

            results.append(result)

            # print progress
            print(f"[{len(results)}/{len(combinations)}] "
                  f"score={total_score:.2f} | {summary}")

        except Exception as e:
            # skip configurations that fail
            continue

    # sort by total score descending
    results.sort(key=lambda x: x["total_score"], reverse=True)

    # save to json
    outdir = Path("experiments/results/hopf_oscillator_test")
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "dynamics_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)


def main():
    # test_wilson_cowan_adaptation_sim()
    test_hopf_oscillator_sim()
    # sweep_dynamics_and_score()
    # test_bio_ode_model_sim()
    # test_custom_ode_sim()


if __name__ == "__main__":
    main()
