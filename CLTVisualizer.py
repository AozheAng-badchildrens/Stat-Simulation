import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import stats

rng = np.random.default_rng(42)

def get_sampler(name: str):
    # Returns:
    # - sampler(size): draws from the chosen base distribution
    # - mu: true mean
    # - sigma: true std dev
    if name == "uniform":
        a, b = 0.0, 1.0
        return (lambda size: rng.uniform(a, b, size=size), (a + b) / 2, np.sqrt((b - a) ** 2 / 12), False)
    if name == "gamma":
        k, theta = 2.0, 2.0
        return (lambda size: rng.gamma(shape=k, scale=theta, size=size), k * theta, np.sqrt(k * theta**2), False)
    if name == "chi2":
        df = 4
        return (lambda size: rng.chisquare(df=df, size=size), df, np.sqrt(2 * df), False)
    if name == "poisson":
        lam = 4.0
        return (lambda size: rng.poisson(lam=lam, size=size), lam, np.sqrt(lam), True)
    raise ValueError(f"Unknown distribution: {name}")

def sample_means(sampler, n: int, reps: int = 20000):
    # Draw reps batches of size n, then mean each row
    x = sampler((reps, n))
    return x.mean(axis=1)

def plot_clt_for_distribution(dist_name: str, n_values=(1, 2, 5, 10, 30, 100), reps=20000):
    sampler, mu, sigma, _ = get_sampler(dist_name)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    for ax, n in zip(axes, n_values):
        means = sample_means(sampler, n=n, reps=reps)

        # Histogram of sampling distribution of sample mean
        ax.hist(means, bins=60, density=True, alpha=0.65, color="skyblue", edgecolor="black")

        # CLT normal approximation
        sd_mean = sigma / np.sqrt(n)
        x = np.linspace(means.min(), means.max(), 500)
        y = stats.norm.pdf(x, loc=mu, scale=sd_mean)
        ax.plot(x, y, "r-", lw=2, label=f"N({mu:.2f}, {sd_mean**2:.3f})")

        ax.set_title(f"{dist_name}, n={n}")
        ax.legend(fontsize=8)

    fig.suptitle(f"CLT visualization for {dist_name}", fontsize=14)
    fig.tight_layout()
    plt.show()

def animate_clt(dist_name: str, n: int = 30, total_reps: int = 20000, chunk_size: int = 250, bins: int = 60):
    """
    Animate the CLT by streaming batches of sample means into a histogram.
    """
    sampler, mu, sigma, is_discrete = get_sampler(dist_name)
    means = sample_means(sampler, n=n, reps=total_reps)

    # Fixed x-range keeps the histogram stable across frames
    sd_mean = sigma / np.sqrt(n)
    x_min = mu - 5 * sd_mean
    x_max = mu + 5 * sd_mean
    x_grid = np.linspace(x_min, x_max, 500)
    y_grid = stats.norm.pdf(x_grid, loc=mu, scale=sd_mean)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel("Sample mean")
    title = ax.set_title(f"CLT animation: {dist_name}, n={n}, reps=0")

    discrete_mode = is_discrete
    if discrete_mode:
        # For Poisson means, use discrete lattice mass to avoid histogram bin artifacts.
        mean_steps = np.rint(means * n).astype(np.int64)
        min_step = int(mean_steps.min())
        max_step = int(mean_steps.max())
        support_steps = np.arange(min_step, max_step + 1, dtype=np.int64)
        support_x = support_steps / n
        step_width = 0.9 / n
        counts = np.zeros(support_steps.size, dtype=np.int64)

        ax.set_xlim(support_x[0] - step_width, support_x[-1] + step_width)
        normal_mass = stats.norm.pdf(support_x, loc=mu, scale=sd_mean) * (1.0 / n)
        initial_top = max(1.25 * normal_mass.max(), 0.02)
        ax.set_ylim(0, initial_top)
        ax.set_ylabel("Probability mass")

        bars = ax.bar(
            support_x,
            np.zeros_like(support_x, dtype=float),
            width=step_width,
            align="center",
            color="skyblue",
            edgecolor="black",
            alpha=0.75,
            label="Empirical mass",
        )
        normal_line, = ax.plot(
            support_x,
            normal_mass,
            "r-",
            lw=2,
            label=f"Normal mass approx N({mu:.2f}, {sd_mean**2:.3f})",
        )
    else:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, y_grid.max() * 1.25)
        ax.set_ylabel("Density")
        normal_line, = ax.plot(x_grid, y_grid, "r-", lw=2, label=f"Normal approx N({mu:.2f}, {sd_mean**2:.3f})")

        # Precompute histogram structure once and update bar heights per frame.
        bin_edges = np.linspace(x_min, x_max, bins + 1)
        bin_widths = np.diff(bin_edges)
        bars = ax.bar(
            bin_edges[:-1],
            np.zeros(bins),
            width=bin_widths,
            align="edge",
            color="skyblue",
            edgecolor="black",
            alpha=0.75,
            label="Empirical density",
        )
        counts = np.zeros(bins, dtype=np.int64)

    ax.legend()

    processed = 0

    # Dynamic schedule: small increments at the beginning, then accelerate.
    # `chunk_size` now acts as the max chunk size for late frames.
    min_chunk = max(5, chunk_size // 10)
    growth = 1.12
    increments = []
    current_chunk = float(min_chunk)
    covered = 0
    while covered < total_reps:
        step = int(min(chunk_size, max(min_chunk, round(current_chunk))))
        increments.append(step)
        covered += step
        current_chunk *= growth

    frame_targets = np.minimum(np.cumsum(increments), total_reps)

    def update(frame_idx):
        nonlocal processed
        k = int(frame_targets[frame_idx])

        if k > processed:
            # Incrementally update counts instead of rescanning means[:k] every frame.
            if discrete_mode:
                new_counts = np.bincount(mean_steps[processed:k] - min_step, minlength=counts.size)
                counts[:] += new_counts
            else:
                new_counts, _ = np.histogram(means[processed:k], bins=bin_edges)
                counts[:] += new_counts
            processed = k

        if discrete_mode:
            heights = counts / k
        else:
            heights = counts / (k * bin_widths)

        for rect, h in zip(bars, heights):
            rect.set_height(h)

        # Keep y-limits stable unless empirical heights need extra headroom.
        baseline_top = (normal_line.get_ydata().max() * 1.25) if normal_line.get_ydata().size else 1.0
        target_top = max(baseline_top, heights.max() * 1.15 if heights.size else 0.0)
        ax.set_ylim(0, target_top)

        title.set_text(f"CLT animation: {dist_name}, n={n}, reps={k}/{total_reps}")
        return list(bars) + [normal_line, title]

    n_frames = len(frame_targets)
    anim = FuncAnimation(fig, update, frames=n_frames, interval=120, repeat=False, blit=False)
    plt.tight_layout()
    plt.show()
    return anim

if __name__ == "__main__":
    # Static grid view:
    # for d in ["uniform", "gamma", "chi2", "poisson"]:
    #     plot_clt_for_distribution(d)

    # Animated single-case view:
    animate_clt(dist_name="poisson", n=5, total_reps=20000, chunk_size=250, bins=60)

# what I want to do in this project
# 1. Being able to support different distributions with diff parameters
# 2. Visualize the central limit theorem 
# 3. Then given some sample data, be able to calculate the MLE for the parameters of the distribution that best fits the data, and visualize the fit.
# 4. Biased vs unbiased estimators, and how the bias decreases as the sample size increases.
# 5. Consistency of estimators, and how the variance decreases as the sample size increases.
# 6. CRLB and how it provides a lower bound on the variance of unbiased estimators.
# 7. Visualize linear combinations of random variables (E.g sum of chi squared random variables follows a gamma distribution, etc.)
# 8. Figure out the backend and frontend 
