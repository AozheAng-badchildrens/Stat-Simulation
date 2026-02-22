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
        return (lambda size: rng.uniform(a, b, size=size), (a + b) / 2, np.sqrt((b - a) ** 2 / 12))
    if name == "gamma":
        k, theta = 2.0, 2.0
        return (lambda size: rng.gamma(shape=k, scale=theta, size=size), k * theta, np.sqrt(k * theta**2))
    if name == "chi2":
        df = 4
        return (lambda size: rng.chisquare(df=df, size=size), df, np.sqrt(2 * df))
    if name == "poisson":
        lam = 4.0
        return (lambda size: rng.poisson(lam=lam, size=size), lam, np.sqrt(lam))
    raise ValueError(f"Unknown distribution: {name}")

def sample_means(sampler, n: int, reps: int = 20000):
    # Draw reps batches of size n, then mean each row
    x = sampler((reps, n))
    return x.mean(axis=1)

def plot_clt_for_distribution(dist_name: str, n_values=(1, 2, 5, 10, 30, 100), reps=20000):
    sampler, mu, sigma = get_sampler(dist_name)

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
    sampler, mu, sigma = get_sampler(dist_name)
    means = sample_means(sampler, n=n, reps=total_reps)

    # Fixed x-range keeps the histogram stable across frames
    sd_mean = sigma / np.sqrt(n)
    x_min = mu - 5 * sd_mean
    x_max = mu + 5 * sd_mean
    x_grid = np.linspace(x_min, x_max, 500)
    y_grid = stats.norm.pdf(x_grid, loc=mu, scale=sd_mean)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, y_grid.max() * 1.25)
    ax.set_xlabel("Sample mean")
    ax.set_ylabel("Density")
    title = ax.set_title(f"CLT animation: {dist_name}, n={n}, reps=0")
    ax.plot(x_grid, y_grid, "r-", lw=2, label=f"Normal approx N({mu:.2f}, {sd_mean**2:.3f})")
    ax.legend()

    def update(frame_idx):
        k = min((frame_idx + 1) * chunk_size, total_reps)
        ax.cla()
        ax.hist(
            means[:k],
            bins=bins,
            range=(x_min, x_max),
            density=True,
            color="skyblue",
            edgecolor="black",
            alpha=0.75,
        )
        ax.plot(x_grid, y_grid, "r-", lw=2, label=f"Normal approx N({mu:.2f}, {sd_mean**2:.3f})")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, y_grid.max() * 1.25)
        ax.set_xlabel("Sample mean")
        ax.set_ylabel("Density")
        title_text = f"CLT animation: {dist_name}, n={n}, reps={k}/{total_reps}"
        ax.set_title(title_text)
        ax.legend()
        return []

    n_frames = int(np.ceil(total_reps / chunk_size))
    anim = FuncAnimation(fig, update, frames=n_frames, interval=120, repeat=False, blit=False)
    plt.tight_layout()
    plt.show()
    return anim

if __name__ == "__main__":
    # Static grid view:
    # for d in ["uniform", "gamma", "chi2", "poisson"]:
    #     plot_clt_for_distribution(d)

    # Animated single-case view:
    animate_clt(dist_name="gamma", n=5, total_reps=20000, chunk_size=250, bins=60)

# what I want to do in this project
# 1. Being able to support different distributions with diff parameters
# 2. Visualize the central limit theorem 
# 3. Then given some sample data, be able to calculate the MLE for the parameters of the distribution that best fits the data, and visualize the fit.
# 4. Biased vs unbiased estimators, and how the bias decreases as the sample size increases.
# 5. Consistency of estimators, and how the variance decreases as the sample size increases.
# 6. CRLB and how it provides a lower bound on the variance of unbiased estimators.
# 7. Visualize linear combinations of random variables (E.g sum of chi squared random variables follows a gamma distribution, etc.)
# 8. Figure out the backend and frontend 
