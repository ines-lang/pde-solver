import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import imageio.v3 as iio

def visualize_results(config):
    pde = config["pde"]
    ic_name = config["ic"]
    base_dir = os.path.join("data", pde, ic_name)
    file_path = os.path.join(base_dir, "dataset.h5")

    with h5py.File(file_path, 'r') as f:
        dataset_name = f"velocity_{config.get('seed', 0):03d}"
        u = f[dataset_name][:, ..., 0]  # Remove channel dim

    plot_dir = os.path.join(base_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    if config["dim"] == 1:
        fig, ax = plt.subplots(figsize=(12, u.shape[0] * 1.5))
        im = ax.imshow(u, aspect="auto", cmap="viridis")
        ax.set_title("1D Time Evolution")
        ax.set_xlabel("x")
        ax.set_ylabel("Time Frame")
        fig.colorbar(im, ax=ax, orientation='vertical')
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "apebench_view.png"))
        plt.close()

    elif config["dim"] in [2, 3]:
        frames = []
        for frame in u:
            if config["dim"] == 2:
                img = (255 * (frame - np.min(frame)) / (np.ptp(frame))).astype(np.uint8)
            elif config["dim"] == 3:
                mid_z = frame.shape[2] // 2
                img = (255 * (frame[:, :, mid_z] - np.min(frame)) / (np.ptp(frame))).astype(np.uint8)
            frames.append(img)

        mp4_path = os.path.join(plot_dir, "animation.mp4")
        iio.imwrite(mp4_path, np.stack(frames), plugin="pyav", fps=10)
