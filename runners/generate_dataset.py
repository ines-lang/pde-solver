import os
import h5py
from exponax.burgers import Burgers1D, Burgers2D, Burgers3D
from utils.initial_conditions import get_initial_condition
import jax.numpy as jnp
from exponax import run

def generate_dataset(config):
    shape = tuple(config["shape"])
    dt = config["dt"]
    num_steps = config["num_steps"]
    save_freq = config["save_freq"]
    nu = config["nu"]
    pde = config["pde"]
    ic_name = config["ic"]

    if config["dim"] == 1:
        stepper = Burgers1D(nu=nu)
    elif config["dim"] == 2:
        stepper = Burgers2D(nu=nu)
    elif config["dim"] == 3:
        stepper = Burgers3D(nu=nu)
    else:
        raise ValueError("Only 1D, 2D, and 3D Burgers supported in this version.")

    u0 = get_initial_condition(ic_name, shape)
    u_history = run(stepper, u0, dt=dt, steps=num_steps, save_every=save_freq)

    # Expand to shape (simulations, time, Nx[, Ny[, Nz]], channels)
    u_history = jnp.expand_dims(u_history, axis=0)
    u_history = jnp.expand_dims(u_history, axis=-1)

    base_dir = os.path.join("data", pde, ic_name)
    os.makedirs(base_dir, exist_ok=True)
    file_path = os.path.join(base_dir, "dataset.h5")

    seed = config.get("seed", 0)
    with h5py.File(file_path, "w") as h5file:
        dataset_name = f"velocity_{seed:03d}"
        h5file.create_dataset(dataset_name, data=u_history[0])
        for key, value in config.items():
            h5file.attrs[key] = str(value)

        print(f"\nFile created at {file_path}")

        def print_structure(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"Group: {name}")
            elif isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {name} - Shape: {obj.shape}, Dtype: {obj.dtype}")

        h5file.visititems(print_structure)
