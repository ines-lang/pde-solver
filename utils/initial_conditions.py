import jax.numpy as jnp
import numpy as np

def get_initial_condition(name, shape):
    if name == "sine":
        if len(shape) == 1:
            x = jnp.linspace(0, 2 * jnp.pi, shape[0])
            return jnp.sin(x)
        elif len(shape) == 2:
            x = jnp.linspace(0, 2 * jnp.pi, shape[0])
            y = jnp.linspace(0, 2 * jnp.pi, shape[1])
            xx, yy = jnp.meshgrid(x, y, indexing='ij')
            return jnp.sin(xx) * jnp.sin(yy)
        elif len(shape) == 3:
            x = jnp.linspace(0, 2 * jnp.pi, shape[0])
            y = jnp.linspace(0, 2 * jnp.pi, shape[1])
            z = jnp.linspace(0, 2 * jnp.pi, shape[2])
            xx, yy, zz = jnp.meshgrid(x, y, z, indexing='ij')
            return jnp.sin(xx) * jnp.sin(yy) * jnp.sin(zz)
    elif name == "random":
        return jnp.array(np.random.rand(*shape))
    else:
        raise ValueError(f"Unknown initial condition '{name}'")
