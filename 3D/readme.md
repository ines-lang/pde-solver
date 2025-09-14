# 3D

To generate datasets in three dimensions, run **`generate_dataset.py`** from within the `3D/` folder.  
This ensures that the output directories are created correctly.

The resulting datasets can be found at:  
[https://huggingface.co/datasets/isuarez/pde_collection](https://huggingface.co/datasets/isuarez/pde_collection)


**Implementation of KdV in 3d note.**  
For the KdV equation, the rollout function expects the number of integration steps rather than a final simulation time. Therefore, in the implementation the number of steps was computed explicitly as:

```python
n_steps = int(np.ceil(t_end / dt_save))
rollout_fn = ex.rollout(kdv_stepper, n_steps, include_init=True)
```
