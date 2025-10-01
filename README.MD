

# SKIRTOR Visualiser

Open source implementation of [this Streamlit App](https://skirtor.streamlit.app/). The [SKIRTOR](https://sites.google.com/site/skirtorus/) library of model SEDS essentially consists of 19,000 files, with each file representing a different set of parameters. This was extremely large and cumbersome to work with, so I converted the files to a single HDF5 file with an optimized structure for fast access. This app allows you to visualise the SEDs for different parameters. Below is a guide on how to use the HDF5 file, as this is much more convenient to carry around, compared to 300MB of text files.

## SKIRTOR HDF5 Usage Guide

The compressed HDF5 file uses a **flat optimized structure** that stores:
- A single wavelength grid (shared by all models)
- A 3D flux array for all models
- A parameter lookup table with structured dtype

This design reduces file size by ~20x and enables fast parameter-based queries.

---

## File Structure

```
skirtor_optimized.h5
├── wavelength         # 1D array: wavelength grid in microns
├── fluxes            # 3D array: (n_models, n_wavelengths, 6)
├── parameters        # Structured array: model parameters
└── attributes        # Metadata and parameter value lists
```

### Datasets

| Dataset | Shape | Type | Description |
|---------|-------|------|-------------|
| `wavelength` | `(n_wavelengths,)` | float32 | Wavelength grid in microns (shared by all models) |
| `fluxes` | `(n_models, n_wavelengths, 6)` | float32 | Flux components for each model |
| `parameters` | `(n_models,)` | structured | Model parameters: t, p, q, oa, R, Mcl, i |

### Flux Components (axis 2 of `fluxes`)

| Index | Component | Description |
|-------|-----------|-------------|
| 0 | Total flux | Total emission (λF_λ in W/m²) |
| 1 | Direct stellar | Direct stellar flux |
| 2 | Scattered stellar | Scattered stellar flux |
| 3 | Dust emission | Total dust emission (torus) |
| 4 | Dust scattered | Dust emission scattered flux |
| 5 | Transparent | Unobscured accretion disk |

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `t` | int | Average edge-on optical depth at 9.7 μm |
| `p` | float | Radial density gradient index |
| `q` | float | Angular density gradient index |
| `oa` | int | Opening angle (half-angle from equator) |
| `R` | int | Ratio of outer to inner radius |
| `Mcl` | float | Mass fraction in clumps (fixed at 0.97) |
| `i` | int | Inclination angle (0° = face-on, 90° = edge-on) |

---

## Basic Usage

### 1. Open the HDF5 File

```python
import h5py
import numpy as np

# Open the file (read-only)
h5file = h5py.File('skirtor_optimized.h5', 'r')

# Check what's inside
print("Datasets:", list(h5file.keys()))
print("Attributes:", list(h5file.attrs.keys()))
```

### 2. Load Common Data

```python
# Load wavelength grid (shared by all models)
wavelength = h5file['wavelength'][:]  # Shape: (n_wavelengths,)
print(f"Wavelength range: {wavelength.min():.4f} - {wavelength.max():.1f} microns")

# Load parameter table
parameters = h5file['parameters'][:]  # Structured array
print(f"Number of models: {len(parameters)}")

# Get metadata
n_models = h5file.attrs['n_models']
distance = h5file.attrs['distance']  # "10 Mpc"
luminosity = h5file.attrs['luminosity']  # "10^11 L_solar"
```

### 3. Get Available Parameter Values

```python
# Available parameter values are stored as attributes
t_values = sorted(h5file.attrs['t_values'])
p_values = sorted(h5file.attrs['p_values'])
q_values = sorted(h5file.attrs['q_values'])
oa_values = sorted(h5file.attrs['oa_values'])
R_values = sorted(h5file.attrs['R_values'])
Mcl_values = sorted(h5file.attrs['Mcl_values'])
i_values = sorted(h5file.attrs['i_values'])

print(f"Available inclinations: {i_values}")
print(f"Available opening angles: {oa_values}")
```

---

## Querying Specific Models

### Method 1: Direct Parameter Matching (from `skirtor.py`)

```python
def get_sed_data(h5file, t, p, q, oa, R, Mcl, i):
    """
    Retrieve SED for specific parameters
    
    Returns:
    --------
    wavelength : array
        Wavelength grid in microns
    fluxes : array
        Flux components, shape (n_wavelengths, 6)
    """
    params = h5file['parameters'][:]
    
    # Find matching model (with float tolerance)
    mask = (
        (params['t'] == t) &
        (np.abs(params['p'] - p) < 0.01) &  # Float comparison
        (np.abs(params['q'] - q) < 0.01) &
        (params['oa'] == oa) &
        (params['R'] == R) &
        (np.abs(params['Mcl'] - Mcl) < 0.01) &
        (params['i'] == i)
    )
    
    indices = np.where(mask)[0]
    
    if len(indices) == 0:
        return None, None
    
    idx = indices[0]
    
    # Get wavelength and flux data
    wavelength = h5file['wavelength'][:]
    fluxes = h5file['fluxes'][idx]  # Shape: (n_wavelengths, 6)
    
    return wavelength, fluxes


# Example usage
h5file = h5py.File('skirtor_optimized.h5', 'r')

wavelength, fluxes = get_sed_data(
    h5file,
    t=3,      # Optical depth
    p=0.0,    # Radial gradient
    q=0.0,    # Angular gradient
    oa=40,    # Opening angle
    R=10,     # Radius ratio
    Mcl=0.97, # Clump mass fraction (fixed)
    i=60      # Inclination
)

if wavelength is not None:
    # Extract flux components
    total_flux = fluxes[:, 0]
    dust_emission = fluxes[:, 3]
    transparent = fluxes[:, 5]
    
    print(f"SED retrieved successfully")
    print(f"Peak total flux: {total_flux.max():.2e} W/m²")
else:
    print("Model not found")

h5file.close()
```

### Method 2: Find All Models with Specific Properties

```python
# Find all edge-on models (i=90)
params = h5file['parameters'][:]
edge_on_mask = params['i'] == 90
edge_on_indices = np.where(edge_on_mask)[0]

print(f"Found {len(edge_on_indices)} edge-on models")

# Get flux data for all edge-on models
edge_on_fluxes = h5file['fluxes'][edge_on_indices]  # Shape: (n_edge_on, n_wavelengths, 6)

# Calculate mean edge-on spectrum
mean_edge_on = edge_on_fluxes[:, :, 0].mean(axis=0)  # Average total flux
```

### Method 3: Parameter Space Exploration

```python
# Find models with high optical depth and large opening angle
params = h5file['parameters'][:]
mask = (params['t'] >= 7) & (params['oa'] >= 60)
selected_indices = np.where(mask)[0]

print(f"Found {len(selected_indices)} models with t≥7 and oa≥60")

# Get parameter combinations
for idx in selected_indices[:5]:  # Show first 5
    p = params[idx]
    print(f"Model {idx}: t={p['t']}, p={p['p']}, q={p['q']}, oa={p['oa']}, R={p['R']}, i={p['i']}")
```

---

## Complete Example: Plot SED for Multiple Inclinations

```python
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Open file
h5file = h5py.File('skirtor_optimized.h5', 'r')
wavelength = h5file['wavelength'][:]

# Fixed parameters
t, p, q, oa, R, Mcl = 3, 0.0, 0.0, 40, 10, 0.97

# Inclinations to compare
inclinations = [0, 30, 60, 90]
colors = ['blue', 'green', 'orange', 'red']

plt.figure(figsize=(10, 6))

for i_val, color in zip(inclinations, colors):
    wavelength, fluxes = get_sed_data(h5file, t, p, q, oa, R, Mcl, i_val)
    
    if wavelength is not None:
        total_flux = fluxes[:, 0]
        plt.loglog(wavelength, total_flux, label=f'i={i_val}°', color=color, linewidth=2)

plt.xlabel('Wavelength (μm)', fontsize=12)
plt.ylabel('Flux (λF$_λ$) [W/m²]', fontsize=12)
plt.title('SKIRTOR SEDs: Inclination Dependence', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('skirtor_inclination_comparison.pdf')
plt.show()

h5file.close()
```

---

## Advanced Usage

### Efficient Batch Loading

```python
# Load wavelength once (shared)
wavelength = h5file['wavelength'][:]
params = h5file['parameters'][:]

# Find all models with specific criteria
mask = (params['t'] == 3) & (params['R'] == 10)
indices = np.where(mask)[0]

# Load all matching fluxes at once (efficient)
batch_fluxes = h5file['fluxes'][indices]  # Shape: (n_matches, n_wavelengths, 6)

print(f"Loaded {len(indices)} models in batch")
```

### Memory-Efficient Iteration

```python
# Don't load all fluxes into memory at once
n_models = h5file.attrs['n_models']

for idx in range(0, n_models, 100):  # Process in batches of 100
    batch_params = h5file['parameters'][idx:idx+100]
    batch_fluxes = h5file['fluxes'][idx:idx+100]
    
    # Process batch...
    pass
```

### Context Manager (Automatic Close)

```python
# Recommended: Use context manager to auto-close file
with h5py.File('skirtor_optimized.h5', 'r') as h5file:
    wavelength = h5file['wavelength'][:]
    # ... do work ...
    
# File is automatically closed here
```

---

## Caching for Streamlit/Dash Apps

```python
import streamlit as st

@st.cache_resource
def load_hdf5_database():
    """Load HDF5 database once and cache it"""
    return h5py.File('skirtor_optimized.h5', 'r')

# Use in app
h5file = load_hdf5_database()
wavelength, fluxes = get_sed_data(h5file, t, p, q, oa, R, Mcl, i)
```

---

## Common Tasks

### Task 1: Get all SEDs for a specific inclination

```python
params = h5file['parameters'][:]
wavelength = h5file['wavelength'][:]

target_inclination = 60
mask = params['i'] == target_inclination
indices = np.where(mask)[0]

print(f"Found {len(indices)} models with i={target_inclination}°")

# Get all flux data
all_fluxes = h5file['fluxes'][indices]  # Shape: (n_models, n_wavelengths, 6)
```

### Task 2: Calculate IR luminosity (integrate 8-1000 μm)

```python
wavelength, fluxes = get_sed_data(h5file, t, p, q, oa, R, Mcl, i)

if wavelength is not None:
    # Select IR wavelength range
    ir_mask = (wavelength >= 8) & (wavelength <= 1000)
    wl_ir = wavelength[ir_mask]
    flux_ir = fluxes[ir_mask, 0]  # Total flux
    
    # Integrate using trapezoidal rule
    from scipy.integrate import trapezoid
    L_IR = trapezoid(flux_ir, wl_ir)
    
    print(f"IR luminosity (8-1000 μm): {L_IR:.2e} W/m²·μm")
```

### Task 3: Export specific model to ASCII

```python
def export_sed_to_ascii(h5file, filename, t, p, q, oa, R, Mcl, i):
    """Export a specific SED to ASCII format"""
    wavelength, fluxes = get_sed_data(h5file, t, p, q, oa, R, Mcl, i)
    
    if wavelength is None:
        print("Model not found")
        return False
    
    # Prepare output data
    data = np.column_stack([wavelength, fluxes])
    
    # Write to file
    header = (
        f"SKIRTOR SED: t={t}, p={p}, q={q}, oa={oa}, R={R}, Mcl={Mcl}, i={i}\n"
        f"Columns: wavelength(μm), total, direct_stellar, scattered_stellar, "
        f"dust_emission, dust_scattered, transparent"
    )
    
    np.savetxt(filename, data, header=header, fmt='%.6e')
    print(f"Exported to {filename}")
    return True

# Usage
export_sed_to_ascii(h5file, 'sed_i60.txt', 3, 0.0, 0.0, 40, 10, 0.97, 60)
```

---

## Troubleshooting

### Problem: Model not found

**Cause:** Parameter values don't match any model in the database

**Solution:** Check available parameter values first
```python
t_values = h5file.attrs['t_values']
print(f"Available t values: {sorted(t_values)}")
```

### Problem: Float comparison issues

**Cause:** Floating-point parameters (p, q, Mcl) may not match exactly

**Solution:** Use tolerance-based comparison (shown in `get_sed_data()` example)

### Problem: Memory error loading all fluxes

**Cause:** The flux array can be large (e.g., 2GB for 5000+ models)

**Solution:** Use indexing to load only needed models
```python
# Don't do this:
# all_fluxes = h5file['fluxes'][:]  # Loads entire array

# Do this instead:
selected_fluxes = h5file['fluxes'][indices]  # Only selected models
```

---

## Performance Tips

1. **Load wavelength once**: It's shared by all models
2. **Use boolean indexing**: Faster than loops for filtering
3. **Batch operations**: Load multiple models at once when possible
4. **Keep file open**: Don't open/close repeatedly in loops
5. **Use context managers**: Ensure file is closed when done

---

## File Size Comparison

| Format | Size | Compression Ratio |
|--------|------|-------------------|
| Original ASCII files | ~1,000 MB | 1x |
| HDF5 (no compression) | ~250 MB | 4x |
| **HDF5 (GZIP level 9)** | **~50 MB** | **~20x** |

The optimized HDF5 format achieves ~20x compression while enabling fast parameter-based queries.

---

## References

- **Creating the file:** See `compressor.py`
- **Interactive app example:** See `skirtor.py`
- **SKIRTOR models:** [Stalevski et al. 2012](https://ui.adsabs.harvard.edu/abs/2012MNRAS.420.2756S), [2016](https://ui.adsabs.harvard.edu/abs/2016MNRAS.458.2288S)
- **HDF5 documentation:** https://docs.h5py.org/

---

## Quick Reference Card

```python
# Open file
h5file = h5py.File('skirtor_optimized.h5', 'r')

# Get data
wavelength = h5file['wavelength'][:]           # Wavelength grid
fluxes = h5file['fluxes'][:]                  # All flux data (large!)
parameters = h5file['parameters'][:]           # Parameter table

# Get metadata
n_models = h5file.attrs['n_models']
t_values = h5file.attrs['t_values']

# Query specific model
def get_sed_data(h5file, t, p, q, oa, R, Mcl, i):
    params = h5file['parameters'][:]
    mask = ((params['t'] == t) & 
            (np.abs(params['p'] - p) < 0.01) & 
            (np.abs(params['q'] - q) < 0.01) &
            (params['oa'] == oa) & 
            (params['R'] == R) & 
            (np.abs(params['Mcl'] - Mcl) < 0.01) & 
            (params['i'] == i))
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return None, None
    idx = indices[0]
    return h5file['wavelength'][:], h5file['fluxes'][idx]

# Use it
wavelength, fluxes = get_sed_data(h5file, 3, 0.0, 0.0, 40, 10, 0.97, 60)
total_flux = fluxes[:, 0]  # Component 0: total flux

# Close file
h5file.close()
```

