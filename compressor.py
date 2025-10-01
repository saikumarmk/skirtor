"""
compressor.py

Convert SKIRTOR SED files directly to optimized flat HDF5 structure
"""

import h5py
import numpy as np
from pathlib import Path
import re
from tqdm import tqdm
import sys


def parse_filename(filename):
    """
    Extract parameters from SKIRTOR filename
    Example: t3_p0.5_q0.5_oa10_R10_Mcl0.97_i0_sed.dat
    """
    pattern = r"t(\d+)_p([\d.]+)_q([\d.]+)_oa(\d+)_R(\d+)_Mcl([\d.]+)_i(\d+)_sed\.dat"
    match = re.match(pattern, filename)

    if not match:
        return None

    return {
        "t": int(match.group(1)),
        "p": float(match.group(2)),
        "q": float(match.group(3)),
        "oa": int(match.group(4)),
        "R": int(match.group(5)),
        "Mcl": float(match.group(6)),
        "i": int(match.group(7)),
    }


def create_optimized_hdf5(
    input_dir, output_file="skirtor_optimized.h5", compression_level=9
):
    """
    Convert all SKIRTOR SED files directly to optimized flat HDF5 structure

    This creates a single 3D array for all flux data with a parameter lookup table,
    dramatically reducing file size and metadata overhead.

    Parameters:
    -----------
    input_dir : str or Path
        Directory containing SKIRTOR *_sed.dat files
    output_file : str
        Output HDF5 filename
    compression_level : int
        GZIP compression level (0-9, higher = better compression but slower)
    """

    input_path = Path(input_dir)
    sed_files = sorted(list(input_path.glob("*_sed.dat")))

    if len(sed_files) == 0:
        print(f"ERROR: No *_sed.dat files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(sed_files)} SED files")
    print(f"Creating optimized HDF5 archive: {output_file}")

    # First pass: determine dimensions and collect valid files
    valid_files = []
    param_list = []
    wavelength_data = None

    print("Scanning files...")
    for sed_file in tqdm(sed_files, desc="Validating files"):
        params = parse_filename(sed_file.name)

        if params is None:
            print(f"WARNING: Could not parse filename: {sed_file.name}")
            continue

        try:
            data = np.loadtxt(sed_file, comments="#")

            if wavelength_data is None:
                wavelength_data = data[:, 0]

            valid_files.append(sed_file)
            param_list.append(
                [
                    params["t"],
                    params["p"],
                    params["q"],
                    params["oa"],
                    params["R"],
                    params["Mcl"],
                    params["i"],
                ]
            )
        except Exception as e:
            print(f"WARNING: Error reading {sed_file.name}: {e}")
            continue

    n_models = len(valid_files)
    n_wavelengths = len(wavelength_data)
    n_flux_cols = 6

    print(f"\nValid models: {n_models}")
    print(f"Wavelength points: {n_wavelengths}")
    print(f"Creating optimized structure...")

    # Create HDF5 file with optimized structure
    with h5py.File(output_file, "w") as f:

        # Store wavelength grid once
        wl_ds = f.create_dataset(
            "wavelength",
            data=wavelength_data.astype("float32"),
            compression="gzip",
            compression_opts=compression_level,
        )
        wl_ds.attrs["units"] = "micron"
        wl_ds.attrs["description"] = "Wavelength grid (same for all models)"

        # Create single 3D array for all flux data
        print(
            f"Creating flux array: shape=({n_models}, {n_wavelengths}, {n_flux_cols})"
        )
        flux_dataset = f.create_dataset(
            "fluxes",
            shape=(n_models, n_wavelengths, n_flux_cols),
            dtype="float32",
            compression="gzip",
            compression_opts=compression_level,
            chunks=(1, n_wavelengths, n_flux_cols),  # Chunk by individual model
        )

        # Add column descriptions
        flux_dataset.attrs["column_0"] = "total flux; lambda*F_lambda (W/m2)"
        flux_dataset.attrs["column_1"] = "direct stellar flux; lambda*F_lambda (W/m2)"
        flux_dataset.attrs["column_2"] = (
            "scattered stellar flux; lambda*F_lambda (W/m2)"
        )
        flux_dataset.attrs["column_3"] = (
            "total dust emission flux; lambda*F_lambda (W/m2)"
        )
        flux_dataset.attrs["column_4"] = (
            "dust emission scattered flux; lambda*F_lambda (W/m2)"
        )
        flux_dataset.attrs["column_5"] = "transparent flux; lambda*F_lambda (W/m2)"

        # Load and store all flux data
        print("Loading flux data...")
        for idx, sed_file in enumerate(tqdm(valid_files, desc="Processing models")):
            data = np.loadtxt(sed_file, comments="#")

            # Verify wavelength grid matches
            if not np.allclose(data[:, 0], wavelength_data, rtol=1e-5):
                print(f"WARNING: Wavelength mismatch in {sed_file.name}")

            # Store flux data (columns 1-6)
            flux_dataset[idx] = data[:, 1:7].astype("float32")

        # Create parameter lookup table
        print("Creating parameter lookup table...")
        param_array = np.array(
            [tuple(p) for p in param_list],
            dtype=[
                ("t", "i4"),  # Optical depth
                ("p", "f4"),  # Radial density gradient
                ("q", "f4"),  # Angular density gradient
                ("oa", "i4"),  # Opening angle
                ("R", "i4"),  # Radius ratio
                ("Mcl", "f4"),  # Mass fraction in clumps
                ("i", "i4"),  # Inclination
            ],
        )

        param_ds = f.create_dataset(
            "parameters",
            data=param_array,
            compression="gzip",
            compression_opts=compression_level,
        )

        param_ds.attrs["description"] = "Model parameters for each flux array entry"

        # Store metadata
        f.attrs["n_models"] = n_models
        f.attrs["description"] = "SKIRTOR AGN torus SED models - optimized flat storage"
        f.attrs["source"] = "https://sites.google.com/site/skirtorus/"
        f.attrs["reference"] = (
            "Stalevski et al. 2012 MNRAS 420, 2756; Stalevski et al. 2016 MNRAS 458, 2288"
        )
        f.attrs["storage_format"] = "optimized_flat"
        f.attrs["distance"] = "10 Mpc"
        f.attrs["luminosity"] = "10^11 L_solar"

        # Store unique parameter values for sliders
        param_values = {
            "t": set(),
            "p": set(),
            "q": set(),
            "oa": set(),
            "R": set(),
            "Mcl": set(),
            "i": set(),
        }

        for p in param_list:
            param_values["t"].add(p[0])
            param_values["p"].add(p[1])
            param_values["q"].add(p[2])
            param_values["oa"].add(p[3])
            param_values["R"].add(p[4])
            param_values["Mcl"].add(p[5])
            param_values["i"].add(p[6])

        # Store as attributes
        for key, values in param_values.items():
            f.attrs[f"{key}_values"] = sorted(list(values))

        print(f"\nParameter ranges:")
        for key, values in param_values.items():
            sorted_vals = sorted(list(values))
            print(f"  {key}: {sorted_vals}")

    # Calculate compression statistics
    original_size = sum(f.stat().st_size for f in valid_files)
    compressed_size = Path(output_file).stat().st_size
    compression_ratio = original_size / compressed_size

    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"{'='*60}")
    print(f"Original size:     {original_size / 1e6:.2f} MB")
    print(f"Compressed size:   {compressed_size / 1e6:.2f} MB")
    print(f"Compression ratio: {compression_ratio:.1f}x")
    print(
        f"Space saved:       {(original_size - compressed_size) / 1e6:.2f} MB ({100*(1-1/compression_ratio):.1f}%)"
    )
    print(f"Output file:       {output_file}")


def verify_hdf5(hdf5_file, input_dir, n_samples=10):
    """
    Verify the optimized HDF5 by comparing against original files
    """

    print(f"\n{'='*60}")
    print(f"Verifying {hdf5_file}")
    print(f"{'='*60}")

    with h5py.File(hdf5_file, "r") as f:
        print(f"\nMetadata:")
        print(f"  Models: {f.attrs['n_models']}")
        print(f"  Storage format: {f.attrs['storage_format']}")

        print(f"\nDatasets:")
        print(
            f"  Wavelength: shape={f['wavelength'].shape}, dtype={f['wavelength'].dtype}"
        )
        print(f"  Fluxes: shape={f['fluxes'].shape}, dtype={f['fluxes'].dtype}")
        print(
            f"  Parameters: shape={f['parameters'].shape}, dtype={f['parameters'].dtype}"
        )

        # Verify random samples
        print(f"\nVerifying {n_samples} random samples...")

        n_models = f.attrs["n_models"]
        sample_indices = np.random.choice(
            n_models, min(n_samples, n_models), replace=False
        )

        params = f["parameters"][:]
        wavelength = f["wavelength"][:]

        input_path = Path(input_dir)

        for idx in sample_indices:
            p = params[idx]
            t, p_val, q, oa, R, Mcl, i = (
                p["t"],
                p["p"],
                p["q"],
                p["oa"],
                p["R"],
                p["Mcl"],
                p["i"],
            )

            # Construct original filename
            filename = f"t{t}_p{p_val}_q{q}_oa{oa}_R{R}_Mcl{Mcl}_i{i}_sed.dat"
            filepath = input_path / filename

            if not filepath.exists():
                print(f"WARNING: Original file not found: {filename}")
                continue

            # Load original
            data_orig = np.loadtxt(filepath, comments="#")

            # Load from HDF5
            flux_h5 = f["fluxes"][idx]

            # Compare
            if not np.allclose(data_orig[:, 1:7], flux_h5, rtol=1e-5):
                print(f"ERROR: Data mismatch for {filename}")
                return False

            print(f"✓ {filename}")

        print(f"\n✓ All {n_samples} samples verified successfully!")
        return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert SKIRTOR SED files to optimized HDF5 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_skirtor_to_hdf5_optimized.py /path/to/skirtor/data
  python convert_skirtor_to_hdf5_optimized.py data/skirtor --verify
  python convert_skirtor_to_hdf5_optimized.py data/skirtor -o custom.h5 -c 9
        """,
    )

    parser.add_argument(
        "input_dir", type=str, help="Directory containing SKIRTOR *_sed.dat files"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="skirtor_optimized.h5",
        help="Output HDF5 filename (default: skirtor_optimized.h5)",
    )
    parser.add_argument(
        "-c",
        "--compression",
        type=int,
        default=9,
        help="GZIP compression level 0-9 (default: 9)",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify archive after creation"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of samples to verify (default: 10)",
    )

    args = parser.parse_args()

    # Create optimized archive
    create_optimized_hdf5(args.input_dir, args.output, args.compression)

    # Verify if requested
    if args.verify:
        verify_hdf5(args.output, args.input_dir, args.samples)
