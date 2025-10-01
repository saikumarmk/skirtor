"""
skirtor_app.py

Interactive SKIRTOR AGN Torus Model Visualizer
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import h5py
from pathlib import Path

# Add the project root to the path
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Page config
st.set_page_config(page_title="Create your own AGN!", layout="wide")


# Load HDF5 database
@st.cache_resource
def load_hdf5_database():
    """Load optimized HDF5 database once"""
    return h5py.File("skirtor_optimized.h5", "r")


def get_sed_data(h5file, t, p, q, oa, R, Mcl, i):
    """Retrieve SED for specific parameters from optimized structure"""
    params = h5file["parameters"][:]

    # Find matching model
    mask = (
        (params["t"] == t)
        & (np.abs(params["p"] - p) < 0.01)  # Float comparison with tolerance
        & (np.abs(params["q"] - q) < 0.01)
        & (params["oa"] == oa)
        & (params["R"] == R)
        & (np.abs(params["Mcl"] - Mcl) < 0.01)
        & (params["i"] == i)
    )

    indices = np.where(mask)[0]

    if len(indices) == 0:
        return None, None

    idx = indices[0]

    wavelength = h5file["wavelength"][:]
    fluxes = h5file["fluxes"][idx]

    return wavelength, fluxes

def draw_torus_schematic(ax, oa, inclination, R):
    """
    Draw a simplified 2D schematic of the torus geometry
    Observer is always fixed on the right, torus rotates based on inclination
    
    oa: Opening angle (half-angle of the dust-free cone) in degrees
    inclination: Viewing angle in degrees (0 = face-on, 90 = edge-on)
    R: Ratio of outer radius to inner radius
    """
    
    # Torus radii
    r_outer = 1.5
    r_inner = r_outer / R
    
    # Opening angle defines the dust-free cone
    oa_rad = np.radians(oa)
    
    n_points = 50
    
    # Upper torus section: sweeps from (90° + oa) to (90° - oa)
    # Only the arc between these angles
    theta_upper_start = np.pi / 2 + oa_rad  # Upper bound
    theta_upper_end = np.pi / 2 - oa_rad    # Lower bound
    
    theta_upper = np.linspace(theta_upper_start, theta_upper_end, n_points)
    
    # Outer arc
    x_outer_upper = r_outer * np.cos(theta_upper)
    y_outer_upper = r_outer * np.sin(theta_upper)
    
    # Inner arc (reversed)
    x_inner_upper = r_inner * np.cos(theta_upper[::-1])
    y_inner_upper = r_inner * np.sin(theta_upper[::-1])
    
    # Complete wedge with straight line closures
    x_wedge_upper = np.concatenate([x_outer_upper, x_inner_upper, [x_outer_upper[0]]])
    y_wedge_upper = np.concatenate([y_outer_upper, y_inner_upper, [y_outer_upper[0]]])
    
    # Lower torus section: sweeps from (-90° - oa) to (-90° + oa)
    # Only the arc between these angles
    theta_lower_start = -np.pi / 2 - oa_rad  # Lower bound
    theta_lower_end = -np.pi / 2 + oa_rad    # Upper bound
    
    theta_lower = np.linspace(theta_lower_start, theta_lower_end, n_points)
    
    # Outer arc
    x_outer_lower = r_outer * np.cos(theta_lower)
    y_outer_lower = r_outer * np.sin(theta_lower)
    
    # Inner arc (reversed)
    x_inner_lower = r_inner * np.cos(theta_lower[::-1])
    y_inner_lower = r_inner * np.sin(theta_lower[::-1])
    
    # Complete wedge with straight line closures
    x_wedge_lower = np.concatenate([x_outer_lower, x_inner_lower, [x_outer_lower[0]]])
    y_wedge_lower = np.concatenate([y_outer_lower, y_inner_lower, [y_outer_lower[0]]])
    
    # Apply rotation for inclination
    inc_rad = np.radians(inclination)
    cos_inc = np.cos(inc_rad)
    sin_inc = np.sin(inc_rad)
    
    # Rotate upper wedge
    x_upper_rot = x_wedge_upper * cos_inc - y_wedge_upper * sin_inc
    y_upper_rot = x_wedge_upper * sin_inc + y_wedge_upper * cos_inc
    
    # Rotate lower wedge
    x_lower_rot = x_wedge_lower * cos_inc - y_wedge_lower * sin_inc
    y_lower_rot = x_wedge_lower * sin_inc + y_wedge_lower * cos_inc
    
    # Draw filled wedges
    ax.fill(
        x_upper_rot,
        y_upper_rot,
        color="lightgray",
        alpha=0.9,
        edgecolor="white",
        linewidth=2,
    )
    ax.fill(
        x_lower_rot,
        y_lower_rot,
        color="lightgray",
        alpha=0.9,
        edgecolor="white",
        linewidth=2,
    )
    
    # Draw all four cone edge lines (solid lines) to create the >< pattern
    cone_length = 2.0
    
    # Upper cone edges
    angle_upper_top = np.pi / 2 + oa_rad
    angle_upper_bot = np.pi / 2 - oa_rad
    
    # Upper top edge
    cx_ut = cone_length * np.cos(angle_upper_top)
    cy_ut = cone_length * np.sin(angle_upper_top)
    
    # Upper bottom edge
    cx_ub = cone_length * np.cos(angle_upper_bot)
    cy_ub = cone_length * np.sin(angle_upper_bot)
    
    # Lower cone edges
    angle_lower_top = -np.pi / 2 + oa_rad
    angle_lower_bot = -np.pi / 2 - oa_rad
    
    # Lower top edge
    cx_lt = cone_length * np.cos(angle_lower_top)
    cy_lt = cone_length * np.sin(angle_lower_top)
    
    # Lower bottom edge
    cx_lb = cone_length * np.cos(angle_lower_bot)
    cy_lb = cone_length * np.sin(angle_lower_bot)
    
    # Rotate all cone lines
    cx_ut_rot = cx_ut * cos_inc - cy_ut * sin_inc
    cy_ut_rot = cx_ut * sin_inc + cy_ut * cos_inc
    
    cx_ub_rot = cx_ub * cos_inc - cy_ub * sin_inc
    cy_ub_rot = cx_ub * sin_inc + cy_ub * cos_inc
    
    cx_lt_rot = cx_lt * cos_inc - cy_lt * sin_inc
    cy_lt_rot = cx_lt * sin_inc + cy_lt * cos_inc
    
    cx_lb_rot = cx_lb * cos_inc - cy_lb * sin_inc
    cy_lb_rot = cx_lb * sin_inc + cy_lb * cos_inc
    
    # Draw all four cone edge lines as SOLID lines
    ax.plot([0, cx_ut_rot], [0, cy_ut_rot], 'w-', linewidth=1.5, alpha=0.8)
    ax.plot([0, cx_ub_rot], [0, cy_ub_rot], 'w-', linewidth=1.5, alpha=0.8)
    ax.plot([0, cx_lt_rot], [0, cy_lt_rot], 'w-', linewidth=1.5, alpha=0.8)
    ax.plot([0, cx_lb_rot], [0, cy_lb_rot], 'w-', linewidth=1.5, alpha=0.8)
    
    # Central black hole
    circle = patches.Circle((0, 0), 0.1, color="black", zorder=10)
    ax.add_patch(circle)
    
    # Observer - always on the right
    observer_x = 2.5
    ax.annotate(
        "Observer",
        xy=(0.3, 0),
        xytext=(observer_x, 0),
        fontsize=16,
        color="white",
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="->", color="white", lw=2),
    )
    
    # Set equal aspect and limits
    ax.set_aspect("equal")
    ax.set_xlim(-2.5, 3.5)
    ax.set_ylim(-2.5, 2.5)
    ax.axis("off")
    ax.set_facecolor("#1a1a2e")

# Title and description
st.markdown("# Create your own AGN!")
st.markdown(
    """
This app displays the emission from an AGN as a function of various torus parameters 
using a radiative transfer based clumpy torus model **SKIRTOR** 
([Stalevski+'12](https://ui.adsabs.harvard.edu/abs/2012MNRAS.420.2756S) and 
[Stalevski+'16](https://ui.adsabs.harvard.edu/abs/2016MNRAS.458.2288S)).
"""
)

# Load database
try:
    h5file = load_hdf5_database()

    # Get available parameter values from attributes
    t_values = sorted(h5file.attrs["t_values"])
    p_values = sorted(h5file.attrs["p_values"])
    q_values = sorted(h5file.attrs["q_values"])
    oa_values = sorted(h5file.attrs["oa_values"])
    R_values = sorted(h5file.attrs["R_values"])
    Mcl_values = sorted(h5file.attrs["Mcl_values"])
    i_values = sorted(h5file.attrs["i_values"])

except Exception as e:
    st.error(f"Error loading HDF5 database: {e}")
    st.info(
        "Please ensure 'skirtor_optimized.h5' is in the same directory as this script."
    )
    st.stop()

# Sidebar with sliders
st.sidebar.markdown("### Use the following sliders to vary torus parameters.")

i = st.sidebar.select_slider(
    "i: inclination angle",
    options=i_values,
    value=i_values[len(i_values) // 2] if len(i_values) > 0 else 0,
)

oa = st.sidebar.select_slider(
    "oa: Angle from equator to edge of torus",
    options=oa_values,
    value=oa_values[len(oa_values) // 2] if len(oa_values) > 0 else 40,
)

R = st.sidebar.select_slider(
    "R: Ratio of outer radius to inner radius",
    options=R_values,
    value=R_values[0] if len(R_values) > 0 else 10,
)

t = st.sidebar.select_slider(
    "t: Average edge-on optical depth at 9.7 micron",
    options=t_values,
    value=t_values[0] if len(t_values) > 0 else 3,
)

p = st.sidebar.select_slider(
    "p: Index for radial density gradient",
    options=p_values,
    value=p_values[0] if len(p_values) > 0 else 0.0,
)

q = st.sidebar.select_slider(
    "q: Index for angular density gradient",
    options=q_values,
    value=q_values[0] if len(q_values) > 0 else 0.0,
)

# Fixed parameter (always 0.97 in SKIRTOR models)
Mcl = 0.97

# Create two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    # Draw torus geometry schematic
    fig_geom, ax_geom = plt.subplots(figsize=(6, 6), facecolor="#0e1117")
    ax_geom.set_facecolor("#0e1117")

    # Draw torus shape based on parameters
    draw_torus_schematic(ax_geom, oa, i, R)

    ax_geom.set_xlim(-2, 2)
    ax_geom.set_ylim(-2, 2)
    ax_geom.axis("off")

    st.pyplot(fig_geom)
    plt.close(fig_geom)

with col2:
    # Load and plot SED
    wavelength, fluxes = get_sed_data(h5file, t, p, q, oa, R, Mcl, i)

    if wavelength is not None and fluxes is not None:
        # Extract flux components
        total_flux = fluxes[:, 0]  # Column 0: total flux
        dust_emission = fluxes[:, 3]  # Column 3: dust emission
        transparent = fluxes[:, 5]  # Column 5: transparent (unobscured disk)

        # Create Plotly figure
        fig_sed = go.Figure()

        # Add traces
        fig_sed.add_trace(
            go.Scatter(
                x=wavelength,
                y=total_flux,
                mode="lines",
                name="Total emission",
                line=dict(color="#636EFA", width=2),
                hovertemplate="λ: %{x:.3f} μm<br>Flux: %{y:.3e} W/m²<extra></extra>",
            )
        )

        fig_sed.add_trace(
            go.Scatter(
                x=wavelength,
                y=dust_emission,
                mode="lines",
                name="Torus dust",
                line=dict(color="#EF553B", width=2),
                hovertemplate="λ: %{x:.3f} μm<br>Flux: %{y:.3e} W/m²<extra></extra>",
            )
        )

        fig_sed.add_trace(
            go.Scatter(
                x=wavelength,
                y=transparent,
                mode="lines",
                name="Unobscured accretion disk",
                line=dict(color="#00CC96", width=2),
                hovertemplate="λ: %{x:.3f} μm<br>Flux: %{y:.3e} W/m²<extra></extra>",
            )
        )

        # Calculate reasonable y-axis limits
        max_flux = max(
            np.max(total_flux[total_flux > 0]),
            np.max(dust_emission[dust_emission > 0]),
            np.max(transparent[transparent > 0]),
        )

        min_flux = min(
            np.min(total_flux[total_flux > 0]),
            np.min(dust_emission[dust_emission > 0]),
            np.min(transparent[transparent > 0]),
        )

        # Set y-range with padding
        y_min = max(min_flux * 0.1, 1e-18)
        y_max = min(max_flux * 10, 1e-10)

        # Update layout
        fig_sed.update_layout(
            xaxis_title="Wavelength (micron)",
            yaxis_title="Flux (νFν)",
            xaxis_type="log",
            yaxis_type="log",
            xaxis=dict(
                range=[np.log10(0.001), np.log10(1000)],
                gridcolor="rgba(128, 128, 128, 0.2)",
                showline=True,
                linewidth=1,
                linecolor="white",
                mirror=True,
            ),
            yaxis=dict(
                range=[np.log10(y_min), np.log10(y_max)],
                gridcolor="rgba(128, 128, 128, 0.2)",
                showline=True,
                linewidth=1,
                linecolor="white",
                mirror=True,
            ),
            plot_bgcolor="#262730",
            paper_bgcolor="#0e1117",
            font=dict(color="white", size=12),
            legend=dict(
                x=0.02,
                y=0.98,
                bordercolor="white",
                borderwidth=1,
                bgcolor="rgba(0,0,0,0.5)",  # Semi-transparent background
                font=dict(color="white", size=12),  # Explicit legend font color
            ),
            hovermode="x unified",
            height=500,
            margin=dict(l=60, r=20, t=20, b=60),
        )

        st.plotly_chart(fig_sed, use_container_width=True)
    else:
        st.error(f"Model not found for selected parameters")
        st.info(f"Parameters: t={t}, p={p}, q={q}, oa={oa}, R={R}, Mcl={Mcl}, i={i}")