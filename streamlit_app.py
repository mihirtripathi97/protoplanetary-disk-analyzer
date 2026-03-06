import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="ALMA Disk Analyzer", layout="wide")

st.title("🌌 Protoplanetary Disk Analyzer")
st.markdown("""
Upload an ALMA FITS file to calculate the deprojected radial intensity profile and find the **R70** (70% flux) radius.
""")

# --- SIDEBAR PARAMETERS ---
st.sidebar.header("📂 Data & Geometry")
uploaded_file = st.file_uploader("Upload FITS file", type=["fits", "fit"])

if uploaded_file:
    with fits.open(uploaded_file) as hdul:
        header = hdul[0].header
        # Handle 4D ALMA cubes (Stokes, Freq, Y, X)
        data = np.nan_to_num(hdul[0].data)
        if data.ndim == 4:
            data = data[0, 0, :, :]
        elif data.ndim == 3:
            data = data[0, :, :]
        
        wcs = WCS(header).celestial
        ny, nx = data.shape

    # UI for Coordinates
    st.sidebar.subheader("Center Coordinates")
    coord_type = st.sidebar.radio("Coordinate System", ["Pixel", "Spatial (RA/Dec)"])
    
    if coord_type == "Pixel":
        cx = st.sidebar.number_input("Center X (px)", value=float(nx/2), step=0.5)
        cy = st.sidebar.number_input("Center Y (px)", value=float(ny/2), step=0.5)
        center = (cx, cy)
    else:
        ra = st.sidebar.number_input("RA (deg)", value=float(header.get('CRVAL1', 0)), format="%.6f")
        dec = st.sidebar.number_input("Dec (deg)", value=float(header.get('CRVAL2', 0)), format="%.6f")
        # Convert WCS to Pixel
        cx, cy = wcs.all_world2pix(ra, dec, 0)
        center = (cx, cy)

    # UI for Disk Geometry
    st.sidebar.subheader("Disk Orientation")
    inc = st.sidebar.slider("Inclination (deg)", 0.0, 90.0, 0.0, help="0 is face-on, 90 is edge-on")
    pa = st.sidebar.slider("Position Angle (deg)", 0.0, 360.0, 0.0, help="Measured North-to-East")

    # --- ANALYSIS LOGIC ---
    # 1. Coordinate Grids
    y, x = np.indices(data.shape)
    dx = x - cx
    dy = y - cy

    # 2. Rotation (PA)
    # Most astronomy PA conventions are North to East. 
    # We rotate the coordinates to align with the Major Axis.
    phi = np.radians(pa + 90)
    x_rot = dx * np.cos(phi) + dy * np.sin(phi)
    y_rot = -dx * np.sin(phi) + dy * np.cos(phi)

    # 3. Deprojection (Inclination)
    cos_inc = np.cos(np.radians(inc))
    # Avoid division by zero for edge-on
    cos_inc = max(cos_inc, 0.01)
    r_deproj = np.sqrt(x_rot**2 + (y_rot / cos_inc)**2)

    # 4. Radial Binning
    r_max = st.sidebar.number_input("Max Radius for Analysis (px)", value=float(min(nx, ny)/2))
    bins = np.arange(0, r_max, 1)
    bin_centers = bins[:-1] + 0.5
    
    radial_profile = []
    for i in range(len(bins)-1):
        mask = (r_deproj >= bins[i]) & (r_deproj < bins[i+1])
        if np.any(mask):
            radial_profile.append(np.mean(data[mask]))
        else:
            radial_profile.append(0)
    
    radial_profile = np.array(radial_profile)

    # 5. Flux Calculation (R70)
    # Area of elliptical annulus: dA = 2 * pi * r * dr / cos(i)
    areas = 2 * np.pi * bin_centers / cos_inc
    flux_elements = radial_profile * areas
    cumulative_flux = np.cumsum(flux_elements)
    total_flux = cumulative_flux[-1]
    
    r_70 = np.interp(0.70 * total_flux, cumulative_flux, bin_centers)

    # --- VISUALIZATION ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image & Center")
        fig_img, ax_img = plt.subplots()
        im = ax_img.imshow(data, origin='lower', cmap='inferno')
        ax_img.scatter(cx, cy, color='cyan', marker='+', s=100, label='Center')
        plt.colorbar(im, label='Intensity')
        ax_img.legend()
        st.pyplot(fig_img)

    with col2:
        st.subheader("Deprojected Profile")
        fig_prof, ax_prof = plt.subplots()
        ax_prof.plot(bin_centers, radial_profile, color='white', lw=2)
        ax_prof.axvline(r_70, color='red', linestyle='--', label=f'R70: {r_70:.2f} px')
        ax_prof.set_xlabel("Radius (pixels)")
        ax_prof.set_ylabel("Mean Intensity")
        ax_prof.set_title("Radial Intensity Profile")
        ax_prof.legend()
        plt.style.use('dark_background')
        st.pyplot(fig_prof)

    st.success(f"**Analysis Complete!** The calculated R70 radius is **{r_70:.2f} pixels**.")

    # --- EXPORT DATA ---
    csv_data = f"radius_px,intensity\n"
    for r, i in zip(bin_centers, radial_profile):
        csv_data += f"{r},{i}\n"
    
    st.download_button(
        label="Download Radial Profile (CSV)",
        data=csv_data,
        file_name=f"profile_{uploaded_file.name}.csv",
        mime="text/csv"
    )

else:
    st.info("Please upload a FITS file from the sidebar to begin.")
