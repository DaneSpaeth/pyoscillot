from three_dim_star import ThreeDimStar, TwoDimProjector
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


out_dir = Path(
    "/home/dane/Documents/PhD/NGC4349-127/talk/new_simulation_plots")

star = ThreeDimStar()
projector = TwoDimProjector(star, inclination=60)

# Create plots
fig, ax = plt.subplots(1, figsize=(8, 8))
ax.imshow(projector.temperature(), origin="lower",
          cmap="hot", vmin=0, vmax=8000)
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.savefig(out_dir / "temp.png", dpi=600)
plt.close()

fig, ax = plt.subplots(1, figsize=(8, 8))
star = ThreeDimStar()
projector = TwoDimProjector(star, inclination=60)
star.create_rotation()
rot_map = projector.rotation()
rot_map = np.where(np.abs(rot_map) < 1e-5, np.nan, rot_map)
ax.imshow(rot_map, origin="lower",
          cmap="seismic", vmin=-3000, vmax=3000)
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.savefig(out_dir / "rotation.png", dpi=600)
plt.close()

fig, ax = plt.subplots(1, figsize=(8, 8))
star = ThreeDimStar()
projector = TwoDimProjector(star, inclination=60, line_of_sight=False)
star.add_pulsation(l=2, m=2, k=1.2, v_p=30)
ax.imshow(projector.pulsation_rad(), origin="lower",
          cmap="seismic", vmin=-50, vmax=50)
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.savefig(out_dir / "pulse_rad_nolos.png", dpi=600)
plt.close()

fig, ax = plt.subplots(1, figsize=(8, 8))
star = ThreeDimStar()
projector = TwoDimProjector(star, inclination=60, line_of_sight=False)
star.add_pulsation(l=2, m=2, k=1.2, v_p=30)
ax.imshow(projector.pulsation_phi(), origin="lower",
          cmap="seismic", vmin=-50, vmax=50)
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.savefig(out_dir / "pulse_phi_nolos.png", dpi=600)
plt.close()

fig, ax = plt.subplots(1, figsize=(8, 8))
star = ThreeDimStar()
projector = TwoDimProjector(star, inclination=60, line_of_sight=False)
star.add_pulsation(l=2, m=2, k=1.2, v_p=30)
ax.imshow(projector.pulsation_theta(), origin="lower",
          cmap="seismic", vmin=-50, vmax=50)
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.savefig(out_dir / "pulse_theta_nolos.png", dpi=600)
plt.close()

fig, ax = plt.subplots(1, figsize=(8, 8))
star = ThreeDimStar()
projector = TwoDimProjector(star, inclination=60, line_of_sight=True)
star.add_pulsation(l=2, m=2, k=1.2, v_p=30)
ax.imshow(projector.pulsation_rad(), origin="lower",
          cmap="seismic", vmin=-50, vmax=50)
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.savefig(out_dir / "pulse_rad_proj.png", dpi=600)
plt.close()

fig, ax = plt.subplots(1, figsize=(8, 8))
star = ThreeDimStar()
projector = TwoDimProjector(star, inclination=60, line_of_sight=True)
star.add_pulsation(l=2, m=2, k=1.2, v_p=30)
ax.imshow(projector.pulsation_phi(), origin="lower",
          cmap="seismic", vmin=-50, vmax=50)
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.savefig(out_dir / "pulse_phi_proj.png", dpi=600)
plt.close()

fig, ax = plt.subplots(1, figsize=(8, 8))
star = ThreeDimStar()
projector = TwoDimProjector(star, inclination=60, line_of_sight=True)
star.add_pulsation(l=2, m=2, k=1.2, v_p=30)
ax.imshow(projector.pulsation_theta(), origin="lower",
          cmap="seismic", vmin=-50, vmax=50)
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.savefig(out_dir / "pulse_theta_proj.png", dpi=600)
plt.close()

fig, ax = plt.subplots(1, figsize=(8, 8))
star = ThreeDimStar()
projector = TwoDimProjector(star, inclination=60, line_of_sight=True)
star.add_pulsation(l=2, m=2, k=1.2, v_p=30)
ax.imshow(projector.pulsation(), origin="lower",
          cmap="seismic", vmin=-50, vmax=50)
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.savefig(out_dir / "pulse_all_proj.png", dpi=600)
plt.close()
