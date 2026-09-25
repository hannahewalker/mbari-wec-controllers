import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

# usage: plot_wave_pred.py [run_dir]   (default: latest sim pblog run folder)
if len(sys.argv) > 1:
    run_dir = sys.argv[1]
else:
    run_dir = os.path.realpath(os.path.expanduser('~/.pblogs/latest_csv_dir'))
print(f'Plotting {run_dir}')

df = pd.read_csv(os.path.join(run_dir, 'wave_pred.csv'))
leads = sorted(df.lead_s.unique())

fig, axes = plt.subplots(len(leads), 1, sharex=True, figsize=(10, 2.5 * len(leads)))
for ax, lead in zip(axes, leads):
    d = df[df.lead_s == lead].sort_values('t_target')
    ax.plot(d.t_target, d.buoy_z - d.buoy_z.mean(), 'k-', label='buoy z (demeaned)')
    ax.plot(d.t_target, d.pred_eta, 'r--', label='predicted wave η')
    ax.set_ylabel('z [m]')
    ax.set_title(f'lead time {lead:.0f} s')
    ax.grid(True)
axes[0].legend()
axes[-1].set_xlabel('sim time [s]')
plt.tight_layout()
plt.show()
