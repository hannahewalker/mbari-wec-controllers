"""Run one sim per (wave, scale_factor, retract_factor) combination with the rolling_stats controller.

usage: sweep_damping.py sweep.yaml [--out DIR] [--no-rosbag] [--dry-run]

sweep.yaml is a normal mbari_wec_batch sim_params_yaml, plus arrays of damping factors:

    duration: 600
    seed: 42
    physics_rtf: 11
    enable_gui: False
    IncidentWaveSpectrumType:
     - MonoChromatic:
         A: [0.5, 1.0]
         T: [8.0, 12.0]
    scale_factor:   [0.8, 1.0, 1.2]   # 0.5 - 1.4
    retract_factor: [0.6, 0.8]        # 0.4 - 1.0

Waves are listed the same way as for mbari_wec_batch: A/T (MonoChromatic) and Hs/Tp
(Bretschneider) are paired by position, so the example has 2 waves, 3 x 2 damping
combinations, and 2 x 6 = 12 sim runs. Keep the other sim params single-valued.

Each run starts the controller first (it takes a while to check for services), then one sim
via mbari_wec_batch.launch.py. Once the sim is done, its files are moved out of the batch
runner's nested folders so each run is one folder:

    <out>/sweep_<time>/
        sweep.yaml
        sweep_runs.csv                         -- run, factors, wave, status (ok/FAILED), folder
        run00_s0.80_r0.60_A0.5_T8/
            <sim pblog>.csv, latest
            rolling_stats.csv, wave_pred.csv, controller_params.yaml
            controller.log, sim.log, sim.yaml, batch_runs.log
            rosbag2/                           -- unless --no-rosbag
"""

import argparse
import csv
import glob
import itertools
import os
import shutil
import signal
import subprocess
import sys
import time

import yaml

RANGES = {'scale_factor': (0.5, 1.4), 'retract_factor': (0.4, 1.0)}
DEFAULTS = {'scale_factor': 1.0, 'retract_factor': 0.6}
# wave types whose params list several waves, paired by position (first key sets the count)
PAIRED = {'MonoChromatic': ('A', 'T'), 'Bretschneider': ('Hs', 'Tp')}
SHORT = {'MonoChromatic': 'mono', 'Bretschneider': 'bret'}
READY_LINE = 'retract_factor:'   # controller logs this once its parameters are set
READY_TIMEOUT_S = 120.0


def as_list(value):
    return list(value) if isinstance(value, (list, tuple)) else [value]


def split_waves(spectrum_types):
    """IncidentWaveSpectrumType list -> [(label, one-wave IncidentWaveSpectrumType list)]."""
    if spectrum_types is None:
        return [('default', None)]   # leave it out: sim's default waves
    waves = []
    for entry in spectrum_types:
        (wave_type, params), = entry.items() if isinstance(entry, dict) else ((entry, None),)
        keys = PAIRED.get(wave_type)
        if not keys or not params or keys[0] not in params:
            # Custom or default params: one wave as given
            waves.append((SHORT.get(wave_type, wave_type), [entry]))
            continue
        n = len(as_list(params[keys[0]]))
        if any(len(as_list(params[k])) != n for k in keys if k in params):
            sys.exit(f'{wave_type}: {" and ".join(keys)} must have the same number of values')
        for i in range(n):
            # paired keys (and any other per-wave list, e.g. n_phases) take element i
            one = {k: [as_list(v)[i]] if k in keys or (isinstance(v, list) and len(v) == n)
                   else v for k, v in params.items()}
            label = '_'.join(f'{k}{one[k][0]:g}' for k in keys if k in one)
            waves.append((f'{SHORT.get(wave_type, wave_type)}_{label}', [{wave_type: one}]))
    return waves


def stop(proc, name, timeout=20.0):
    """Ctrl-C a launch process group (so the controller closes its CSVs), then kill it."""
    if proc is None or proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGINT)
    try:
        proc.wait(timeout)
    except subprocess.TimeoutExpired:
        print(f'  {name} did not stop on SIGINT; killing it')
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def wait_until_ready(proc, log_path):
    t_end = time.time() + READY_TIMEOUT_S
    while time.time() < t_end:
        if proc.poll() is not None:
            return False
        with open(log_path) as f:
            if READY_LINE in f.read():
                return True
        time.sleep(0.5)
    return False


def run_one(run_dir, sim_yaml, scale, retract, rosbag):
    """Controller first, then the sim, in run_dir/_batch; returns the sim's return code."""
    work = os.path.join(run_dir, '_batch')   # batch runner's nested output goes here
    os.makedirs(work)
    pbloghome = os.path.join(work, 'latest_batch_results')
    ctl = sim = None
    code = None
    try:
        with open(os.path.join(run_dir, 'controller.log'), 'w') as ctl_log:
            ctl = subprocess.Popen(
                ['ros2', 'launch', 'rolling_stats', 'controller.launch.py',
                 f'scale_factor:={scale}', f'retract_factor:={retract}',
                 f'pbloghome:={pbloghome}'],
                cwd=work, stdout=ctl_log, stderr=subprocess.STDOUT,
                start_new_session=True)
        if not wait_until_ready(ctl, os.path.join(run_dir, 'controller.log')):
            print('  controller did not start; see controller.log')
            return None

        with open(os.path.join(run_dir, 'sim.log'), 'w') as sim_log:
            sim = subprocess.Popen(
                ['ros2', 'launch', 'buoy_gazebo', 'mbari_wec_batch.launch.py',
                 f'sim_params_yaml:={sim_yaml}', f'rosbag2:={str(rosbag).lower()}'],
                cwd=work, stdout=sim_log, stderr=subprocess.STDOUT,
                start_new_session=True)
            code = sim.wait()
    finally:
        stop(sim, 'sim')
        stop(ctl, 'controller')   # controller files are closed after this
        flatten(work, run_dir)
    return code


def flatten(work, run_dir):
    """Move the pblog files (and rosbag2, batch_runs.log) up into run_dir, drop the rest."""
    for results in glob.glob(os.path.join(work, 'batch_results_*', 'results_run_*')):
        pblog = os.path.join(results, 'pblog')
        for name in os.listdir(pblog) if os.path.isdir(pblog) else []:
            shutil.move(os.path.join(pblog, name), run_dir)
        rosbag = os.path.join(results, 'rosbag2')
        if os.path.isdir(rosbag) and os.listdir(rosbag):
            shutil.move(rosbag, os.path.join(run_dir, 'rosbag2'))
    for log in glob.glob(os.path.join(work, 'batch_results_*', 'batch_runs.log')):
        shutil.move(log, run_dir)
    shutil.rmtree(work)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('sweep_yaml')
    parser.add_argument('--out', default='~/.pblogs/sweeps',
                        help='parent folder for the sweep (default: %(default)s)')
    parser.add_argument('--no-rosbag', action='store_true',
                        help="don't record a rosbag for each run (they are large)")
    parser.add_argument('--dry-run', action='store_true',
                        help='write the per-run sim yamls and list the runs, but run nothing')
    args = parser.parse_args()

    # both workspaces must be sourced, or every run fails at launch
    for pkg, ws in (('buoy_gazebo', '~/mbari_wec_ws'), ('rolling_stats', '~/controller_ws')):
        if subprocess.run(['ros2', 'pkg', 'prefix', pkg], capture_output=True).returncode:
            sys.exit(f"ROS can't find the {pkg} package. Run this first:\n"
                     f'    source {ws}/install/setup.bash')

    with open(args.sweep_yaml) as f:
        sim_params = yaml.safe_load(f)

    factors = {}
    for name, (lo, hi) in RANGES.items():
        values = [float(v) for v in as_list(sim_params.pop(name, DEFAULTS[name]))]
        bad = [v for v in values if not lo <= v <= hi]
        if bad:
            sys.exit(f'{name} values {bad} outside the valid range [{lo}, {hi}]')
        factors[name] = values
    if sim_params.pop('controller', None) is not None:
        print('Ignoring the controller entry in the yaml: this script runs rolling_stats')
    waves = split_waves(sim_params.pop('IncidentWaveSpectrumType', None))
    multi = [k for k, v in sim_params.items() if isinstance(v, list) and len(v) > 1]
    if multi:
        sys.exit(f'{multi} have several values; only waves and damping factors can be swept')

    runs = list(itertools.product(waves, factors['scale_factor'], factors['retract_factor']))
    # absolute: the sim and controller run from inside each run folder
    sweep_dir = os.path.join(os.path.abspath(os.path.expanduser(args.out)),
                             time.strftime('sweep_%Y%m%d_%H%M%S'))
    os.makedirs(sweep_dir)
    shutil.copy(args.sweep_yaml, os.path.join(sweep_dir, 'sweep.yaml'))
    print(f'{len(waves)} waves x {len(runs) // len(waves)} damping combos'
          f' = {len(runs)} runs -> {sweep_dir}')

    index_path = os.path.join(sweep_dir, 'sweep_runs.csv')
    with open(index_path, 'w', newline='', buffering=1) as index_file:
        index = csv.writer(index_file)
        index.writerow(['run', 'scale_factor', 'retract_factor', 'wave', 'status', 'folder'])
        for i, ((wave, wave_params), scale, retract) in enumerate(runs):
            name = f'run{i:02d}_s{scale:.2f}_r{retract:.2f}_{wave}'
            run_dir = os.path.join(sweep_dir, name)
            os.makedirs(run_dir)
            # the sim's own damping gets the same scale, used before our first command
            one_run = {**sim_params, 'scale_factor': [scale]}
            if wave_params is not None:
                one_run['IncidentWaveSpectrumType'] = wave_params
            sim_yaml = os.path.join(run_dir, 'sim.yaml')
            with open(sim_yaml, 'w') as f:
                yaml.safe_dump(one_run, f, sort_keys=False)

            print(f'[{i + 1}/{len(runs)}] {name}')
            if args.dry_run:
                continue
            t0 = time.time()
            run_one(run_dir, sim_yaml, scale, retract, not args.no_rosbag)
            # ros2 launch exits 0 even when the sim dies, so check for the output instead
            missing = [f for f in ('latest', 'rolling_stats.csv')   # latest -> sim pblog CSV
                       if not os.path.exists(os.path.join(run_dir, f))]
            status = 'ok' if not missing else 'FAILED'
            print(f'  {status} after {time.time() - t0:.0f} s'
                  + (f': no {" or ".join(missing)}; see sim.log / controller.log'
                     if missing else ''))
            index.writerow([i, scale, retract, wave, status, name])

    print(f'Done. Index of runs: {index_path}')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:   # run_one has already stopped the sim and controller
        sys.exit('\nStopped. Finished runs are listed in sweep_runs.csv')
