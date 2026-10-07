# Copyright 2022 Open Source Robotics Foundation, Inc. and Monterey Bay Aquarium Research Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Find the sim's current pblog run folder for the controller's output."""

import os
import threading
import time

import yaml


# sim_pblog writes its CSV at ~10 Hz while running; older than this means it isn't running
MAX_SIM_CSV_AGE_SEC = 30.0


class RunLog(object):
    """
    Put controller output next to the sim's pblog CSV for the same run.

    sim_pblog creates <pbloghome>/<YYYY-MM-DD.NNN>/ for each sim launch and points
    <pbloghome>/latest_csv_dir at it. We follow that link the first time there is something
    to write (so the sim is already running), then keep using that same folder (run_dir):
        controller_params.yaml  -- controller params (usable with --params-file) plus run info
    The controller writes its rolling_stats.csv there too.
    """

    def __init__(self, node, pbloghome='~/.pblogs', run_info=None):
        self.node = node
        self.pbloghome = os.path.expanduser(pbloghome)
        self.run_info = run_info or {}
        self.run_dir = None
        self.warned = False
        self.lock = threading.Lock()

    def ensure_started(self):
        """True once run_dir is set, finding it now if the sim's pblog is being written."""
        with self.lock:
            return self.run_dir is not None or self.start()

    def start(self):
        """Find the sim's current run folder and write controller_params.yaml there."""
        latest = os.path.join(self.pbloghome, 'latest_csv_dir')
        if not os.path.isdir(latest):
            return self.not_logging(f'No sim pblog folder at {latest}')

        # latest_csv_dir only moves when sim_pblog starts, so if sim_pblog isn't running it
        # still points at an old run. Only trust it if the sim's CSV is being written now,
        # so an old run's files are never overwritten.
        sim_csv = os.path.join(latest, 'latest')
        if not os.path.exists(sim_csv):
            return self.not_logging(f'No sim pblog CSV in {os.path.realpath(latest)}')
        age = time.time() - os.path.getmtime(sim_csv)
        if age > MAX_SIM_CSV_AGE_SEC:
            return self.not_logging(
                f'Sim pblog CSV in {os.path.realpath(latest)} was last written {age:.0f} s ago;'
                ' is sim_pblog running?')

        self.run_dir = os.path.realpath(latest)   # pin it; later sim launches move the link
        self.write_params()
        self.node.get_logger().info(f'Writing controller output to {self.run_dir}')
        return True

    def not_logging(self, reason):
        """Warn (once) that controller output isn't being written. Returns False."""
        if not self.warned:
            self.node.get_logger().warn(f'{reason}; not writing controller output')
            self.warned = True
        return False

    def write_params(self):
        """Save all of the node's ROS parameters, plus run_info, to controller_params.yaml."""
        params = {}
        for p in self.node.get_parameters(self.node.list_parameters([], 0).names):
            value = p.value
            if value is not None and not isinstance(value, (bool, int, float, str)):
                value = list(value)   # array.array / numpy -> plain list for YAML
            params[p.name] = value

        record = {self.node.get_name(): {'ros__parameters': params},
                  'run_info': self.run_info}
        with open(os.path.join(self.run_dir, 'controller_params.yaml'), 'w') as f:
            yaml.safe_dump(record, f, default_flow_style=None, sort_keys=False)
