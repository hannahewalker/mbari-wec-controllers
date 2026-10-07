#!/usr/bin/python3

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
import math
import os

from buoy_api import Interface
from .rollingstats import StatsRecorder, SeaStateEstimator, stamp_to_sec
import numpy as np
import rclpy
from scipy import interpolate

from .run_log import RunLog


class ControlPolicy(object):

    def __init__(self):
        self.Torque_constant = 0.438  # N-m/amps
        # Desired damping torque vs rpm relationship
        self.N_Spec = np.array([0.0, 300.0, 600.0, 1000.0, 1700.0, 4400.0, 6790.0])  # RPM
        self.Torque_Spec = np.array([0.0, 0.0, 0.8, 2.9, 5.6, 9.8, 16.6])  # N-m
        self.update_params()

    def update_params(self):
        """Update dependent variables after reading in params."""
        # Convert to Motor Winding Current vs RPM and generate interpolator for f(RPM) = I
        self.I_Spec = self.Torque_Spec / self.Torque_constant  # Amps
        self.windcurr_interp1d = interpolate.interp1d(self.N_Spec, self.I_Spec,
                                                      fill_value=self.I_Spec[-1],
                                                      bounds_error=False)

    def target(self, rpm, scale_factor, retract_factor):
        """Calculate target value from feedback inputs."""
        N = abs(rpm)
        current = self.windcurr_interp1d(N)

        # Apply damping gain
        current *= scale_factor

        # Hysteresis due to gravity / wave assist
        if rpm > 0.0:
            current *= -retract_factor

        return float(current)

    def __str__(self):
        return """ControlPolicy:
\tTorque_constant: {tc}
\tN_Spec: {nspec}
\tTorque_Spec: {tspec}
\tI_Spec: {ispec}""".format(tc=self.Torque_constant,
                            nspec=self.N_Spec,
                            tspec=self.Torque_Spec,
                            ispec=self.I_Spec)


class Controller(Interface):

    def __init__(self):
        super().__init__('controller')
        self.use_sim_time()   # run timers on /clock (output folder is still named by wall clock)
        self.policy = ControlPolicy()
        self.set_params()
        self.pc_factors_sent = False   # scale/retract sent to the power controller yet?

        # set packet rates from controllers here
        # controller defaults to publishing @ 10Hz
        # call these to set rate to 50Hz or provide argument for specific rate
        # self.set_pc_pack_rate(blocking=False)  # set PC publish rate to 50Hz
        self.sim_t = None   # set by latent_callback; wave heights are requested once it is

        # output goes into the sim's pblog run folder (same pbloghome as the sim launch arg)
        self.declare_parameter('pbloghome', '~/.pblogs')
        self.run_log = RunLog(self, self.get_parameter('pbloghome').value)

        self.create_timer(1.0, self.wave_height_timer)   # sea-state samples every 1 s

        self.stats = StatsRecorder(
            ['piston_pos', 'piston_vel', 'elec_power', 'mech_power', 'rpm',
             'end_stop_margin', 'force'],
            self.wec_window_s,
            extra_fields=['Hs', 'Tp', 'steepness'])


        # incident wave height at the buoy, 0-0.75 s ahead: 4 samples/s for the estimator
        self.sea_state_offsets = [0.0, 0.25, 0.5, 0.75]
        self.sea_state = SeaStateEstimator(window_s=self.sea_state_window_s, fs=4.0)

        self._prev_pos = None  # (t, x) for piston velocity
        self.stats_timer = self.create_timer(self.stats_update_period_s,
                                             self.update_stats)



    def ahrs_callback(self, data):
        """Provide feedback of '/ahrs_data' topic from XBowAHRS."""
        # ROLLING STATS: buoy vertical acceleration -> sea-state estimate
        #t = stamp_to_sec(data.header.stamp)
        #self.sea_state.add(t, vertical_accel(data.imu))
        pass


    def battery_callback(self, data):
        """Provide feedback of '/battery_data' topic from Battery Controller."""
        # Update class variables, get control policy target, send commands, etc.
        # target_value = self.policy.target(data)
        pass  # remove if there's anything to do above

    def spring_callback(self, data):
        """Provide feedback of '/spring_data' topic from Spring Controller."""
        # ROLLING STATS: piston position, velocity, end-stop margin, force
        t = stamp_to_sec(data.header.stamp)
        x = data.range_finder
        self.stats.add('piston_pos', t, x)
 
        if self._prev_pos is not None and t > self._prev_pos[0]:
            v = (x - self._prev_pos[1]) / (t - self._prev_pos[0])
            self.stats.add('piston_vel', t, v)
        self._prev_pos = (t, x)
 
        margin = min(x - self.stroke_min, self.stroke_max - x)
        self.stats.add('end_stop_margin', t, margin)
        self.stats.add('force', t, data.load_cell)
 
        # safety check on every sample, not once per second
        if margin < self.end_stop_warn_margin:
            self.get_logger().warn(f'Piston {margin:.3f} from end stop',
                                   throttle_duration_sec=1.0)


    def power_callback(self, data):
        """Provide feedback of '/power_data' topic from Power Controller."""
        # Update class variables, get control policy target, send commands, etc.
        # also set the factors on the power controller, so its default damping matches if our
        # commands stop (sim_pblog leaves its PC Scale / PC Retract columns empty).
        # Done on the first message rather than at startup: the sim may not be up yet then.
        if not self.pc_factors_sent:
            self.send_pc_scale_command(self.scale_factor, blocking=False)
            self.send_pc_retract_command(self.retract_factor, blocking=False)
            self.pc_factors_sent = True

        t = stamp_to_sec(data.header.stamp)
        self.stats.add('rpm',t,data.rpm)
        self.stats.add('mech_power',t,data.torque*data.rpm*2*math.pi / 60)
        self.stats.add('elec_power', t, data.voltage * data.bcurrent)

        wind_curr = self.policy.target(data.rpm, self.scale_factor, self.retract_factor)

        self.get_logger().info('WindingCurrent:' +
                               f' f({data.rpm:.02f}, {self.scale_factor:.02f},'
                               f' {self.retract_factor:.02f})' +
                               f' = {wind_curr:.02f}')

        self.send_pc_wind_curr_command(wind_curr, blocking=False)

    def trefoil_callback(self, data):
        """Provide feedback of '/trefoil_data' topic from Trefoil Controller."""
        # Update class variables, get control policy target, send commands, etc.
        # target_value = self.policy.target(data)
        pass  # remove if there's anything to do above

    def powerbuoy_callback(self, data):
        """Provide feedback of '/powerbuoy_data' topic -- Aggregated data from all topics."""
        # Update class variables, get control policy target, send commands, etc.
        # target_value = self.policy.target(data)
        pass  # remove if there's anything to do above

    def latent_callback(self, data):
        """Sim-only '/latent_data': note the sim time, so wave heights can be requested."""
        self.sim_t = stamp_to_sec(data.header.stamp)

    def update_stats(self):
        self.stats.update()
        ss = self.sea_state.estimate()
        self.get_logger().info(self.stats.format())
        self.get_logger().info(self.sea_state.format())

        # open the CSV in RunLog's run folder once data is flowing; RunLog only uses a folder
        # whose sim pblog is being written now, so an old run's file is never overwritten
        if (not self.stats.csv_is_open() and self.stats.latest_t is not None
                and self.run_log.ensure_started()):
            self.stats.open_csv(os.path.join(self.run_log.run_dir, 'rolling_stats.csv'))

        self.stats.write_row(extra=ss)

    def set_params(self):
        """Use ROS2 declare_parameter and get_parameter to set policy params."""
        self.declare_parameter('torque_constant', self.policy.Torque_constant)
        self.policy.Torque_constant = \
            self.get_parameter('torque_constant').get_parameter_value().double_value

        self.declare_parameter('n_spec', self.policy.N_Spec.tolist())
        self.policy.N_Spec = \
            np.array(self.get_parameter('n_spec').get_parameter_value().double_array_value)

        self.declare_parameter('torque_spec', self.policy.Torque_Spec.tolist())
        self.policy.Torque_Spec = \
            np.array(self.get_parameter('torque_spec').get_parameter_value().double_array_value)
        
        # recompute any dependent variables
        self.policy.update_params()
        self.get_logger().info(str(self.policy))

        # damping gain and extra retract-direction gain applied in power_callback.
        # The sim ignores its own scale/retract while we command winding current, so these
        # (not data.scale / data.retract) set the damping. Ranges match the sim's PC services.
        for name, default, lo, hi in (('scale_factor', 1.0, 0.5, 1.4),
                                      ('retract_factor', 0.6, 0.4, 1.0)):
            self.declare_parameter(name, default)
            value = float(self.get_parameter(name).value)
            if not lo <= value <= hi:
                raise ValueError(f'{name}={value} outside the valid range [{lo}, {hi}]')
            setattr(self, name, value)
        self.get_logger().info(f'scale_factor: {self.scale_factor}  '
                               f'retract_factor: {self.retract_factor}')


        # ROLLING STATS parameters (override in the params YAML)
        stats_params = {
            'wec_window_s': 25.0,          # WEC-state window (20-30 s)
            'sea_state_window_s': 300.0,   # Hs/Tp window
            'stats_update_period_s': 1.0,
            'stats_csv_path': '',          # empty = no CSV
            # piston end stops (m): range_finder runs 0 (fully retracted) to the
            # <stroke> in buoy_description mbari_wec/model.sdf.em
            'stroke_min': 0.0,
            'stroke_max': 2.03,
            'end_stop_warn_margin': 0.1,   # m
        }
        for name, default in stats_params.items():
            self.declare_parameter(name, default)
            setattr(self, name, self.get_parameter(name).value)



    def wave_height_timer(self):
        """Feed the sea-state estimator incident wave heights at the buoy."""
        if self.sim_t is None:
            return   # no latent data yet
        n = len(self.sea_state_offsets)
        resp = self.get_inc_wave_height(
            x=[0.0] * n, y=[0.0] * n, t=self.sea_state_offsets,
            use_buoy_origin=True,     # (0, 0) = at the buoy
            use_relative_time=True,   # t = seconds from now
            timeout=0.5)
        if resp is None or resp.result.value != resp.result.OK:
            return
        for h in resp.heights:
            t = stamp_to_sec(h.pose.header.stamp) + h.relative_time
            self.sea_state.add(t, h.pose.pose.position.z)


def main():
    rclpy.init()
    controller = Controller()
    try:
        controller.spin()   # note: buoy_api's spin() calls sys.exit() when done
    finally:
        controller.stats.close()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
