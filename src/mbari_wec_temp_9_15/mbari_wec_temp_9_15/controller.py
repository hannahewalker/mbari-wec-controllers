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
import threading

from buoy_api import Interface
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

        # set packet rates from controllers here
        # controller defaults to publishing @ 10Hz
        # call these to set rate to 50Hz or provide argument for specific rate
        # self.set_pc_pack_rate(blocking=False)  # set PC publish rate to 50Hz
        # --- wave prediction logging ---
        self.lead_times = [0.0, 2.0, 5.0, 10.0]   # seconds into the future
        self.sim_t = None
        self.buoy_z = None
        self.pending = []   # (target_time, lead, predicted_eta)
        self.pending_lock = threading.Lock()   # timer and latent_callback run concurrently

        # output goes into the sim's pblog run folder (same pbloghome as the sim launch arg)
        self.declare_parameter('pbloghome', '~/.pblogs')
        self.run_log = RunLog(self, self.get_parameter('pbloghome').value,
                              run_info={'lead_times_s': self.lead_times})

        self.create_timer(1.0, self.predict_timer)   # ask for a new prediction every 1 s

    def ahrs_callback(self, data):
        """Provide feedback of '/ahrs_data' topic from XBowAHRS."""
        # Update class variables, get control policy target, send commands, etc.
        # target_value = self.policy.target(data)
        pass  # remove if there's anything to do above

    def battery_callback(self, data):
        """Provide feedback of '/battery_data' topic from Battery Controller."""
        # Update class variables, get control policy target, send commands, etc.
        # target_value = self.policy.target(data)
        pass  # remove if there's anything to do above

    def spring_callback(self, data):
        """Provide feedback of '/spring_data' topic from Spring Controller."""
        # Update class variables, get control policy target, send commands, etc.
        # target_value = self.policy.target(data)
        pass  # remove if there's anything to do above

    def power_callback(self, data):
        """Provide feedback of '/power_data' topic from Power Controller."""
        # Update class variables, get control policy target, send commands, etc.
        wind_curr = self.policy.target(data.rpm, data.scale, data.retract)

        self.get_logger().info('WindingCurrent:' +
                               f' f({data.rpm:.02f}, {data.scale:.02f}, {data.retract:.02f})' +
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
        # current sim time and buoy vertical position
        self.sim_t = data.header.stamp.sec + data.header.stamp.nanosec * 1e-9
        self.buoy_z = data.wave_body.pose.position.z

        # any predictions whose target time has now arrived?
        with self.pending_lock:
            still_waiting = []
            for t_target, lead, eta in self.pending:
                if self.sim_t >= t_target:
                    print(f't={t_target:8.2f}  lead={lead:5.1f}s  '
                          f'pred={eta:7.3f} m  buoy z={self.buoy_z:7.3f} m')
                    self.run_log.log_prediction(t_target, lead, eta, self.buoy_z, self.sim_t)
                else:
                    still_waiting.append((t_target, lead, eta))
            self.pending = still_waiting

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

    def predict_timer(self):
        """Request predicted wave heights at the buoy."""
        if self.sim_t is None:
            return   # no latent data yet
        n = len(self.lead_times)
        resp = self.get_inc_wave_height(
            x=[0.0] * n, y=[0.0] * n, t=self.lead_times,
            use_buoy_origin=True,     # (0, 0) = at the buoy
            use_relative_time=True,   # t = seconds from now
            timeout=0.5)
        if resp is None or resp.result.value != resp.result.OK:
            return
        with self.pending_lock:
            for h in resp.heights:
                stamp = h.pose.header.stamp.sec + h.pose.header.stamp.nanosec * 1e-9
                t_target = stamp + h.relative_time
                lead = round(h.relative_time, 3)   # sim echoes it back with float noise
                self.pending.append((t_target, lead, h.pose.pose.position.z))


def main():
    rclpy.init()
    controller = Controller()
    try:
        controller.spin()   # note: buoy_api's spin() calls sys.exit() when done
    finally:
        controller.run_log.close()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
