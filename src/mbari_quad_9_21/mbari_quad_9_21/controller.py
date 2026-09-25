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

# Test change: verifying git commit/push works.

from buoy_api import Interface
# import numpy as np
import rclpy
# from scipy import interpolate


class ControlPolicy(object):

    def __init__(self):
        # Define any parameter variables here
        self.Torque_constant = 0.438  # N-m/amps
        self.Quad_Coeff = 5.0e-7
        self.Torque_Max = 16.6

        self.update_params()

    def update_params(self):
        """Update dependent variables after reading in params."""
        # Convert to Motor Winding Current vs RPM and generate interpolator for f(RPM) = I
        self.I_max = self.Torque_Max / self.Torque_constant

    def target(self, rpm, scale_factor, retract_factor):
        """Calculate target value from feedback inputs."""
        N = abs(rpm)
        torque = min(self.Quad_Coeff * N * N, self.Torque_Max)  # N-m saturated
        wind_curr = scale_factor * torque / self.Torque_constant

        # Hysteresis due to gravity / wave assist
        if rpm > 0.0:
            wind_curr *= -retract_factor

        return float(wind_curr)

    def __str__(self):
        return """ControlPolicy:
\tTorque_constant: {tc}
\tQuad_Coeff: {k}
\tTorque_Max: {tmax}
\tI_max: {imax}""".format(tc=self.Torque_constant, k=self.Quad_Coeff,
                          tmax=self.Torque_Max, imax=self.I_max)


class Controller(Interface):

    def __init__(self):
        super().__init__('controller')

        self.policy = ControlPolicy()
        self.set_params()

        # set packet rates from controllers here
        # controller defaults to publishing @ 10Hz
        # call these to set rate to 50Hz or provide argument for specific rate
        self.set_pc_pack_rate(blocking=False)  # set PC publish rate to 50Hz
        # self.set_sc_pack_rate(blocking=False)  # set SC publish rate to 50Hz

        # Use this to set node clock to use sim time from /clock (from gazebo sim time)
        # Access node clock via self.get_clock() or other various
        # time-related functions of rclpy.Node
        # self.use_sim_time()

    # To subscribe to any topic, simply define the specific callback, e.g. power_callback
    # def power_callback(self, data):
    #     """Enables feedback of '/power_data' topic from Power Controller"""
    #     # get target value from control policy
    #     target_value = self.policy.target(data.rpm, data.scale, data.retract)

    #     # send a command, e.g. winding current
    #     self.send_pc_wind_curr_command(target_value, blocking=False)

    # Available commands to send within any callback:
    # self.send_pump_command(duration_mins, blocking=False)
    # self.send_valve_command(duration_sec, blocking=False)
    # self.send_pc_wind_curr_command(wind_curr_amps, blocking=False)
    # self.send_pc_bias_curr_command(bias_curr_amps, blocking=False)
    # self.send_pc_scale_command(scale_factor, blocking=False)
    # self.send_pc_retract_command(retract_factor, blocking=False)

    # Delete any unused callback

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
        """Provide feedback of '/latent_data' topic -- SIM ONLY values (e.g. losses, wave data)."""
        # Potentially evaluate controller performance against sim only (latent data NOT AVAILABLE
        # in physical buoy)
        pass  # remove if there's anything to do above

    def set_params(self):
        """Use ROS2 declare_parameter and get_parameter to set policy params."""
        self.declare_parameter('torque_constant', self.policy.Torque_constant)
        self.policy.Torque_constant = \
            self.get_parameter(
                'torque_constant').get_parameter_value().double_value

        self.declare_parameter('quad_coeff', self.policy.Quad_Coeff)
        self.policy.Quad_Coeff = \
            self.get_parameter('quad_coeff').get_parameter_value().double_value

        self.declare_parameter('torque_max', self.policy.Torque_Max)
        self.policy.Torque_Max = \
            self.get_parameter('torque_max').get_parameter_value().double_value

        # recompute any dependent variables
        self.policy.update_params()
        self.get_logger().info(str(self.policy))


def main():
    rclpy.init()
    controller = Controller()
    controller.spin()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
