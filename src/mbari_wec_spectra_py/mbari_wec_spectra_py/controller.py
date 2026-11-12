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

from buoy_api import Interface
import rclpy
import matplotlib.pyplot as plt
import numpy as np
import threading 
from scipy import interpolate
from collections import deque
from scipy import signal
from scipy.ndimage import uniform_filter1d   

class ControlPolicy(object):
    """
    Simple Linear Damper Control Policy.

    Implements a simple linear damper controller for the piston in the WEC
    Power-Take-Off (PTO). Given motor RPM, outputs desired motor winding current (interpolated
    from RPM->Torque lookup table) to resist piston velocity. Configurable gains
    (scale/retract factor) are applied before output.
    """

    def __init__(self):
        # Define any parameter variables here
        self.Torque_constant = 0.438  # N-m/Amps
        # Desired damping Torque vs RPM relationship
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
        I = self.windcurr_interp1d(N)  # noqa: E741

        # Apply damping gain
        I *= scale_factor  # noqa: E741

        # Hysteresis due to gravity / wave assist
        if rpm > 0.0:
            I *= -retract_factor  # noqa: E741

        return float(I)

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
        super().__init__('wave_spectra')

        self.policy = ControlPolicy()
        self.set_params()
        
        self.fig = None
        self.plot_data = None
        self.input_Tp = 12.0      # what you launched the sim with, for reference

        # Transfer Function Hardcode
        '''
        self.H_f = np.array([0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 
                             0.085, 0.09, 0.095, 0.1, 0.105, 0.11, 0.115, 0.12, 0.125, 0.13, 
                             0.135, 0.14, 0.145, 0.15, 0.155, 0.16, 0.165, 0.17, 0.175, 0.18, 
                             0.185, 0.19, 0.195, 0.2, 0.205, 0.21, 0.215, 0.22, 0.225, 0.23, 
                             0.235, 0.24, 0.245, 0.25125, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 
                             0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 
                             0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5])
        
        self.H_mag = np.array([7.49854, 4.43054, 2.84895, 1.57879, 0.829563, 0.475693, 0.435668, 
                               0.43447, 0.443438, 0.479634, 0.519998, 0.571398, 0.621828, 0.692412, 
                               0.740265, 0.770822, 0.827115, 0.865987, 0.910515, 0.929934, 0.969199, 
                               1.01743, 1.02651, 1.08081, 1.08514, 1.14151, 1.15476, 1.1742, 1.21427, 
                               1.23057, 1.25479, 1.26963, 1.23838, 1.24369, 1.27284, 1.25135, 1.25615, 
                               1.28951, 1.2984, 1.27366, 1.29362, 1.22912, 1.22577, 1.21168, 1.22396, 
                               1.21105, 1.14128, 1.11837, 1.06231, 1.02961, 0.994335, 0.957677, 0.922544, 
                               0.9008, 0.869672, 0.822952, 0.778715, 0.734017, 0.710221, 0.672304, 
                               0.640843, 0.625323, 0.592792, 0.561378, 0.527751, 0.517878, 0.484027, 
                               0.469125, 0.444058])
        '''
        self.H_f = np.array([0.0400, 0.0450, 0.0500, 0.0550, 0.0600, 0.0650, 0.0700, 0.0750,
                             0.0800, 0.0850, 0.0900, 0.0950, 0.1000, 0.1050, 0.1100, 0.1150,
                             0.1200, 0.1250, 0.1300, 0.1350, 0.1400, 0.1450, 0.1500, 0.1550,
                             0.1600, 0.1650, 0.1700, 0.1750, 0.1800, 0.1850, 0.1900, 0.1950,
                             0.2000, 0.2050, 0.2100, 0.2150, 0.2200, 0.2250, 0.2300, 0.2350,
                             0.2400, 0.2450, 0.2500, 0.2550, 0.2600, 0.2650, 0.2700, 0.2750,
                             0.2800, 0.2850, 0.2900, 0.2950, 0.3000, 0.3050, 0.3100, 0.3150,
                             0.3200, 0.3250, 0.3300, 0.3350, 0.3400, 0.3450, 0.3500, 0.3550,
                             0.3600, 0.3650, 0.3700, 0.3750, 0.3800, 0.3850, 0.3900, 0.3950,
                             0.4000, 0.4050, 0.4100, 0.4150, 0.4200, 0.4250, 0.4300, 0.4350,
                             0.4400, 0.4450, 0.4500, 0.4550, 0.4600, 0.4650, 0.4700, 0.4750,
                             0.4800, 0.4850, 0.4900, 0.4950, 0.5000])
        self.H_mag  = 0.9*np.array([4.6274, 4.6274, 4.5756, 3.9168, 2.4184, 1.8757, 1.4871, 1.0248,
                             0.96545, 1.0709, 1.085, 0.99164, 0.7714, 0.86036, 0.93251, 1.0043,
                             1.0767, 1.1491, 1.2222, 1.2981, 1.3792, 1.4672, 1.5647, 1.674,
                             1.794, 1.9241, 2.0627, 2.2054, 2.3468, 2.4842, 2.616, 2.7401,
                             2.8585, 2.9739, 3.0916, 3.2172, 3.3553, 3.5086, 3.6731, 3.8438,
                             4.0118, 4.1673, 4.2982, 4.3978, 4.4641, 4.4973, 4.5023, 4.4851,
                             4.4524, 4.4095, 4.3606, 4.3085, 4.2496, 4.181, 4.0983, 3.9997,
                             3.8837, 3.7515, 3.601, 3.4366, 3.2645, 3.0893, 2.9153, 2.7456,
                             2.5824, 2.4274, 2.2794, 2.1389, 2.0065, 1.8811, 1.7618, 1.6486,
                             1.5404, 1.4355, 1.3338, 1.2355, 1.1416, 1.0537, 0.97305, 0.89955,
                             0.83339, 0.77478, 0.72369, 0.67994, 0.64335, 0.6136, 0.59031, 0.57267,
                             0.55973, 0.5507, 0.54468, 0.54081, 0.53848])
        
        # Rolling window info 
        #self.window_sec = 400.0
        self.t_buf = deque()
        self.seg_sec = 128.0
        self.window_sec = 3 * self.seg_sec + 10.0    # a little extra so all 5 segments fit
        self.x_buf = deque()
        self.lock = threading.Lock()
        self.create_timer(5.0, self.compute_spectrum)

        self.set_pc_pack_rate(blocking=False)  # set PC feedback publish rate to 50Hz

    def power_callback(self, data):
        """Provide feedback of '/power_data' topic from Power Controller."""
        # Update class variables, get control policy target, send commands, etc.
        wind_curr = self.policy.target(data.rpm, data.scale, data.retract)
        self.send_pc_wind_curr_command(wind_curr, blocking=False)

        # APPEND THE SAMPLE + TRIM (locked so compute_spectrum can't read mid-update)
        t = data.header.stamp.sec + data.header.stamp.nanosec*1e-9
        with self.lock:
            self.t_buf.append(t)
            self.x_buf.append(data.rpm)
            while self.t_buf[-1] - self.t_buf[0] > self.window_sec:
                self.t_buf.popleft()
                self.x_buf.popleft()

 
    def compute_spectrum(self):
        # take a consistent snapshot of the window
        with self.lock:
            t = np.array(self.t_buf)
            x = np.array(self.x_buf)

        # if the window isn't full yet, return
        if len(t) < 2:
            self.get_logger().info('no data yet')
            return

        span = t[-1] - t[0]
        self.get_logger().info(f'{len(t)} samples, span {span:.1f} s')

        if span < 0.95 * self.window_sec:
            return

        # resample to a uniform time grid
        t = np.array(self.t_buf)
        x = np.array(self.x_buf)

        fs = 10.0                                   # resample rate, Hz
        t_uniform = np.arange(t[0], t[-1], 1.0 / fs)
        x_uniform = np.interp(t_uniform, t, x)
        x0 = x_uniform - np.mean(x_uniform)

        # FFT / Welch -> frequencies and PSD
        dt = np.diff(t)
        #f, S = signal.welch(x_uniform, fs=fs, window='hann',
        #            nperseg=int(100 * fs), noverlap=int(50 * fs),
        #            detrend='linear', scaling='density')
        nseg = int(self.seg_sec * fs)
        f, S = signal.welch(x_uniform, fs=fs, window='hann',
                            nperseg=nseg, noverlap=3 * nseg // 4,
                            detrend='linear', scaling='density')
        var_time = np.var(x0)
        var_spec = np.trapezoid(S, f)
        peak_f = f[np.argmax(S[1:]) + 1]   # skip 0 Hz

        # restrict to the wave band (also avoids dividing by f = 0)
        band = (f >= 0.03) & (f <= 0.5)
        fb = f[band]

        S_vel = 0.0254**2 * S[band] / 60.0**2                   # rpm to m approx
        S_disp = S_vel / (2 * np.pi * fb)**2
        
        m0  = np.trapezoid(S_disp, fb)
        m_1 = np.trapezoid(S_disp / fb, fb)        # m_{-1}
        Te  = m_1 / m0
        Tp  = 1.0 / fb[np.argmax(S_disp)]


        # APPLY TRANSFER FUNCTION
        H = np.interp(fb, self.H_f, self.H_mag)
        H = np.maximum(H, 1e-3)
        S_eta = S_disp / H
        S_eta_s = uniform_filter1d(S_eta, size=3, mode='nearest')

        m0_eta = np.trapezoid(S_eta_s, fb)
        m_1_eta = np.trapezoid(S_eta_s / fb, fb)
        Te_eta = m_1_eta / m0_eta
        Tp_eta = 1.0 / fb[np.argmax(S_eta_s)]
        Hm0_eta = 4.0 * np.sqrt(m0_eta)

        #self.get_logger().info(f'Tp {Tp:.1f} s, Te {Te:.1f} s, m0 {m0:.1f}')
        #self.get_logger().info(f'Tp_eta {Tp_eta:.1f} s, Te_eta {Te_eta:.1f} s, m0_eta {m0_eta:.1f}')
        self.get_logger().info(f'rpm2dist: Tp {Tp:.1f} s, Te {Te:.1f} s, Hm0 {4*np.sqrt(m0):.3f} m')
        self.get_logger().info(f'wave est: Tp {Tp_eta:.1f} s, Te {Te_eta:.1f} s, Hm0 {Hm0_eta:.2f} m')
        
        #self.plot_data = (t_uniform, x_uniform - np.mean(x_uniform), fb, S_disp, Tp, Te)
        self.plot_data = (t_uniform, x_uniform - np.mean(x_uniform), fb, S_disp, Tp, Te,
                          S_eta, Tp_eta, Te_eta, Hm0_eta)
        pass

    def update_plot(self, t, x, f, S, Tp, Te, S_eta, Tp_eta, Te_eta, Hm0_eta):
        if self.fig is None:
            plt.ion()
            self.fig, (self.ax_t, self.ax_f, self.ax_w) = plt.subplots(3, 1, figsize=(8, 9))

            # time series
            self.line_t, = self.ax_t.plot([], [])
            self.ax_t.set_xlabel('time in window (s)')
            self.ax_t.set_ylabel('rpm (mean removed)')

            # displacement spectrum
            self.line_f, = self.ax_f.semilogy([], [], label='displacement PSD')
            self.vline_Tp = self.ax_f.axvline(0, color='r', ls='--', label='measured Tp')
            self.ax_f.axvline(1 / self.input_Tp, color='k', ls=':', label='input Tp')
            self.ax_f.set_ylabel('PSD (rev²/Hz)')
            self.ax_f.legend(loc='upper right')

            # transfer-function-corrected wave spectrum
            self.line_w, = self.ax_w.semilogy([], [], color='C2', label='wave est. (S_disp/|H|²)')
            self.vline_Tp_w = self.ax_w.axvline(0, color='r', ls='--', label='est. Tp')
            self.ax_w.axvline(1 / self.input_Tp, color='k', ls=':', label='input Tp')
            self.ax_w.set_xlabel('frequency (Hz)')
            self.ax_w.set_ylabel('PSD (m²/Hz)')
            self.ax_w.legend(loc='upper right')

            for ax in (self.ax_f, self.ax_w):
                ax.set_xlim(0, 0.5)
            self.fig.tight_layout()

        self.line_t.set_data(t - t[0], x)
        self.line_f.set_data(f, S)
        self.line_w.set_data(f, S_eta)
        self.vline_Tp.set_xdata([1 / Tp, 1 / Tp])
        self.vline_Tp_w.set_xdata([1 / Tp_eta, 1 / Tp_eta])

        for ax in (self.ax_t, self.ax_f, self.ax_w):
            ax.relim()
            ax.autoscale_view()
        for ax in (self.ax_f, self.ax_w):
            ax.set_xlim(0, 0.5)

        self.ax_t.set_title(f'window ending t = {t[-1]:.1f} s')
        self.ax_f.set_title(f'displacement: Tp {Tp:.1f} s, Te {Te:.1f} s', fontsize=10)
        self.ax_w.set_title(f'wave est.: Hm0 {Hm0_eta:.2f} m, Tp {Tp_eta:.1f} s, Te {Te_eta:.1f} s',
                            fontsize=10)
        self.fig.canvas.draw_idle()
    # def update_plot(self, t, x, f, S, Tp, Te):
    #     if self.fig is None:
    #         plt.ion()
    #         self.fig, (self.ax_t, self.ax_f) = plt.subplots(2, 1, figsize=(8, 6))

    #         self.line_t, = self.ax_t.plot([], [])
    #         self.ax_t.set_xlabel('time in window (s)')
    #         self.ax_t.set_ylabel('rpm (mean removed)')

    #         self.line_f, = self.ax_f.semilogy([], [], label='displacement PSD')
    #         self.vline_Tp = self.ax_f.axvline(0, color='r', ls='--', label='measured Tp')
    #         self.ax_f.axvline(1 / self.input_Tp, color='k', ls=':', label='input Tp')
    #         self.ax_f.set_xlabel('frequency (Hz)')
    #         self.ax_f.set_ylabel('PSD (rev²/Hz)')
    #         self.ax_f.set_xlim(0, 0.5)
    #         self.ax_f.legend(loc='upper right')
    #         self.fig.tight_layout()

    #     self.line_t.set_data(t - t[0], x)
    #     self.line_f.set_data(f, S)
    #     self.vline_Tp.set_xdata([1 / Tp, 1 / Tp])

    #     for ax in (self.ax_t, self.ax_f):
    #         ax.relim()
    #         ax.autoscale_view()
    #     self.ax_f.set_xlim(0, 0.5)

    #     self.ax_t.set_title(f'window ending t = {t[-1]:.1f} s   |   Tp {Tp:.1f} s, Te {Te:.1f} s')
    #     self.fig.canvas.draw_idle()
    #     plt.pause(0.001)

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


# def main():
#    rclpy.init()
#    controller = Controller()
#    controller.spin()
#    rclpy.shutdown()
def main():
    import threading

    rclpy.init()
    controller = Controller()

    # ROS runs in a background thread...
    ros_thread = threading.Thread(target=controller.spin, daemon=True)
    ros_thread.start()

    # ...and the main thread only draws
    try:
        while rclpy.ok():
            if controller.plot_data is not None:
                data = controller.plot_data
                controller.plot_data = None
                controller.update_plot(*data)
            plt.pause(0.2)
        print('rclpy no longer ok; ROS thread alive:', ros_thread.is_alive())
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
