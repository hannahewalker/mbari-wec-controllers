"""Time-based rolling window statistics for MBARI WEC controller signals.
 
RollingStats   - one signal, one time window (e.g. the last 20 s of rpm)
StatsRecorder  - a named set of RollingStats, with a periodic update(),
                 log formatting and optional CSV output
 
Every sample is stored as it arrives, at whatever rate its topic publishes.
The window is defined only by the sim time vector: a sample stays in while
its time is within window_s of the newest time. Stats are computed from
every sample in the window each time update() is called (e.g. once per second).
 
In the controller, pass the message header stamp (stamp_to_sec(data.header.stamp))
so windows follow simulated time.
"""

from collections import deque 
import csv
import math 
import numpy as np

def stamp_to_sec(stamp):
    """Convert a builtin_interfaces/Time (msg.header.stamp) to float seconds."""
    return stamp.sec + stamp.nanosec / 1e9


class RollingStats:
    """Rolling window over one scalar signal, trimmed by time rather than sample count."""

    def __init__(self, window_s):
        if window_s <= 0:
            raise ValueError("Window size must be positive")
        self.window_s = window_s
        self.clear()

    def clear(self):
        self._t = deque()  # sample time (s)
        self._x = deque()  # sample values

    def add(self, t, x):
        x = float(x)
        if not math.isfinite(x):
            return # skip NaN/inf 
        if self._t and t < self._t[-1]:
            self.clear() 
        self._t.append(t)
        self._x.append(x)
        self.trim(t)

    def trim(self, t_now):
        """Remove samples that are outside the current window."""
        cutoff = t_now - self.window_s
        while self._t and self._t[0] < cutoff:
            self._t.popleft()
            self._x.popleft()

    def __len__(self):
        return len(self._x)

    def last_time(self):
        return self._t[-1] if self._t else None

    def span(self):
        """Return the time span of the current window."""
        return self._t[-1] - self._t[0] if self._t else 0.0

    def is_full(self, fraction=0.95):
        """True once the window holds most of window_s data
        Right after startup the window only covers a few seconds, so a 
        controller should not trust stats until this is true
        """
        return self.span() >= fraction * self.window_s

    def stats(self):
        """Mean, std. min, max over every sample currently in the window"""
        n = len(self._x)
        if n == 0:
            return None
        mean = sum(self._x) / n
        var = sum((v - mean) ** 2 for v in self._x) / n
        return {
            'n': n,
            'mean': mean,
            'std': math.sqrt(var),
            'min': min(self._x),
            'max': max(self._x),
            'span': self.span(),
        }


 
class StatsRecorder:
    """ A set of named rolling windows sharing one window length.

    add()       - call from ech topic callback with every sample 
    update()    - call from a timer (e.g. 1 Hz). Aligns all windows to the newest sample time,
                  then computes stats from every sample in each window. 
    latest      - the snapshot from the last update(); format() and write_row() use it.
    """

    FIELDS = ('mean', 'std', 'min', 'max', 'n')

    def __init__(self, signals, window_s, csv_path=None, extra_fields=()):
        self.signals = list(signals)
        self.extra_fields = list(extra_fields)
        self.windows = {name: RollingStats(window_s) for name in self.signals}
        self.latest_t = None
        self.latest = {}
        self.latest_update_t = None
        self._csv_file = None
        self._writer = None
        if csv_path:
            self.open_csv(csv_path)

    def open_csv(self, csv_path):
        # line-buffered so rows reach disk even if the node is killed with Ctrl-C
        self._csv_file = open(csv_path, 'w', newline='', buffering=1)
        self._writer = csv.writer(self._csv_file)
        header = (['t'] + [f'{s}_{f}' for s in self.signals for f in self.FIELDS]
                  + self.extra_fields)
        self._writer.writerow(header)

    def csv_is_open(self):
        return self._writer is not None

    def add(self, name, t, x):
        w = self.windows[name]
        last = w.last_time()
        if last is not None and t < last:
            # this signal's time went backwards: the sim restarted, so reset everything
            for other in self.windows.values():
                other.clear()
            self.latest_t = None
        w.add(t,x)
        if self.latest_t is None or t > self.latest_t:
            self.latest_t = t

    def __getitem__(self,name):
        return self.windows[name]

    def update(self):
        """Algin all windows to the newest sample time and snapshot their stats."""
        if self.latest_t is None:
            return self.latest
        for w in self.windows.values():
            w.trim(self.latest_t)
        self.latest = {name: w.stats() for name, w in self.windows.items()}
        self.latest_update_t = self.latest_t
        return self.latest

    def format(self):
        parts = []
        for name in self.signals:
            s = self.latest.get(name)
            if s:
                parts.append(
                    f"{name}: mean={s['mean']:.3g} std={s['std']:.3g} "
                    f"min={s['min']:.3g} max={s['max']:.3g}"
                    f"(n={s['n']}, {s['span']:.1f}s)")
        return ' | '.join(parts) if parts else 'no data yet'
    
    def write_row(self, extra=None):
        """Write the latest snapshot. extra: dict of values for extra_fields (or None)."""
        if self._writer is None or self.latest_update_t is None:
            return
        row = [f'{self.latest_update_t:.3f}']
        for name in self.signals:
            s = self.latest.get(name)
            row += [s[f] if s else '' for f in self.FIELDS]
        extra = extra or {}
        row += [extra.get(f, '') for f in self.extra_fields]
        self._writer.writerow(row)

    def close(self):
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._writer = None


class SeaStateEstimator:
    """Hs, Tp and steepness from a rolling window of wave elevation (m).

    add() each elevation sample with its sim time; estimate() from the 1 Hz timer.
    Each estimate uses every sample in the window: interpolate onto a uniform
    grid at fs, compute a Welch spectrum, then within [f_min, f_max]:
        Hs = 4*sqrt(m0),  Tp = 1/f_peak,  steepness = 2*pi*Hs / (g*Tp^2)
    """

    G = 9.81

    def __init__(self, window_s=300.0, fs=4.0, segment_s=100.0,
                 f_min=0.04, f_max=0.5):
        self.window = RollingStats(window_s)
        self.fs = fs
        self.segment_s = segment_s
        self.f_min = f_min
        self.f_max = f_max
        self.latest = None

    def add(self, t, eta):
        last = self.window.last_time()
        if last is not None and t <= last:
            if t < last - 10.0:
                self.window.clear()   # large jump back: sim restarted
            else:
                return                # duplicate or out-of-order sample: skip it
        self.window.add(t, eta)

    def is_full(self, fraction=0.95):
        return self.window.is_full(fraction)

    def estimate(self):
        """Returns {'Hs', 'Tp', 'steepness'}, or None until 2 segments of data exist."""
        t = np.array(self.window._t)
        eta = np.array(self.window._x)
        if len(t) < 4 or t[-1] - t[0] < 2 * self.segment_s:
            self.latest = None
            return None

        # uniform grid, mean removed
        tu = np.arange(t[0], t[-1], 1.0 / self.fs)
        eu = np.interp(tu, t, eta)
        eu -= eu.mean()

        # Welch spectrum: Hann window, 50% overlap, one-sided
        nseg = int(round(self.segment_s * self.fs))
        step = nseg // 2
        win = np.hanning(nseg)
        scale = 1.0 / (self.fs * np.sum(win ** 2))
        psd = np.zeros(nseg // 2 + 1)
        count = 0
        for start in range(0, len(eu) - nseg + 1, step):
            seg = eu[start:start + nseg]
            seg = (seg - seg.mean()) * win
            psd += np.abs(np.fft.rfft(seg)) ** 2 * scale
            count += 1
        psd /= count
        psd[1:-1] *= 2.0
        f = np.fft.rfftfreq(nseg, 1.0 / self.fs)

        band = (f >= self.f_min) & (f <= self.f_max)
        fb = f[band]
        s_eta = psd[band]
        m0 = np.sum(s_eta) * (f[1] - f[0])

        hs = 4.0 * math.sqrt(m0)
        tp = float(1.0 / fb[np.argmax(s_eta)])
        self.latest = {'Hs': hs, 'Tp': tp,
                       'steepness': 2 * math.pi * hs / (self.G * tp ** 2)}
        return self.latest

    def format(self):
        s = self.latest
        if not s:
            return f'sea state: waiting for data ({self.window.span():.0f}s)'
        return f"Hs={s['Hs']:.2f} m  Tp={s['Tp']:.1f} s  steepness={s['steepness']:.4f}"
    

if __name__ == '__main__':
    # Quick offline self-test: python3 rolling_stats.py
    # 'fast' at 50 Hz, 'slow' at 10 Hz, stats updated once per simulated second
    import random
    rec = StatsRecorder(['fast', 'slow'], window_s=20.0)
    dt = 0.02
    for i in range(1, 3001):  # 60 s at 50 Hz
        t = i * dt
        rec.add('fast', t, math.sin(2 * math.pi * t / 10.0))
        if i % 5 == 0:
            rec.add('slow', t, random.gauss(5.0, 2.0))
        if i % 50 == 0:
            rec.update()
    print(rec.format())
    print('fast window full:', rec['fast'].is_full())

