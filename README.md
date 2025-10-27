# audio_noise_lab.py
# Desktop audio-noise-reduction lab with tabs (like your ECG app)
# Uses only: numpy, scipy, matplotlib, tkinter

import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# -------------------------- App State --------------------------
class State:
def __init__(self):
# numeric params (no tk vars in state)
self.fs = 8000
self.duration = 3.0
self.t = np.linspace(0, self.duration, int(self.fs*self.duration), endpoint=False)

self.clean = None
self.noisy = None
self.filtered = None

# noise params
self.white_amp = 0.30
self.hum_amp = 0.20
self.hum_freq = 50.0

# filter params
self.notch_on = True
self.notch_f0 = 50.0
self.notch_Q = 30.0

self.hp_on = False
self.hp_cut = 80.0
self.hp_order = 4

self.lp_on = True
self.lp_cut = 1500.0
self.lp_order = 6

S = State()

# -------------------------- Utilities --------------------------
def new_time():
S.t = np.linspace(0, S.duration, int(S.fs*S.duration), endpoint=False)

def to_mono(x):
return x if x.ndim == 1 else np.mean(x, axis=1)

def normalize(x):
m = np.max(np.abs(x)) + 1e-12
return x / m

def plot_time(ax, t, x, title):
ax.clear()
ax.plot(t, x, linewidth=1.0)
ax.set_title(title)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amp")
ax.grid(alpha=0.3)

def plot_two(ax1, ax2, t, x1, x2, title1, title2):
ax1.clear(); ax2.clear()
ax1.plot(t, x1, linewidth=1.0); ax1.set_title(title1); ax1.grid(alpha=0.3)
ax2.plot(t, x2, linewidth=1.0); ax2.set_title(title2); ax2.grid(alpha=0.3)
ax2.set_xlabel("Time (s)")
ax1.set_ylabel("Amp"); ax2.set_ylabel("Amp")

def plot_psd(ax, fs, x_noisy, x_filt):
ax.clear()
if x_noisy is not None and len(x_noisy) > 8:
nseg = max(256, min(2048, len(x_noisy)//4))
f, Pn = signal.welch(x_noisy, fs, nperseg=nseg)
ax.semilogy(f, Pn, label="Noisy")
if x_filt is not None and len(x_filt) > 8:
nseg = max(256, min(2048, len(x_filt)//4))
f, Pf = signal.welch(x_filt, fs, nperseg=nseg)
ax.semilogy(f, Pf, label="Filtered")
ax.set_title("Power Spectral Density (Welch)")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("PSD")
ax.grid(alpha=0.3)
ax.legend(loc="best")

def refresh(canvas):
canvas.draw_idle()

# -------------------------- Core Ops --------------------------
def generate_tone(f1=440.0, f2=880.0, a2=0.4):
new_time()
x = np.sin(2*np.pi*f1*S.t) + a2*np.sin(2*np.pi*f2*S.t)
S.clean = x.astype(float)
S.noisy = None
S.filtered = None

def load_wav():
path = filedialog.askopenfilename(title="Select WAV file", filetypes=[("WAV files","*.wav")])
if not path:
return
fs, y = wavfile.read(path)
# Convert to float in [-1,1]
if np.issubdtype(y.dtype, np.integer):
maxv = np.iinfo(y.dtype).max
y = y.astype(np.float64) / maxv
else:
y = y.astype(np.float64)
y = to_mono(y)

S.fs = int(fs)
S.duration = len(y)/S.fs
S.t = np.arange(len(y))/S.fs
S.clean = normalize(y)
S.noisy = None
S.filtered = None
messagebox.showinfo("Loaded", f"Loaded: {path}\nfs={S.fs} Hz, duration={S.duration:.2f}s")

def add_noise():
if S.clean is None:
messagebox.showwarning("Missing signal", "Generate or load audio first.")
return
white = S.white_amp * np.random.randn(len(S.t))
hum = S.hum_amp * np.sin(2*np.pi*S.hum_freq*S.t)
S.noisy = S.clean + white + hum
S.filtered = None

def apply_filters():
if S.noisy is None and S.clean is None:
messagebox.showwarning("Missing signal", "Generate or load audio first.")
return
x = S.noisy if S.noisy is not None else S.clean
y = np.copy(x)

if S.notch_on:
b, a = signal.iirnotch(S.notch_f0, S.notch_Q, S.fs)
y = signal.filtfilt(b, a, y)

if S.hp_on:
b, a = signal.butter(S.hp_order, S.hp_cut/(S.fs/2), btype="high", output="ba")
y = signal.filtfilt(b, a, y)

if S.lp_on:
b, a = signal.butter(S.lp_order, S.lp_cut/(S.fs/2), btype="low", output="ba")
y = signal.filtfilt(b, a, y)

S.filtered = y

def export_wav():
if S.filtered is None and S.noisy is None and S.clean is None:
messagebox.showwarning("Nothing to export", "Create a signal first.")
return
x = S.filtered if S.filtered is not None else (S.noisy if S.noisy is not None else S.clean)
x = normalize(x)
x16 = np.int16(np.clip(x, -1.0, 1.0) * 32767)
path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files","*.wav")])
if not path:
return
wavfile.write(path, S.fs, x16)
messagebox.showinfo("Saved", f"Saved: {path}")

# -------------------------- GUI --------------------------
root = tk.Tk()
root.title("Audio Noise Lab — Desktop")

nb = ttk.Notebook(root)
nb.pack(fill="both", expand=True)

# ===== Tab 1: Generate / Load =====
tab1 = ttk.Frame(nb); nb.add(tab1, text="Generate/Load Audio")
frame1 = ttk.Frame(tab1); frame1.pack(side="top", fill="x", padx=10, pady=6)

tk.Label(frame1, text="Sampling rate (Hz):").grid(row=0, column=0, sticky="w")
fs_var = tk.IntVar(value=S.fs)
ttk.Entry(frame1, textvariable=fs_var, width=10).grid(row=0, column=1, padx=5)

tk.Label(frame1, text="Duration (s):").grid(row=0, column=2, sticky="w")
dur_var = tk.DoubleVar(value=S.duration)
ttk.Entry(frame1, textvariable=dur_var, width=10).grid(row=0, column=3, padx=5)

tk.Label(frame1, text="f1 (Hz):").grid(row=1, column=0, sticky="w")
f1_var = tk.DoubleVar(value=440.0)
ttk.Entry(frame1, textvariable=f1_var, width=10).grid(row=1, column=1, padx=5)

tk.Label(frame1, text="f2 (Hz):").grid(row=1, column=2, sticky="w")
f2_var = tk.DoubleVar(value=880.0)
ttk.Entry(frame1, textvariable=f2_var, width=10).grid(row=1, column=3, padx=5)

tk.Label(frame1, text="2nd tone amplitude:").grid(row=1, column=4, sticky="w")
a2_var = tk.DoubleVar(value=0.4)
ttk.Entry(frame1, textvariable=a2_var, width=10).grid(row=1, column=5, padx=5)

def on_generate():
S.fs = int(fs_var.get()); S.duration = float(dur_var.get())
generate_tone(f1_var.get(), f2_var.get(), a2_var.get())
plot_time(ax_t1, S.t, S.clean, "Raw Clean Audio"); refresh(canvas_t1)

ttk.Button(frame1, text="Generate Tone", command=on_generate).grid(row=0, column=4, padx=6)
ttk.Button(frame1, text="Load WAV…", command=load_wav).grid(row=0, column=5, padx=6)

fig_t1 = Figure(figsize=(8, 3), dpi=100); ax_t1 = fig_t1.add_subplot(111)
canvas_t1 = FigureCanvasTkAgg(fig_t1, master=tab1)
canvas_t1.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=6)

# ===== Tab 2: Add Noise =====
tab2 = ttk.Frame(nb); nb.add(tab2, text="Add Noise")
frame2 = ttk.Frame(tab2); frame2.pack(side="top", fill="x", padx=10, pady=6)

tk.Label(frame2, text="White noise amp:").grid(row=0, column=0, sticky="w")
wn_var = tk.DoubleVar(value=S.white_amp); ttk.Entry(frame2, textvariable=wn_var, width=10).grid(row=0, column=1, padx=5)

tk.Label(frame2, text="Hum amp:").grid(row=0, column=2, sticky="w")
humA_var = tk.DoubleVar(value=S.hum_amp); ttk.Entry(frame2, textvariable=humA_var, width=10).grid(row=0, column=3, padx=5)

tk.Label(frame2, text="Hum freq (Hz):").grid(row=0, column=4, sticky="w")
humF_var = tk.DoubleVar(value=S.hum_freq); ttk.Entry(frame2, textvariable=humF_var, width=10).grid(row=0, column=5, padx=5)

def on_noise():
if S.clean is None:
messagebox.showwarning("Missing signal", "Generate or load audio first.")
return
S.white_amp = float(wn_var.get()); S.hum_amp = float(humA_var.get()); S.hum_freq = float(humF_var.get())
add_noise()
plot_time(ax_t2, S.t, S.noisy, "Audio + Noise (White + Hum)"); refresh(canvas_t2)

ttk.Button(frame2, text="Apply Noise", command=on_noise).grid(row=0, column=6, padx=6)

fig_t2 = Figure(figsize=(8, 3), dpi=100); ax_t2 = fig_t2.add_subplot(111)
canvas_t2 = FigureCanvasTkAgg(fig_t2, master=tab2)
canvas_t2.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=6)

# ===== Tab 3: Apply Filters =====
tab3 = ttk.Frame(nb); nb.add(tab3, text="Apply Filters")
frame3 = ttk.LabelFrame(tab3, text="Filters"); frame3.pack(side="top", fill="x", padx=10, pady=6)

# Tk variables bound to state toggles
notch_on_var = tk.BooleanVar(value=S.notch_on)
hp_on_var = tk.BooleanVar(value=S.hp_on)
lp_on_var = tk.BooleanVar(value=S.lp_on)

ttk.Checkbutton(frame3, text="Notch (powerline)", variable=notch_on_var).grid(row=0, column=0, sticky="w")
tk.Label(frame3, text="f0:").grid(row=0, column=1, sticky="e")
notch_f0_var = tk.DoubleVar(value=S.notch_f0); ttk.Entry(frame3, textvariable=notch_f0_var, width=7).grid(row=0, column=2)
tk.Label(frame3, text="Q:").grid(row=0, column=3, sticky="e")
notch_Q_var = tk.DoubleVar(value=S.notch_Q); ttk.Entry(frame3, textvariable=notch_Q_var, width=7).grid(row=0, column=4)

ttk.Checkbutton(frame3, text="High-pass", variable=hp_on_var).grid(row=1, column=0, sticky="w")
tk.Label(frame3, text="cut (Hz):").grid(row=1, column=1, sticky="e")
hp_cut_var = tk.DoubleVar(value=S.hp_cut); ttk.Entry(frame3, textvariable=hp_cut_var, width=7).grid(row=1, column=2)
tk.Label(frame3, text="order:").grid(row=1, column=3, sticky="e")
hp_ord_var = tk.IntVar(value=S.hp_order); ttk.Entry(frame3, textvariable=hp_ord_var, width=7).grid(row=1, column=4)

ttk.Checkbutton(frame3, text="Low-pass", variable=lp_on_var).grid(row=2, column=0, sticky="w")
tk.Label(frame3, text="cut (Hz):").grid(row=2, column=1, sticky="e")
lp_cut_var = tk.DoubleVar(value=S.lp_cut); ttk.Entry(frame3, textvariable=lp_cut_var, width=7).grid(row=2, column=2)
tk.Label(frame3, text="order:").grid(row=2, column=3, sticky="e")
lp_ord_var = tk.IntVar(value=S.lp_order); ttk.Entry(frame3, textvariable=lp_ord_var, width=7).grid(row=2, column=4)

def on_filter():
if S.noisy is None and S.clean is None:
messagebox.showwarning("Missing signal", "Generate or load audio first.")
return
# sync UI->state
S.notch_on = bool(notch_on_var.get()); S.notch_f0 = float(notch_f0_var.get()); S.notch_Q = float(notch_Q_var.get())
S.hp_on = bool(hp_on_var.get()); S.hp_cut = float(hp_cut_var.get()); S.hp_order = int(hp_ord_var.get())
S.lp_on = bool(lp_on_var.get()); S.lp_cut = float(lp_cut_var.get()); S.lp_order = int(lp_ord_var.get())

apply_filters()
x_before = S.noisy if S.noisy is not None else S.clean
x_after = S.filtered if S.filtered is not None else x_before
plot_two(ax3a, ax3b, S.t, x_before, x_after, "Before Filtering", "After Filtering (Noise Reduced)")
refresh(canvas_t3)

ttk.Button(frame3, text="Apply Filters", command=on_filter).grid(row=0, column=5, rowspan=3, padx=10)

fig_t3 = Figure(figsize=(8, 5), dpi=100)
ax3a = fig_t3.add_subplot(211); ax3b = fig_t3.add_subplot(212)
canvas_t3 = FigureCanvasTkAgg(fig_t3, master=tab3)
canvas_t3.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=6)

# ===== Tab 4: Analyze & Export =====
tab4 = ttk.Frame(nb); nb.add(tab4, text="Analyze & Export")
frame4 = ttk.Frame(tab4); frame4.pack(side="top", fill="x", padx=10, pady=6)
ttk.Button(frame4, text="Export WAV…", command=export_wav).pack(side="left")

fig_t4 = Figure(figsize=(8, 3.2), dpi=100); ax_t4 = fig_t4.add_subplot(111)
canvas_t4 = FigureCanvasTkAgg(fig_t4, master=tab4)
canvas_t4.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=6)

def update_psd_plot(*args):
x_noisy = S.noisy if S.noisy is not None else S.clean
plot_psd(ax_t4, S.fs, x_noisy, S.filtered); refresh(canvas_t4)

def on_tab_changed(event):
if nb.tab(nb.select(), "text") == "Analyze & Export":
update_psd_plot()
nb.bind("<<NotebookTabChanged>>", on_tab_changed)

# Initial placeholders
plot_time(ax_t1, S.t, np.zeros_like(S.t), "Raw Clean Audio")
plot_time(ax_t2, S.t, np.zeros_like(S.t), "Audio + Noise (White + Hum)")
plot_two(ax3a, ax3b, S.t, np.zeros_like(S.t), np.zeros_like(S.t),
"Before Filtering", "After Filtering (Noise Reduced)")
plot_psd(ax_t4, S.fs, None, None)
refresh(canvas_t1); refresh(canvas_t2); refresh(canvas_t3); refresh(canvas_t4)

root.mainloop()
