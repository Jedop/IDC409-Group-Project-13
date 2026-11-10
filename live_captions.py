#!/usr/bin/env python3
"""
Live captions for the speech-to-text project (standalone, not part of the repo).

Like YouTube subtitles, but for your own microphone: a caption bar floats at the bottom of the
screen, on top of every other window (slides, browser, ...), and a control panel lets you change
the model, add noise, move/resize the captions, pick the microphone, and so on.
The caption bar ignores mouse clicks, so it never gets in your way. Close the control panel
(or press Quit) to stop everything.

How it works: microphone -> simple voice-activity detection -> faster-whisper (CPU).
Whisper is not a streaming model, so while you speak the script re-transcribes the growing phrase
every ~0.7 s (grey "live" text that may still change) and replaces it with a final, more careful
transcription when you pause.

Install (Arch / EndeavourOS), inside the same virtual environment as the project:
  sudo pacman -S portaudio xcb-util-cursor
  pip install PySide6 sounddevice

Run:
  python live_captions.py                       # default microphone, base.en
  python live_captions.py --list-devices        # list microphones
  python live_captions.py --model tiny.en       # faster, a bit less accurate
  python live_captions.py --file clip.wav       # rehearse with an audio file instead of the mic
  python live_captions.py --offline             # never touch the internet (models must be cached)
  python live_captions.py --no-model            # test mic level / phrase detection without Whisper
  python live_captions.py --bypass-wm           # if fullscreen apps hide the captions (X11 only)

KDE Plasma on Wayland: the script runs through XWayland (the default), which keeps "always on top" and
click-through working. A fullscreen window that is active can still cover the bar; see the README notes
in the chat (use --bypass-wm, a KWin window rule for the title "Live captions overlay", or present
maximised instead of fullscreen).
"""
import argparse
import collections
import datetime
import html
import os
import queue
import sys
import threading
import time
from math import gcd

import numpy as np

SR = 16000
FRAME = 480                     # 30 ms at 16 kHz
FRAME_S = FRAME / SR
LOW_CONF, MID_CONF = 0.50, 0.65
NOISE_CYCLE = [None, 10, 5, 0]
NOISE_LABELS = ["off", "10 dB", "5 dB", "0 dB"]
WHITE, AMBER, RED, LIVE_GREY = "#ffffff", "#fbbf24", "#f87171", "#cbd5e1"


# =============================================================================== audio helpers
def to_16k(x, sr):
    if sr == SR:
        return x.astype(np.float32)
    from scipy.signal import resample_poly
    g = gcd(SR, sr)
    return resample_poly(x, SR // g, sr // g).astype(np.float32)


def add_noise(signal, snr_db, rng):
    """Same white-noise recipe as the project's data_prep.py."""
    power = np.mean(signal ** 2) + 1e-12
    noise = rng.normal(0.0, np.sqrt(power / (10 ** (snr_db / 10))), size=signal.shape)
    noisy = signal + noise
    peak = np.max(np.abs(noisy))
    return (noisy / peak if peak > 1.0 else noisy).astype(np.float32)


class Segmenter:
    """Energy-based phrase detector: cuts the audio stream into spoken phrases."""

    def __init__(self, margin=3.0, min_floor=0.006, min_speech_s=0.4, end_silence_s=0.6, max_len_s=8.0):
        self.margin, self.min_floor = margin, min_floor
        self.min_speech_s, self.end_silence_s, self.max_len_s = min_speech_s, end_silence_s, max_len_s
        self.rest = np.zeros(0, np.float32)
        self.preroll = collections.deque(maxlen=10)       # 0.3 s of audio before speech starts
        self.level = 0.0
        self.reset_calibration()
        self.reset_buffers()

    def reset_calibration(self):
        self.floor, self.calib = None, []

    def reset_buffers(self):
        self.buf, self.in_speech = [], False
        self.voiced_run = self.silence_run = 0
        self.preroll.clear()

    @property
    def threshold(self):
        return max((self.floor or 0.0) * self.margin, self.min_floor)

    def snapshot(self):
        """Audio of the phrase spoken so far (None if nobody is speaking)."""
        buf = list(self.buf)
        return np.concatenate(buf) if (self.in_speech and buf) else None

    def feed(self, x):
        done = []
        self.rest = np.concatenate([self.rest, x])
        while len(self.rest) >= FRAME:
            frame, self.rest = self.rest[:FRAME], self.rest[FRAME:]
            seg = self._frame(frame)
            if seg is not None:
                done.append(seg)
        return done

    def _frame(self, f):
        rms = float(np.sqrt(np.mean(f * f) + 1e-12))
        self.level = rms
        if self.floor is None:                            # calibration: first ~1 s
            self.calib.append(rms)
            if len(self.calib) >= 33:
                self.floor = float(np.median(self.calib))
            return None
        voiced = rms > self.threshold
        if not self.in_speech:
            self.preroll.append(f)
            if voiced:
                self.voiced_run += 1
            else:
                self.voiced_run = 0
                self.floor = 0.95 * self.floor + 0.05 * rms     # slowly follow the room noise
            if self.voiced_run >= 3:
                self.in_speech, self.buf, self.silence_run = True, list(self.preroll), 0
            return None
        self.buf.append(f)
        self.silence_run = 0 if voiced else self.silence_run + 1
        n = len(self.buf)
        if self.silence_run * FRAME_S >= self.end_silence_s or n * FRAME_S >= self.max_len_s:
            speech_s = (n - self.silence_run) * FRAME_S
            audio = np.concatenate(self.buf)
            self.reset_buffers()
            if speech_s >= self.min_speech_s:
                return audio
        return None


# =============================================================================== speech recognition
class Transcriber:
    def __init__(self, offline=False, fake=False):
        self.offline, self.fake = offline, fake
        self.models = {}

    def load(self, name):
        if self.fake or name in self.models:
            return
        from faster_whisper import WhisperModel
        self.models[name] = WhisperModel(name, device="cpu", compute_type="int8",
                                         local_files_only=self.offline)

    def transcribe(self, name, audio, interim=False):
        """Returns (text, confidence). Interim = fast greedy decoding for the grey live text."""
        if self.fake:
            return (f"[hearing speech... {len(audio) / SR:.1f} s]" if interim
                    else f"[phrase detected: {len(audio) / SR:.1f} s of speech]"), 0.8
        kw = dict(beam_size=1, temperature=0.0) if interim else dict(beam_size=5)
        segments, _ = self.models[name].transcribe(audio, language="en",
                                                   condition_on_previous_text=False, **kw)
        segs = list(segments)
        text = " ".join(s.text.strip() for s in segs).strip()
        if not segs:
            return "", 0.0
        w = [max(s.end - s.start, 1e-3) for s in segs]
        return text, float(np.exp(np.average([s.avg_logprob for s in segs], weights=w)))


class Engine:
    """Microphone + recognition running in background threads. The window only reads/sets attributes."""

    def __init__(self, model, offline=False, fake=False, device=None):
        self.model, self.device = model, device
        self.noise_idx, self.live, self.listening = 0, True, True
        self.stop, self.loading, self.busy, self.mic_changed = False, False, False, False
        self.mic_error = None
        self.bad = set()
        self.seg = Segmenter()
        self.tr = Transcriber(offline, fake)
        self.raw_q, self.res_q = queue.Queue(), queue.Queue()

    @property
    def noise_db(self):
        return NOISE_CYCLE[self.noise_idx]

    def start(self, file=None):
        threading.Thread(target=self._worker, daemon=True).start()
        target = (lambda: self._file_loop(file)) if file else self._mic_loop
        threading.Thread(target=target, daemon=True).start()

    # ---------------------------------------------------------------- input
    def _open_stream(self, sd):
        cb16 = lambda indata, f, t, s: self.raw_q.put(indata[:, 0].copy())
        try:
            return sd.InputStream(samplerate=SR, channels=1, dtype="float32", device=self.device, callback=cb16)
        except Exception:                                 # device cannot do 16 kHz: use its own rate
            native = int(sd.query_devices(self.device, "input")["default_samplerate"])
            cb = lambda indata, f, t, s: self.raw_q.put(to_16k(indata[:, 0].copy(), native))
            return sd.InputStream(samplerate=native, channels=1, dtype="float32", device=self.device, callback=cb)

    def _mic_loop(self):
        import sounddevice as sd
        while not self.stop:
            self.mic_changed = False
            try:
                stream = self._open_stream(sd)
                self.mic_error = None
            except Exception as e:
                self.mic_error = f"Cannot open microphone: {e}"
                while not self.stop and not self.mic_changed:
                    time.sleep(0.2)
                continue
            with stream:
                while not self.stop and not self.mic_changed:
                    time.sleep(0.1)

    def _file_loop(self, path):
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = to_16k(data, sr)
        quiet = np.zeros(int(1.5 * SR), np.float32)
        data = np.concatenate([quiet, data, quiet, quiet])
        step = FRAME * 3
        for i in range(0, len(data), step):
            if self.stop:
                return
            self.raw_q.put(data[i:i + step])
            time.sleep(step / SR)

    # ---------------------------------------------------------------- recognition
    def _ensure_model(self):
        m = self.model
        if self.tr.fake or m in self.tr.models:
            return True
        if m in self.bad:
            ok = list(self.tr.models)
            if ok:
                self.model = ok[0]
            return bool(ok)
        self.loading = True
        try:
            self.tr.load(m)
        except Exception as e:                            # e.g. not downloaded and --offline
            self.bad.add(m)
            self.res_q.put({"kind": "error", "text": f"Could not load model '{m}': {e}"})
            ok = list(self.tr.models)
            if ok:
                self.model = ok[0]
            return bool(ok)
        finally:
            self.loading = False
        return True

    def _transcribe(self, audio, kind):
        model, noise = self.model, self.noise_db
        heard = add_noise(audio, noise, np.random.default_rng(0)) if noise is not None else audio
        self.busy = True
        t0 = time.time()
        try:
            text, conf = self.tr.transcribe(model, heard, interim=(kind == "interim"))
        except Exception as e:
            self.res_q.put({"kind": "error", "text": f"Recognition error: {e}"})
            return
        finally:
            self.busy = False
        self.res_q.put({"kind": kind, "text": text, "conf": conf, "secs": time.time() - t0,
                        "dur": len(audio) / SR, "model": model, "noise": noise})

    def _worker(self):
        last_interim, was_listening = 0.0, True
        while not self.stop:
            if not self._ensure_model():
                time.sleep(0.2)
                while not self.raw_q.empty():
                    self.raw_q.get_nowait()
                continue
            try:
                x = self.raw_q.get(timeout=0.05)
            except queue.Empty:
                x = None
            if not self.listening:
                if was_listening:
                    self.seg.reset_buffers()
                    was_listening = False
                continue
            was_listening = True
            finals = self.seg.feed(x) if x is not None else []
            for audio in finals:
                self._transcribe(audio, "final")
                last_interim = time.time()
            if (not finals and self.live and self.seg.in_speech
                    and time.time() - last_interim >= 0.7 and self.raw_q.qsize() <= 2):
                snap = self.seg.snapshot()
                if snap is not None and len(snap) / SR >= 0.8:
                    self._transcribe(snap, "interim")
                    last_interim = time.time()


# =============================================================================== windows
def build_gui(engine, args):
    from PySide6 import QtCore, QtGui, QtWidgets
    Qt = QtCore.Qt

    class CaptionOverlay(QtWidgets.QWidget):
        """The YouTube-style caption bar: frameless, always on top, ignores the mouse."""

        def __init__(self):
            super().__init__()
            flags = (Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                     | Qt.WindowType.Tool | Qt.WindowType.WindowTransparentForInput
                     | Qt.WindowType.WindowDoesNotAcceptFocus)
            if args.bypass_wm:
                flags |= Qt.WindowType.X11BypassWindowManagerHint
            self.setWindowFlags(flags)
            self.setWindowTitle("Live captions overlay")      # lets a KDE window rule find this window
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
            self.opacity, self.font_px, self.width_pct, self.bottom_pct, self.screen_idx = 0.75, 36, 70, 6, 0
            self.label = QtWidgets.QLabel(self)
            self.label.setWordWrap(True)
            self.label.setTextFormat(Qt.TextFormat.RichText)
            self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.html = ""

        def geometry_for_screen(self):
            screens = QtGui.QGuiApplication.screens()
            return screens[min(self.screen_idx, len(screens) - 1)].availableGeometry()

        def chars_per_line(self):
            return max(20, int((self.geometry_for_screen().width() * self.width_pct / 100 - 50) / (self.font_px * 0.6)))

        def set_html(self, html_text):
            self.html = html_text
            self.relayout()

        def relayout(self):
            geo = self.geometry_for_screen()
            w = int(geo.width() * self.width_pct / 100)
            self.label.setStyleSheet(f"color: white; font-family: 'DejaVu Sans', sans-serif; "
                                     f"font-weight: 600; font-size: {self.font_px}px;")
            self.label.setText(self.html)
            self.label.setFixedWidth(w - 44)
            h = self.label.heightForWidth(w - 44)
            self.label.setFixedHeight(h)
            self.label.move(22, 10)
            self.setFixedSize(w, h + 20)
            self.move(geo.x() + (geo.width() - w) // 2,
                      geo.y() + geo.height() - self.height() - int(geo.height() * self.bottom_pct / 100))
            self.update()

        def paintEvent(self, _):
            p = QtGui.QPainter(self)
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            p.setBrush(QtGui.QColor(0, 0, 0, int(255 * self.opacity)))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(self.rect(), 12, 12)

    class Dashboard(QtWidgets.QWidget):
        def __init__(self, overlay):
            super().__init__()
            self.overlay = overlay
            self.setWindowTitle("Live captions - control panel")
            self.finals = collections.deque()             # (text, conf, timestamp)
            self.interim = ""
            self.show_captions, self.color_by_conf, self.hold_s = True, True, 5.0
            self.history, self.last_info = [], "-"
            self._build()
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.poll)
            self.timer.start(60)

        # ------------------------------------------------------------- layout
        def _build(self):
            lay = QtWidgets.QVBoxLayout(self)
            top = QtWidgets.QHBoxLayout()
            self.toggle = QtWidgets.QPushButton()
            self.toggle.setCheckable(True)
            self.toggle.setChecked(True)
            self.toggle.setMinimumHeight(46)
            self.toggle.toggled.connect(self.on_toggle)
            top.addWidget(self.toggle, 2)
            self.status = QtWidgets.QLabel("starting...")
            self.status.setWordWrap(True)
            top.addWidget(self.status, 3)
            lay.addLayout(top)
            self.level = QtWidgets.QProgressBar()
            self.level.setRange(0, 100)
            self.level.setTextVisible(False)
            self.level.setFixedHeight(10)
            lay.addWidget(self.level)
            self.stats = QtWidgets.QLabel("-")
            self.stats.setStyleSheet("color: #94a3b8;")
            lay.addWidget(self.stats)

            rec = QtWidgets.QGroupBox("Recognition")
            f = QtWidgets.QFormLayout(rec)
            self.model_box = QtWidgets.QComboBox()
            self.model_box.addItems(["tiny.en", "base.en"])
            self.model_box.setCurrentText(engine.model)
            self.model_box.currentTextChanged.connect(lambda m: setattr(engine, "model", m))
            f.addRow("Model", self.model_box)
            self.noise_box = QtWidgets.QComboBox()
            self.noise_box.addItems(NOISE_LABELS)
            self.noise_box.currentIndexChanged.connect(lambda i: setattr(engine, "noise_idx", i))
            f.addRow("Added noise", self.noise_box)
            self.live_check = QtWidgets.QCheckBox("Live (grey) text while speaking")
            self.live_check.setChecked(True)
            self.live_check.toggled.connect(lambda v: setattr(engine, "live", v))
            f.addRow("", self.live_check)
            self.mic_box = QtWidgets.QComboBox()
            self.mic_box.addItem("System default", None)
            if not args.file:
                try:
                    import sounddevice as sd
                    for i, d in enumerate(sd.query_devices()):
                        if d["max_input_channels"] > 0:
                            self.mic_box.addItem(f"{i}: {d['name']}", i)
                    if engine.device is not None:
                        self.mic_box.setCurrentIndex(max(0, self.mic_box.findData(engine.device)))
                except Exception:
                    pass
            else:
                self.mic_box.addItem(f"file: {os.path.basename(args.file)}", None)
                self.mic_box.setCurrentIndex(1)
                self.mic_box.setEnabled(False)
            self.mic_box.currentIndexChanged.connect(self.on_mic)
            f.addRow("Microphone", self.mic_box)
            lay.addWidget(rec)

            cap = QtWidgets.QGroupBox("Captions")
            g = QtWidgets.QFormLayout(cap)
            self.show_check = QtWidgets.QCheckBox("Show caption bar")
            self.show_check.setChecked(True)
            self.show_check.toggled.connect(lambda v: setattr(self, "show_captions", v))
            g.addRow("", self.show_check)
            self.color_check = QtWidgets.QCheckBox("Colour by confidence (amber < 0.65, red < 0.50)")
            self.color_check.setChecked(True)
            self.color_check.toggled.connect(lambda v: setattr(self, "color_by_conf", v))
            g.addRow("", self.color_check)
            g.addRow("Text size", self.slider(20, 90, self.overlay.font_px, "font_px"))
            g.addRow("Background", self.slider(0, 100, int(self.overlay.opacity * 100), "opacity", 0.01))
            g.addRow("Width", self.slider(30, 100, self.overlay.width_pct, "width_pct"))
            g.addRow("Height above bottom", self.slider(0, 80, self.overlay.bottom_pct, "bottom_pct"))
            self.screen_box = QtWidgets.QComboBox()
            for i, s in enumerate(QtGui.QGuiApplication.screens()):
                self.screen_box.addItem(f"{i}: {s.name()}  ({s.geometry().width()}x{s.geometry().height()})")
            self.screen_box.currentIndexChanged.connect(self.on_screen)
            g.addRow("Display", self.screen_box)
            self.hold = QtWidgets.QDoubleSpinBox()
            self.hold.setRange(1, 30)
            self.hold.setValue(self.hold_s)
            self.hold.setSuffix(" s")
            self.hold.valueChanged.connect(lambda v: setattr(self, "hold_s", v))
            g.addRow("Keep captions for", self.hold)
            lay.addWidget(cap)

            row = QtWidgets.QHBoxLayout()
            for text, fn in (("Test caption", self.test_caption), ("Clear", self.clear),
                             ("Recalibrate mic", engine.seg.reset_calibration),
                             ("Save transcript", self.save), ("Quit", self.close)):
                b = QtWidgets.QPushButton(text)
                b.clicked.connect(fn)
                row.addWidget(b)
            lay.addLayout(row)

            self.log = QtWidgets.QTextEdit()
            self.log.setReadOnly(True)
            self.log.setMinimumHeight(140)
            lay.addWidget(self.log)
            self.on_toggle(True)

        def slider(self, lo, hi, val, attr, scale=1):
            s = QtWidgets.QSlider(Qt.Orientation.Horizontal)
            s.setRange(lo, hi)
            s.setValue(val)
            s.valueChanged.connect(lambda v: (setattr(self.overlay, attr, v * scale if scale != 1 else v),
                                              self.refresh_overlay(force=True)))
            return s

        # ------------------------------------------------------------- actions
        def on_toggle(self, on):
            engine.listening = on
            self.toggle.setText("LISTENING  (click to pause)" if on else "PAUSED  (click to listen)")
            self.toggle.setStyleSheet("background:#15803d;color:white;font-weight:bold;" if on
                                      else "background:#475569;color:white;font-weight:bold;")
            if not on:
                self.interim = ""

        def on_mic(self, _):
            engine.device = self.mic_box.currentData()
            engine.mic_changed = True

        def on_screen(self, i):
            self.overlay.screen_idx = i
            self.refresh_overlay(force=True)

        def test_caption(self):
            self.finals.append(("This is a test caption - drag the sliders to place it.", 0.9, time.time()))

        def clear(self):
            self.finals.clear()
            self.interim = ""
            self.log.clear()
            self.history.clear()

        def save(self):
            name = datetime.datetime.now().strftime("live_transcript_%Y%m%d_%H%M%S.txt")
            with open(name, "w", encoding="utf-8") as fh:
                for t, c, text in self.history:
                    fh.write(f"{t}\t{c:.2f}\t{text}\n")
            self.stats.setText(f"saved {len(self.history)} phrases to {os.path.abspath(name)}")

        def closeEvent(self, e):
            engine.stop = True
            self.overlay.hide()
            QtWidgets.QApplication.quit()
            e.accept()

        # ------------------------------------------------------------- live updates
        def colour(self, conf):
            if not self.color_by_conf or conf >= MID_CONF:
                return WHITE
            return AMBER if conf >= LOW_CONF else RED

        def refresh_overlay(self, force=False):
            now = time.time()
            while self.finals and now - self.finals[0][2] > self.hold_s:
                self.finals.popleft()
            runs = [(t, self.colour(c)) for t, c, _ in self.finals]
            if self.interim:
                runs.append((self.interim, LIVE_GREY))
            maxchars = self.overlay.chars_per_line() * 2          # show at most two lines, like YouTube
            total = sum(len(t) + 1 for t, _ in runs)
            while total > maxchars and runs:
                t, col = runs[0]
                over = total - maxchars
                if over >= len(t) + 1 or len(runs) == 1 and over >= len(t):
                    runs.pop(0)
                else:
                    cut = t[over:]
                    cut = cut.split(" ", 1)[1] if " " in cut else cut
                    runs[0] = (cut, col)
                total = sum(len(t) + 1 for t, _ in runs)
            htm = " ".join(f'<span style="color:{col}">{html.escape(t)}</span>' for t, col in runs if t)
            if htm and self.show_captions:
                if force or htm != self.overlay.html or not self.overlay.isVisible():
                    self.overlay.set_html(htm)
                if not self.overlay.isVisible():
                    self.overlay.show()
            elif self.overlay.isVisible():
                self.overlay.hide()
                self.overlay.html = ""

        def poll(self):
            if engine.stop:
                self.close()
                return
            try:
                while True:
                    r = engine.res_q.get_nowait()
                    if r["kind"] == "error":
                        self.log.append(f'<span style="color:{RED}">{html.escape(r["text"])}</span>')
                    elif r["kind"] == "interim":
                        self.interim = r["text"]
                    else:                                         # final
                        self.interim = ""
                        if r["text"]:
                            self.finals.append((r["text"], r["conf"], time.time()))
                            stamp = datetime.datetime.now().strftime("%H:%M:%S")
                            self.history.append((stamp, r["conf"], r["text"]))
                            col = {WHITE: "#e2e8f0"}.get(self.colour(r["conf"]), self.colour(r["conf"]))
                            self.log.append(f'<span style="color:#64748b">{stamp}</span> '
                                            f'<span style="color:{col}">[{r["conf"]:.2f}] {html.escape(r["text"])}</span>')
                            self.last_info = (f"last phrase: confidence {r['conf']:.2f} | {r['secs']:.1f} s to "
                                              f"transcribe {r['dur']:.1f} s of speech | {r['model']}")
            except queue.Empty:
                pass
            if self.model_box.currentText() != engine.model:       # model fell back after a load error
                self.model_box.setCurrentText(engine.model)
            seg = engine.seg
            if engine.mic_error:
                s = engine.mic_error
            elif engine.loading:
                s = "loading model... (first time can take a few seconds)"
            elif not engine.listening:
                s = "paused - not listening"
            elif seg.floor is None:
                s = "calibrating background level - stay quiet for 1 second"
            elif engine.busy:
                s = "transcribing..."
            elif seg.in_speech:
                s = "hearing speech..."
            else:
                s = "listening"
            self.status.setText(f"{s}\nmodel {engine.model} | added noise: {NOISE_LABELS[engine.noise_idx]}")
            self.level.setValue(int(min(seg.level / max(seg.threshold, 1e-6), 3.0) / 3.0 * 100))
            if not self.stats.text().startswith("saved"):
                self.stats.setText(self.last_info)
            elif time.time() % 8 < 0.1:
                self.stats.setText(self.last_info)
            self.refresh_overlay()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    for role, col in ((QtGui.QPalette.ColorRole.Window, "#111827"), (QtGui.QPalette.ColorRole.WindowText, "#e5e7eb"),
                      (QtGui.QPalette.ColorRole.Base, "#0b1220"), (QtGui.QPalette.ColorRole.Text, "#e5e7eb"),
                      (QtGui.QPalette.ColorRole.Button, "#1f2937"), (QtGui.QPalette.ColorRole.ButtonText, "#e5e7eb"),
                      (QtGui.QPalette.ColorRole.Highlight, "#2a9d8f")):
        pal.setColor(role, QtGui.QColor(col))
    app.setPalette(pal)
    overlay = CaptionOverlay()
    dash = Dashboard(overlay)
    return app, dash, overlay


# =============================================================================== main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="base.en", choices=["tiny.en", "base.en"])
    ap.add_argument("--device", type=int, default=None, help="microphone number (see --list-devices)")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--file", help="play a wav/flac file as if it came from the microphone")
    ap.add_argument("--offline", action="store_true", help="use only models that are already downloaded")
    ap.add_argument("--no-model", action="store_true", help="do not run Whisper, only show detected phrases")
    ap.add_argument("--bypass-wm", action="store_true", help="draw the bar above fullscreen windows (X11 only)")
    ap.add_argument("--wayland", action="store_true", help="use native Wayland instead of XWayland")
    args = ap.parse_args(argv)

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return
    if not args.wayland:
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")      # XWayland keeps "always on top" and click-through working
    if not args.file:
        try:
            import sounddevice  # noqa: F401
        except (ImportError, OSError) as e:
            sys.exit(f"sounddevice is not available ({e}).\nInstall: sudo pacman -S portaudio && pip install sounddevice")

    engine = Engine(args.model, offline=args.offline, fake=args.no_model, device=args.device)
    app, dash, overlay = build_gui(engine, args)
    engine.start(file=args.file)
    dash.show()
    code = app.exec()
    engine.stop = True
    sys.exit(code)


if __name__ == "__main__":
    main()
