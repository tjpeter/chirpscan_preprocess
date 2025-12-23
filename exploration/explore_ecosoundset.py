import os
from functools import lru_cache

import random
import numpy as np
import pandas as pd
import librosa

import panel as pn
import holoviews as hv
import logging

import hashlib
import tempfile
from pathlib import Path

try:
    import soundfile as sf
    _HAVE_SF = True
except Exception:
    _HAVE_SF = False
    from scipy.io import wavfile


# FIX 1: "audio" extension is not needed anymore
pn.extension()
hv.extension("bokeh")

# ----------------------------
# CONFIG
# ----------------------------
ANNOTATIONS_PATH = "./data/ECOSoundSet/annotated_audio_segments.csv" 
METADATA_PATH = "./data/ECOSoundSet/recording_metadata.csv"
SEP = ","
AUDIO_BASE_DIR = "./data/ECOSoundSet/Split recordings"              
N_FFT = 1024
HOP = 256
MAX_VIEWS = 8                       

HEIGHT = 500
WIDTH = 800
FONT_SIZE = "12pt"
FONT_COLOR = "blue"

NORMALIZED_CACHE_DIR = Path("./.normalized_cache")
NORMALIZED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
MAX_FILES_IN_CACHE = 100  # max number of normalized files to keep

# ----------------------------
# Utils
# ----------------------------

def prune_normalized_cache(max_files: int = 100) -> None:
    """Keep only the most recently modified normalized wav files."""
    try:
        files = sorted(
            NORMALIZED_CACHE_DIR.glob("*.wav"),
            key=lambda p: p.stat().st_mtime,
        )
        for p in files[:-max_files]:
            p.unlink(missing_ok=True)
    except Exception as e:
        logging.warning(f"Could not prune normalized cache: {e}")


# Prune once on startup so the cache doesn't grow without bound.
prune_normalized_cache(max_files=MAX_FILES_IN_CACHE)


# ----------------------------
# DATA (lazy loaded)
# ----------------------------
_df = None

def get_df() -> pd.DataFrame:
    """Load annotations dataframe joined with metadata (cached after first call)."""
    global _df
    if _df is None:
        if not os.path.exists(ANNOTATIONS_PATH):
            raise FileNotFoundError(f"Annotations file not found: {ANNOTATIONS_PATH}")
        if not os.path.exists(METADATA_PATH):
            raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")
       
        annotations = pd.read_csv(ANNOTATIONS_PATH, sep=SEP)
        metadata = pd.read_csv(METADATA_PATH, sep=SEP)
       
        _df = annotations.merge(metadata, on="recording_id", how="left")
        _df["full_fpath"] = _df.apply(
            lambda row: os.path.join(AUDIO_BASE_DIR, str(row["license"]), row["audio_segment_file_name"]),
            axis=1
        )
    return _df

def audio_path(fname: str) -> str:
    df = get_df()
    match = df[df.audio_segment_file_name == fname]
    if not match.empty:
        return match.iloc[0]["full_fpath"]
    return fname if os.path.isabs(fname) else os.path.join(AUDIO_BASE_DIR, fname)

def get_segment_time_range(fname: str) -> tuple[float, float]:
    df = get_df()
    match = df[df.audio_segment_file_name == fname]
    if not match.empty:
        row = match.iloc[0]
        return float(row["audio_segment_initial_time"]), float(row["audio_segment_final_time"])
    return 0.0, 0.0

# ----------------------------
# Audio normalization
# ----------------------------
def rms_dbfs(y: np.ndarray, eps: float = 1e-12) -> float:
    """RMS level in dBFS (0 dBFS == full scale sine with rms=1.0)."""
    rms = float(np.sqrt(np.mean(np.square(y), dtype=np.float64)))
    return 20.0 * np.log10(max(rms, eps))

def peak(y: np.ndarray) -> float:
    return float(np.max(np.abs(y))) if y.size else 0.0

@lru_cache(maxsize=256)
def normalized_audio_path(
    fname: str,
    method: str = "rms",
    target_dbfs: float = -20.0,
    boost_db: float = 0.0,
) -> str | None:
    """
    Create (and cache) a normalized wav file for playback.
    method: "rms" (targets RMS dBFS) or "peak" (peak normalize to -1 dBFS-ish).
    boost_db: post-normalization gain in dB (like an "Amplify" knob).
    """
    result = load_audio(fname)
    if result is None:
        return None
    y, sr = result

    # Build a stable filename based on original path + params + file mtime
    src = audio_path(fname)
    try:
        mtime = os.path.getmtime(src)
    except Exception:
        mtime = 0.0

    key = f"{src}|{mtime}|{method}|{target_dbfs}|{boost_db}"
    out_name = hashlib.md5(key.encode("utf-8")).hexdigest() + ".wav"
    out_path = NORMALIZED_CACHE_DIR / out_name
    if out_path.exists():
        return str(out_path)

    y2 = y.astype(np.float32, copy=True)

    # 1) Normalize
    if method == "peak":
        p = peak(y2)
        if p > 0:
            target_peak = 10 ** (-1.0 / 20.0)  # -1 dBFS
            y2 *= (target_peak / p)
    else:
        cur = rms_dbfs(y2)
        y2 *= 10 ** ((target_dbfs - cur) / 20.0)

    # 2) Optional boost (post-normalization)
    if boost_db != 0.0:
        y2 *= 10 ** (boost_db / 20.0)

    # 3) Always limit after boost (for BOTH methods)
    p = peak(y2)
    if p > 0.99:
        y2 *= (0.99 / p)

    logging.info(
        f"[normalized_audio_path] OUT RMS {rms_dbfs(y2):.1f} dBFS | "
        f"Peak {20*np.log10(max(peak(y2), 1e-12)):.1f} dBFS | "
        f"boost_db={boost_db}"
    )

    # 4) Write wav (PCM_16 is browser-friendly)
    if _HAVE_SF:
        sf.write(str(out_path), np.clip(y2, -1.0, 1.0), sr, subtype="PCM_16")
    else:
        y16 = np.int16(np.clip(y2, -1.0, 1.0) * 32767)
        wavfile.write(str(out_path), sr, y16)

    prune_normalized_cache(max_files=MAX_FILES_IN_CACHE)
    return str(out_path)


# ----------------------------
# Caching: audio + spec
# ----------------------------
@lru_cache(maxsize=256)
def load_audio(fname: str) -> tuple[np.ndarray, int] | None:
    try:
        y, sr = librosa.load(audio_path(fname), sr=None, mono=True)
        return y, sr
    except Exception as e:
        logging.error(f"Failed to load audio {fname}: {e}")
        return None

@lru_cache(maxsize=256)
def spec_cached(fname: str):
    result = load_audio(fname)
    if result is None:
        return None, None, None
    y, sr = result
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    t_start, t_end = get_segment_time_range(fname)
    times_relative = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=HOP)
    times = times_relative + t_start
    return S_db, times, freqs

def make_view(segment: str, labels=None, cats=None, color_by="label_category", 
              vmin=-80, vmax=0, cmap="Greys",
              do_normalize=False, norm_method="rms", target_dbfs=-20, boost_db=0):
    
    df = get_df()
    ann = df[df.audio_segment_file_name == segment].copy()

    if cats:
        ann = ann[ann.label_category.isin(cats)]
    if labels:
        ann = ann[ann.label.isin(labels)]

    S_db, times, freqs = spec_cached(segment)
    if S_db is None:
        return pn.pane.Markdown(f"**{segment}**: Audio file not found")
    
    t_start, t_end = get_segment_time_range(segment)

    # 1. Base Spectrogram
    img = hv.Image(
        (times, freqs, S_db),
        kdims=["Time (s)", "Frequency (Hz)"],
        vdims=["dB"],
    ).opts(
        cmap=cmap, height=HEIGHT, width=WIDTH, tools=["hover"],
        colorbar=True, colorbar_opts={"title": "dB"},
        xlim=(t_start, t_end), clim=(vmin, vmax),
    )

    overlay_elements = [img]

    # 2. Annotations
    rects = [
        (float(r.annotation_initial_time), float(r.annotation_min_freq),
         float(r.annotation_final_time),   float(r.annotation_max_freq),
         str(r.label), str(r.label_category))
        for _, r in ann.iterrows()
    ]

    if rects:
        boxes = hv.Rectangles(
            rects, kdims=["x0","y0","x1","y1"], vdims=["label","category"]
        ).opts(
            fill_alpha=0.25, line_width=2, line_color="red",
            tools=["hover"], hover_alpha=0.45
        )
        overlay_elements.append(boxes)

        text_labels = []
        for _, r in ann.iterrows():
            x_pos = (float(r.annotation_initial_time) + float(r.annotation_final_time)) / 2
            y_pos = (float(r.annotation_min_freq) + float(r.annotation_max_freq)) / 2 
            label_text = f"{r.label} ({r.label_category})"
            text_labels.append((x_pos, y_pos, label_text))
        
        if text_labels:
            labels_overlay = hv.Labels(text_labels, kdims=["x", "y"], vdims=["text"]).opts(
                text_font_size=FONT_SIZE, text_color=FONT_COLOR, text_align="center"
            )
            overlay_elements.append(labels_overlay)

    # 3. Audio & Synchronization
    src_path = audio_path(segment)
    if do_normalize or float(boost_db) != 0.0:
        norm_path = normalized_audio_path(
            segment,
            method=norm_method,
            target_dbfs=float(target_dbfs),
            boost_db=float(boost_db),
        )
        # norm_path = normalized_audio_path(segment, method=norm_method, target_dbfs=target_dbfs)
        if norm_path:
            src_path = norm_path

    audio = pn.pane.Audio(src_path, autoplay=False, throttle=50)

    #  per-recording volume
    volume_slider = pn.widgets.FloatSlider(
        name="Vol", start=0.0, end=200.0, step=1.0, value=100.0, width=160
    )
    # Instant client-side link: slider -> this audio player's volume
    volume_slider.jslink(audio, value="volume")

    audio_controls = pn.Row(audio, volume_slider, align="center")

    # FIX 2: Correctly define a Stream that accepts an 'x' parameter
    PositionStream = hv.streams.Stream.define('PositionStream', x=t_start)
    position_stream = PositionStream()

    def get_vline(x):
        return hv.VLine(x).opts(color='cyan', line_width=2, line_dash='dashed')

    dmap_vline = hv.DynamicMap(get_vline, streams=[position_stream])
    overlay_elements.append(dmap_vline)

    overlay = hv.Overlay(overlay_elements).opts(shared_axes=False)

    def update_position(event):
        current_time_abs = t_start + event.new
        position_stream.event(x=current_time_abs)
    
    audio.param.watch(update_position, 'time')

    plot_pane = pn.pane.HoloViews(overlay, linked_axes=False)
    # info display (based on original audio, not normalized)
    ysr = load_audio(segment)
    loud_str = ""
    if ysr is not None:
        y0, _sr0 = ysr
        loud_str = f" | RMS: {rms_dbfs(y0):.1f} dBFS | Peak: {20*np.log10(max(peak(y0),1e-12)):.1f} dBFS"

    # old header
    # header = pn.pane.Markdown(
    #     f"**{os.path.basename(segment)}** ({t_start:.2f}s - {t_end:.2f}s){loud_str}",
    #     margin=(0,0,6,0)
    # )

    # header = pn.pane.Markdown(f"**{os.path.basename(segment)}** ({t_start:.2f}s - {t_end:.2f}s)", margin=(0,0,6,0))

    # new header with source file info
    header = pn.pane.Markdown(
        f"**{os.path.basename(segment)}** "
        f"({t_start:.2f}s – {t_end:.2f}s) "
        f"{loud_str}  \n"
        f"src: `{os.path.basename(src_path)}`",
        margin=(0, 0, 6, 0)
    )

    # return pn.Column(header, audio, plot_pane)
    return pn.Column(header, audio_controls, plot_pane)

# ----------------------------
# Widgets & Main
# ----------------------------
df = get_df()
segments = sorted(df.audio_segment_file_name.unique().tolist())
all_labels = sorted(df.label.unique().tolist())
all_cats = sorted(df.label_category.unique().tolist())

segment_pick = pn.widgets.MultiChoice(
    name="Segments", options=segments, value=segments[:2] if len(segments) >= 2 else segments,
)
label_filter = pn.widgets.MultiChoice(name="Labels", options=all_labels)
cat_filter = pn.widgets.MultiChoice(name="Categories", options=all_cats)
color_by_widget = pn.widgets.Select(
    name="Color by (hover fields)", options=["label_category", "label"], value="label_category"
)
vmin_slider = pn.widgets.IntSlider(name="dB floor (vmin)", start=-120, end=-20, value=-80, step=5)
vmax_slider = pn.widgets.IntSlider(name="dB ceiling (vmax)", start=-20, end=20, value=0, step=1)
cmap_widget = pn.widgets.Select(
    name="Spectrogram color", 
    options={"Black": "Greys", "Blue": "Blues", "Green": "Greens", "Red": "Reds", "Purple": "Purples"}, 
    value="Greys"
)

# normalization widgets 
normalize_toggle = pn.widgets.Checkbox(name="Normalize playback", value=False)

norm_method = pn.widgets.Select(
    name="Normalization method",
    options={"RMS (target loudness)": "rms", "Peak (anti-clipping)": "peak"},
    value="rms",
)

target_dbfs = pn.widgets.IntSlider(
    name="Target RMS (dBFS)", start=-40, end=-10, value=-20, step=1
)

boost_db = pn.widgets.IntSlider(
    name="Boost (dB)", start=0, end=24, value=0, step=1
)

random_n_slider = pn.widgets.IntSlider(name="Random segment count", start=1, end=min(20, len(segments)), value=2, step=1)
random_button = pn.widgets.Button(name="Pick random segments", button_type="primary")

def pick_random_segments(event=None):
    n = random_n_slider.value
    if n > len(segments): n = len(segments)
    selected = random.sample(segments, n)
    segment_pick.value = selected

random_button.on_click(pick_random_segments)
info = pn.pane.Alert("Tip: select a few segments (2–8). Too many at once will be slow.", alert_type="info")

@pn.depends(segment_pick, label_filter, cat_filter, color_by_widget, 
            vmin_slider, vmax_slider, cmap_widget,
            normalize_toggle, norm_method, target_dbfs, boost_db)
def grid_view(seg_list, labels, cats, color_by, vmin, vmax, cmap, do_norm, nmeth, tdb, bdb):
    if not seg_list: return pn.pane.Markdown("Select one or more segments.")
    seg_list = seg_list[:MAX_VIEWS]
    cards = [make_view(s, labels=labels, cats=cats, color_by=color_by, 
                       vmin=vmin, vmax=vmax, cmap=cmap,
                       do_normalize=do_norm, norm_method=nmeth,
                       target_dbfs=tdb, boost_db=bdb) for s in seg_list]
    return pn.GridBox(*cards, ncols=2, sizing_mode="stretch_width")

app = pn.template.FastListTemplate(
    title="ECOSoundSet Explorer",
    sidebar=[
        info, segment_pick, random_n_slider, random_button, 
        cat_filter, label_filter, color_by_widget, 
        vmin_slider, vmax_slider, cmap_widget,
        normalize_toggle, norm_method, target_dbfs, boost_db,
    ],
    main=[grid_view],
)

app.servable()