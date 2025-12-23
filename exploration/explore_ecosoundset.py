import os
from functools import lru_cache

import numpy as np
import pandas as pd
import librosa

import panel as pn
import holoviews as hv
import logging

pn.extension("audio")
hv.extension("bokeh")

# ----------------------------
# CONFIG
# ----------------------------
ANNOTATIONS_PATH = "./data/ECOSoundSet/annotated_audio_segments.csv" 
METADATA_PATH = "./data/ECOSoundSet/recording_metadata.csv"
SEP = ","
AUDIO_BASE_DIR = "./data/ECOSoundSet/Split recordings"                  # folder with wav segments
N_FFT = 1024
HOP = 256
MAX_VIEWS = 8                         

HEIGHT = 500
WIDTH = 800
FONT_SIZE = "12pt"
FONT_COLOR = "blue"

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
        
        # Left join annotations with metadata on recording_id
        _df = annotations.merge(metadata, on="recording_id", how="left")
        
        # Create full file path: AUDIO_BASE_DIR / license / audio_segment_file_name
        _df["full_fpath"] = _df.apply(
            lambda row: os.path.join(AUDIO_BASE_DIR, str(row["license"]), row["audio_segment_file_name"]),
            axis=1
        )
    
    return _df


def audio_path(fname: str) -> str:
    """Get full path for audio file from dataframe or fallback to direct path."""
    df = get_df()
    match = df[df.audio_segment_file_name == fname]
    if not match.empty:
        return match.iloc[0]["full_fpath"]
    return fname if os.path.isabs(fname) else os.path.join(AUDIO_BASE_DIR, fname)



def get_segment_time_range(fname: str) -> tuple[float, float]:
    """Get the time range (initial, final) for an audio segment."""
    df = get_df()
    match = df[df.audio_segment_file_name == fname]
    if not match.empty:
        row = match.iloc[0]
        return float(row["audio_segment_initial_time"]), float(row["audio_segment_final_time"])
    return 0.0, 0.0


# ----------------------------
# Caching: audio + spec
# ----------------------------
@lru_cache(maxsize=256)
def load_audio(fname: str) -> tuple[np.ndarray, int] | None:
    """Load audio file and return (samples, sample_rate)."""
    try:
        y, sr = librosa.load(audio_path(fname), sr=None, mono=True)
        return y, sr
    except Exception as e:
        logging.error(f"Failed to load audio {fname}: {e}")
        return None

@lru_cache(maxsize=256)
def spec_cached(fname: str):
    """Compute spectrogram for audio file."""
    result = load_audio(fname)
    if result is None:
        return None, None, None
    y, sr = result
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    # Get segment time offset and shift times accordingly
    t_start, t_end = get_segment_time_range(fname)
    times_relative = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=HOP)
    times = times_relative + t_start  # Offset to absolute position

    return S_db, times, freqs

def make_view(segment: str, labels=None, cats=None, color_by="label_category", vmin=-80, vmax=0, cmap="Greys"):
    df = get_df()
    ann = df[df.audio_segment_file_name == segment].copy()

    if cats:
        ann = ann[ann.label_category.isin(cats)]
    if labels:
        ann = ann[ann.label.isin(labels)]

    S_db, times, freqs = spec_cached(segment)
    
    # Handle missing audio file
    if S_db is None:
        return pn.pane.Markdown(f"**{segment}**: Audio file not found")
    
    # Get time range for x-axis limits
    t_start, t_end = get_segment_time_range(segment)

    # Define explicit bounds: (x_min, y_min, x_max, y_max)
    # bounds = (t_start, freqs.min(), t_end, freqs.max())

    img = hv.Image(
        (times, freqs, S_db),
        # S_db,
        # bounds=bounds,
        kdims=["Time (s)", "Frequency (Hz)"],
        vdims=["dB"],
    ).opts(
        cmap=cmap,
        height=HEIGHT,
        width=WIDTH,
        tools=["hover"],
        colorbar=True,
        colorbar_opts={"title": "dB"},
        xlim=(t_start, t_end),
        clim=(vmin, vmax),
    )

    # Rectangles overlay: (x0, y0, x1, y1)
    rects = [
        (float(r.annotation_initial_time), float(r.annotation_min_freq),
         float(r.annotation_final_time),   float(r.annotation_max_freq),
         str(r.label), str(r.label_category))
        for _, r in ann.iterrows()
    ]

    overlay = img

    if rects:
        boxes = hv.Rectangles(
            rects, kdims=["x0","y0","x1","y1"], vdims=["label","category"]
        ).opts(
            fill_alpha=0.25,
            line_width=2,
            line_color="red",
            tools=["hover"],
            hover_alpha=0.45,
            shared_axes=False,  
        )
    #     overlay = (img * boxes).opts(shared_axes=False)
    # else:
    #     overlay = img.opts(shared_axes=False)

        overlay = overlay * boxes

        # Add text labels for each box
        text_labels = []
        for _, r in ann.iterrows():
            x_pos = (float(r.annotation_initial_time) + float(r.annotation_final_time)) / 2
            y_pos = float(r.annotation_max_freq) + 200  # Slightly above the box
            label_text = f"{r.label} ({r.label_category})"
            text_labels.append((x_pos, y_pos, label_text))
        
        labels_overlay = hv.Labels(text_labels, kdims=["x", "y"], vdims=["text"]).opts(
            text_font_size=FONT_SIZE,
            text_color=FONT_COLOR,
            text_align="center",
            shared_axes=False,
        )
        overlay = overlay * labels_overlay

    overlay = overlay.opts(shared_axes=False)    

    # Wrap in HoloViews pane with linked_axes=False
    plot_pane = pn.pane.HoloViews(overlay, linked_axes=False)

    audio = pn.pane.Audio(audio_path(segment), autoplay=False)

    header = pn.pane.Markdown(f"**{os.path.basename(segment)}** ({t_start:.2f}s - {t_end:.2f}s)", margin=(0,0,6,0))

    return pn.Column(header, audio, plot_pane)

# ----------------------------
# Widgets
# ----------------------------
df = get_df()  # Add this line
segments = sorted(df.audio_segment_file_name.unique().tolist())
all_labels = sorted(df.label.unique().tolist())
all_cats = sorted(df.label_category.unique().tolist())

segment_pick = pn.widgets.MultiChoice(
    name="Segments",
    options=segments,
    value=segments[:2] if len(segments) >= 2 else segments,
)
label_filter = pn.widgets.MultiChoice(name="Labels", options=all_labels)
cat_filter = pn.widgets.MultiChoice(name="Categories", options=all_cats)
color_by_widget = pn.widgets.Select(  # Renamed
    name="Color by (hover fields)", 
    options=["label_category", "label"], 
    value="label_category"
)
vmin_slider = pn.widgets.IntSlider(
    name="dB floor (vmin)",
    start=-120,
    end=-20,
    value=-80,
    step=5,
)

vmax_slider = pn.widgets.IntSlider(
    name="dB ceiling (vmax)",
    start=-20,
    end=20,
    value=0,
    step=1,
)

cmap_options = {
    "Black": "Greys",
    "Blue": "Blues",
    "Green": "Greens",
    "Red": "Reds",
    "Purple": "Purples",
    "Orange": "Oranges",
}
cmap_widget = pn.widgets.Select(
    name="Spectrogram color (vmax)",
    options=cmap_options,
    value="Greys"
)

info = pn.pane.Alert(
    "Tip: select a few segments (2–8). Too many at once will be slow.",
    alert_type="info"
)

@pn.depends(segment_pick, label_filter, cat_filter, color_by_widget, vmin_slider, vmax_slider, cmap_widget)  # Updated
def grid_view(seg_list, labels, cats, color_by, vmin, vmax, cmap):  # Parameter name unchanged
    if not seg_list:
        return pn.pane.Markdown("Select one or more segments.")
    seg_list = seg_list[:MAX_VIEWS]

    cards = [make_view(s, labels=labels, cats=cats, color_by=color_by, vmin=vmin, vmax=vmax, cmap=cmap) for s in seg_list]
    return pn.GridBox(*cards, ncols=2, sizing_mode="stretch_width")

# ----------------------------
# App Layout
# ----------------------------
app = pn.template.FastListTemplate(
    title="ECOSoundSet Explorer - Spectrogram + Playback",
    sidebar=[
        info,
        segment_pick,
        cat_filter,
        label_filter,
        color_by_widget,  # Updated
        vmin_slider,
        vmax_slider,
        cmap_widget,
        pn.pane.Markdown("---\n**Performance**\n- Caches audio+spectrogram in memory\n- Increase MAX_VIEWS if your node is strong"),
    ],
    main=[grid_view],
)

app.servable()