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
ANNOTATIONS_PATH = "/cfs/earth/scratch/peeb/projects/chirpscan_preprocess/data/ECOSoundSet/annotated_audio_segments.csv" 
METADATA_PATH = "/cfs/earth/scratch/peeb/projects/chirpscan_preprocess/data/ECOSoundSet/recording_metadata.csv"
SEP = ","
AUDIO_BASE_DIR = "/cfs/earth/scratch/peeb/projects/chirpscan_preprocess/data/ECOSoundSet/Split recordings"                  # folder with wav segments
N_FFT = 1024
HOP = 256
MAX_VIEWS = 8                         



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
    times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=HOP)
    return S_db, times, freqs

def make_view(segment: str, labels=None, cats=None, color_by="label_category"):
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

    img = hv.Image(
        (times, freqs, S_db),
        kdims=["Time (s)", "Frequency (Hz)"],
        vdims=["dB"],
    ).opts(
        cmap="Viridis",
        height=260,
        width=480,
        tools=["hover"],
        colorbar=False,
    )

    # Rectangles overlay: (x0, y0, x1, y1)
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
            fill_alpha=0.25,
            line_width=2,
            tools=["hover"],
            hover_alpha=0.45,
        )
        overlay = img * boxes
    else:
        overlay = img

    audio = pn.pane.Audio(audio_path(segment), autoplay=False)

    header = pn.pane.Markdown(f"**{os.path.basename(segment)}**", margin=(0,0,6,0))

    return pn.Column(header, audio, overlay)

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

info = pn.pane.Alert(
    "Tip: select a few segments (2–8). Too many at once will be slow.",
    alert_type="info"
)

@pn.depends(segment_pick, label_filter, cat_filter, color_by_widget)  # Updated
def grid_view(seg_list, labels, cats, color_by):  # Parameter name unchanged
    if not seg_list:
        return pn.pane.Markdown("Select one or more segments.")
    seg_list = seg_list[:MAX_VIEWS]

    cards = [make_view(s, labels=labels, cats=cats, color_by=color_by) for s in seg_list]
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
        pn.pane.Markdown("---\n**Performance**\n- Caches audio+spectrogram in memory\n- Increase MAX_VIEWS if your node is strong"),
    ],
    main=[grid_view],
)

app.servable()