from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import os
import sys
import shutil
import json
import yaml

import torch
import faiss
import librosa
import numpy as np
import pandas as pd

try:
    import pyrubberband as pyrb
    PYRB_OK = True
except ImportError:
    PYRB_OK = False

# =========================================================
# PATH GraFP — repo harus ada di folder GraFP/
# =========================================================

GRAFP_DIR = os.path.join(os.path.dirname(__file__), "GraFP")
sys.path.insert(0, GRAFP_DIR)

from encoder.graph_encoder import GraphEncoder
from simclr.simclr import SimCLR
from modules.transformations import GPUTransformNeuralfp  # WAJIB untuk preprocessing

# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(title="GraFPrint Royalti API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# DEVICE
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================================================
# PATH FILE — semua taruh 1 folder dengan app.py
# =========================================================

MODEL_PATH   = "finetune_best.pth"
CONFIG_PATH  = os.path.join(GRAFP_DIR, "config", "grafp.yaml")
FAISS_PATH   = "faiss.index"
REF_EMB_PATH = "ref_embeddings_finetuned.npy"
REF_IDS_PATH = "ref_track_ids_finetuned.json"
TRACKS_CSV   = "tracks_small.csv"
UPLOAD_DIR   = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# =========================================================
# LOAD CONFIG (pakai yaml biasa, bukan load_config)
# =========================================================

print("Loading config...")
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

print(f"Config: n_mels={cfg['n_mels']}, n_frames={cfg['n_frames']}, d={cfg['d']}")

# =========================================================
# SAMPLE RATE & KONSTANTA dari config
# =========================================================

SAMPLE_RATE    = cfg.get("fs", 16000)
QUERY_DURATION = 10.0   # detik — sama dengan notebook
AUDIO_OFFSET   = 0.3    # crop dari 30% durasi lagu

# Global Threshold dari evaluasi
GLOBAL_THRESHOLD = 0.75

# =========================================================
# BUILD & LOAD MODEL
# =========================================================

print("Building model...")
encoder = GraphEncoder(cfg=cfg, in_channels=cfg["n_filters"], k=3)
model   = SimCLR(cfg, encoder=encoder).to(device)

print("Loading checkpoint...")
ckp   = torch.load(MODEL_PATH, map_location=device, weights_only=False)
state = ckp["state_dict"] if "state_dict" in ckp else ckp
state = {k.replace("module.", ""): v for k, v in state.items()}
model.load_state_dict(state, strict=False)
model.eval()
print("Model loaded!")

# =========================================================
# AUGMENTOR — WAJIB untuk preprocessing (tanpa noise/IR)
# =========================================================

print("Building augmentor...")
augment = GPUTransformNeuralfp(
    cfg=cfg,
    ir_dir=None,
    noise_dir=None,
    train=False
).to(device)
print("Augmentor ready!")

# =========================================================
# LOAD FAISS INDEX
# =========================================================

print("Loading FAISS index...")
index = faiss.read_index(FAISS_PATH)
print(f"FAISS loaded: {index.ntotal} vectors")

# =========================================================
# LOAD REFERENCE DATA
# =========================================================

print("Loading reference embeddings...")
ref_emb = np.load(REF_EMB_PATH).astype("float32")

with open(REF_IDS_PATH) as f:
    ref_ids = json.load(f)

print(f"Reference: {len(ref_ids)} tracks, dim={ref_emb.shape[1]}")

# =========================================================
# LOAD METADATA FMA (tracks.csv multi-level header)
# =========================================================

print("Loading metadata...")
print("Loading metadata...")

METADATA_OK = False

try:
    df_tracks = pd.read_csv(TRACKS_CSV)

    metadata_dict = {}

    for _, row in df_tracks.iterrows():
        metadata_dict[str(row["track_id"])] = {
            "title": row["title"],
            "artist": row["artist"],
            "license": row["license"],
            "genre": row["genre"]
        }

    METADATA_OK = True
    print(f"Metadata loaded: {len(metadata_dict)} tracks")

except Exception as e:
    print("Metadata load error:", e)

def get_track_info(track_id):
    if not METADATA_OK:
        return {
            "title": "Unknown",
            "artist": "Unknown",
            "license": "Unknown",
            "genre": "Unknown"
        }

    tid = str(int(track_id))

    return metadata_dict.get(
        tid,
        {
            "title": "Unknown",
            "artist": "Unknown",
            "license": "Unknown",
            "genre": "Unknown"
        }
    )


# =========================================================
# FUNGSI: EXTRACT EMBEDDING — sama persis dengan notebook
# =========================================================

def get_embedding(audio_np: np.ndarray, max_batch: int = 128):
    """
    Extract embedding 128-dim menggunakan SimCLR + GPUTransformNeuralfp.
    Pipeline ini sama persis dengan cell 5 notebook.
    """
    audio_np = np.clip(np.nan_to_num(audio_np), -1.0, 1.0)
    tensor   = torch.FloatTensor(audio_np).to(device)

    with torch.no_grad():
        x_i, _ = augment(tensor, None)
        if x_i is None or x_i.shape[0] == 0:
            return None

        embs = []
        for x in torch.split(x_i, max_batch, dim=0):
            _, _, z, _ = model(x, x)
            z_np = z.detach().cpu().numpy()
            if not (np.isnan(z_np).any() or np.isinf(z_np).any()):
                embs.append(z_np)

    if not embs:
        return None

    emb  = np.concatenate(embs).mean(axis=0)
    norm = np.linalg.norm(emb)
    if norm < 1e-8:
        return None
    return (emb / norm).astype("float32")

# =========================================================
# FUNGSI: MANIPULASI AUDIO — sama dengan cell 5 notebook
# =========================================================

PITCH_STEPS = [-2, -1, 0, 1, 2]          # semitones
TIME_RATES  = [0.8, 0.9, 1.0, 1.1, 1.2]  # faktor kecepatan


def pitch_shift(audio: np.ndarray, sr: int, n_steps: int) -> np.ndarray:
    """Pitch shifting -2 sampai +2 semitone."""
    if n_steps == 0:
        return audio.copy()
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=float(n_steps))


def time_stretch(audio: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """Time stretching 0.8x sampai 1.2x dengan looping (tiling) untuk kompensasi durasi."""
    if rate == 1.0:
        return audio.copy()
    try:
        out = pyrb.time_stretch(audio, sr, rate) if PYRB_OK \
              else librosa.effects.time_stretch(audio, rate=rate)
        
        tgt = len(audio)
        
        # Jika audio memanjang (rate < 1.0), potong sisanya
        if len(out) >= tgt:
            return out[:tgt]
        # Jika audio memendek (rate > 1.0), lakukan looping (tile)
        else:
            repeats = int(np.ceil(tgt / len(out)))
            return np.tile(out, repeats)[:tgt]
            
    except Exception:
        return audio.copy()


def run_manipulation_search(audio_np: np.ndarray, ref_emb_original: np.ndarray) -> dict:
    """
    Jalankan semua variasi pitch shifting dan time stretching,
    hitung cosine similarity tiap variasi terhadap embedding original,
    dan cari top-1 match di FAISS.
    """
    sr = SAMPLE_RATE

    # Embedding audio original (referensi untuk hitung similarity)
    emb_ori = get_embedding(audio_np)

    pitch_results = {}
    for n in PITCH_STEPS:
        label = f"{n:+d} semitone"
        try:
            audio_mod = pitch_shift(audio_np, sr, n)
            emb_mod   = get_embedding(audio_mod)
            if emb_mod is None:
                pitch_results[label] = {"similarity": None, "match": False, "top_track": None}
                continue

            # Cosine similarity terhadap embedding original
            sim = float(np.dot(emb_ori, emb_mod)) if emb_ori is not None else None

            # FAISS search
            qf = emb_mod.reshape(1, -1).copy()
            faiss.normalize_L2(qf)
            scores, indices = index.search(qf, 1)
            top_idx    = indices[0][0]
            top_score  = round(float(scores[0][0]), 4)
            top_tid    = str(ref_ids[top_idx]).zfill(6) if top_idx >= 0 else None
            top_info   = get_track_info(top_tid) if top_tid else None

            pitch_results[label] = {
                "similarity"       : round(sim, 4) if sim is not None else None,
                "faiss_top_score"  : top_score,
                "match"            : top_score >= GLOBAL_THRESHOLD,
                "top_track"        : {
                    "track_id": top_tid,
                    "title"   : top_info["title"] if top_info else "Unknown",
                    "artist"  : top_info["artist"] if top_info else "Unknown",
                    "license" : top_info["license"] if top_info else "Unknown",
                } if top_tid else None,
            }
        except Exception as e:
            pitch_results[label] = {"similarity": None, "match": False,
                                     "top_track": None, "error": str(e)}

    time_results = {}
    for rate in TIME_RATES:
        label = f"{rate:.1f}x"
        try:
            audio_mod = time_stretch(audio_np, sr, rate)
            emb_mod   = get_embedding(audio_mod)
            if emb_mod is None:
                time_results[label] = {"similarity": None, "match": False, "top_track": None}
                continue

            sim = float(np.dot(emb_ori, emb_mod)) if emb_ori is not None else None

            qf = emb_mod.reshape(1, -1).copy()
            faiss.normalize_L2(qf)
            scores, indices = index.search(qf, 1)
            top_idx   = indices[0][0]
            top_score = round(float(scores[0][0]), 4)
            top_tid   = str(ref_ids[top_idx]).zfill(6) if top_idx >= 0 else None
            top_info  = get_track_info(top_tid) if top_tid else None

            time_results[label] = {
                "similarity"       : round(sim, 4) if sim is not None else None,
                "faiss_top_score"  : top_score,
                "match"            : top_score >= GLOBAL_THRESHOLD,
                "top_track"        : {
                    "track_id": top_tid,
                    "title"   : top_info["title"] if top_info else "Unknown",
                    "artist"  : top_info["artist"] if top_info else "Unknown",
                    "license" : top_info["license"] if top_info else "Unknown",
                } if top_tid else None,
            }
        except Exception as e:
            time_results[label] = {"similarity": None, "match": False,
                                    "top_track": None, "error": str(e)}

    return {
        "pitch_shifting" : pitch_results,
        "time_stretching": time_results,
    }


# =========================================================
# FUNGSI: LICENSE -> KEPUTUSAN ROYALTI
# =========================================================

LICENSE_CATEGORY = {
    "cc by-nd"    : "KETAT",
    "cc by-nc-nd" : "KETAT",
    "cc by-nc"    : "MODERAT",
    "cc by-nc-sa" : "MODERAT",
    "cc by"       : "BEBAS",
    "cc by-sa"    : "BEBAS",
    "cc0"         : "PUBLIC DOMAIN",
    "public domain": "PUBLIC DOMAIN",
}

LICENSE_VERDICT = {
    "KETAT"          : {"status": "Kena Royalti",         "category": "strict"},
    "MODERAT"        : {"status": "Perlu Review",          "category": "moderate"},
    "BEBAS"          : {"status": "Aman (Wajib Atribusi)", "category": "free"},
    "PUBLIC DOMAIN"  : {"status": "Bebas Total",           "category": "free"},
    "TIDAK DIKETAHUI": {"status": "Perlu Verifikasi",      "category": "unknown"},
}


def license_decision(license_str: str) -> dict:
    if not isinstance(license_str, str) or license_str in ("Unknown", "nan", ""):
        return LICENSE_VERDICT["TIDAK DIKETAHUI"]
    lic = license_str.lower()
    for key, cat in LICENSE_CATEGORY.items():
        if key in lic:
            return LICENSE_VERDICT[cat]
    return LICENSE_VERDICT["TIDAK DIKETAHUI"]

# =========================================================
# ENDPOINT: HOME
# =========================================================

@app.get("/")
def home():
    return {
        "message"      : "GraFPrint Royalti API",
        "device"       : str(device),
        "threshold"    : GLOBAL_THRESHOLD,
        "faiss_vectors": index.ntotal,
        "sample_rate"  : SAMPLE_RATE,
    }

# =========================================================
# ENDPOINT: CEK ROYALTI
# =========================================================

@app.post("/search")
async def search_audio(file: UploadFile = File(...)):

    # Validasi format
    allowed_ext = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in allowed_ext:
        raise HTTPException(
            status_code=400,
            detail=f"Format tidak didukung: {ext}. Gunakan: {', '.join(allowed_ext)}"
        )

    # Simpan file sementara
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load audio — crop dari 30% durasi sama seperti notebook
        total_dur   = librosa.get_duration(path=save_path)
        offset      = total_dur * AUDIO_OFFSET
        audio_np, _ = librosa.load(
            save_path, sr=SAMPLE_RATE,
            offset=offset, duration=QUERY_DURATION, mono=True
        )
        audio_np = audio_np.astype(np.float32)

        # Extract embedding
        emb = get_embedding(audio_np)
        if emb is None:
            raise HTTPException(status_code=500, detail="Gagal mengekstrak embedding audio.")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses audio: {str(e)}")
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)

    # ── FAISS search audio original (top-5) ──────────────────
    query_f = emb.reshape(1, -1).copy()
    faiss.normalize_L2(query_f)
    scores, indices = index.search(query_f, 5)

    top5 = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        if idx < 0 or idx >= len(ref_ids):
            continue
        track_id = str(ref_ids[idx]).zfill(6)
        info     = get_track_info(track_id)
        top5.append({
            "rank"      : rank,
            "track_id"  : track_id,
            "title"     : info["title"],
            "artist"    : info["artist"],
            "genre"     : info["genre"],
            "license"   : info["license"],
            "similarity": round(float(score), 4),
        })

    if not top5:
        raise HTTPException(status_code=500, detail="Tidak ada hasil dari FAISS.")

    top            = top5[0]
    similarity_top = top["similarity"]

    # ── Keputusan royalti audio original ─────────────────────
    if similarity_top < GLOBAL_THRESHOLD:
        decision = {
            "status"     : "Bebas Klaim",
            "category"   : "clear",
            "description": (
                f"Similarity {similarity_top:.4f} di bawah threshold {GLOBAL_THRESHOLD}. "
                "Audio tidak memiliki kemiripan signifikan dengan database."
            ),
        }
    else:
        d = license_decision(top["license"])
        decision = {
            **d,
            "description": (
                f"Similarity {similarity_top:.4f} >= threshold {GLOBAL_THRESHOLD}. "
                f"Audio terdeteksi mirip dengan '{top['title']}' oleh {top['artist']}. "
                f"Status lisensi: {top['license']}."
            ),
        }

    # ── Uji robustness: semua variasi pitch & time ────────────
    manipulation_results = run_manipulation_search(audio_np, ref_emb)

    # Ringkasan robustness: berapa variasi yang masih terdeteksi
    pitch_matches = sum(
        1 for v in manipulation_results["pitch_shifting"].values() if v.get("match")
    )
    time_matches = sum(
        1 for v in manipulation_results["time_stretching"].values() if v.get("match")
    )
    total_variations = len(PITCH_STEPS) + len(TIME_RATES)
    total_matches    = pitch_matches + time_matches

    return {
        "query_file"  : file.filename,
        "threshold"   : GLOBAL_THRESHOLD,

        # Hasil audio original
        "original"    : {
            "top_match": top,
            "decision" : decision,
            "top5"     : top5,
        },

        # Hasil uji robustness per manipulasi
        "robustness"  : {
            "summary": {
                "total_variations"  : total_variations,
                "detected_matches"  : total_matches,
                "pitch_matches"     : f"{pitch_matches}/{len(PITCH_STEPS)}",
                "time_matches"      : f"{time_matches}/{len(TIME_RATES)}",
            },
            "pitch_shifting" : manipulation_results["pitch_shifting"],
            "time_stretching": manipulation_results["time_stretching"],
        },
    }