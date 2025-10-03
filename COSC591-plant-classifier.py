# plant_classification_app_3model.py
# YOLO (PyTorch) + Xception (TFLite two-head) + ResNet50 (Keras .h5)
# UI: Tkinter panels + Target Crop/State dropdowns
# Charts (between panels & log):
#   1) Inference time — last N calls (mean ± std)
#   2) Top-1 confidence — last N calls (mean ± std)
#   3) Load time (STACKED): Package cold total + Model init
#
# This version fixes the Xception TFLite import to work across TF builds:
#   - prefers tf.lite.Interpreter (full TensorFlow)
#   - falls back to tflite_runtime.interpreter.Interpreter when needed

import os
import sys
import json
import time
import threading
import importlib
import traceback
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ----------------------------
# Tunables
# ----------------------------
METRIC_WINDOW = 10  # rolling window size for charts

# ----------------------------
# Model Paths (EDIT IF NEEDED)
# ----------------------------

# ----------------------------
# Model base dir (relative to this file)
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------
# Model Paths (relative to this file)
# ----------------------------

# YOLO (Ultralytics, PyTorch)
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "models", "yolo", "best.pt")

# Xception (TFLite two-head)
XCEPTION_VIC_DIR      = os.path.join(BASE_DIR, "models", "xception")
XCEPTION_TFLITE       = os.path.join(XCEPTION_VIC_DIR, "xception_pv1.tflite")
XCEPTION_LABELS_CROP  = os.path.join(XCEPTION_VIC_DIR, "labels_crop.txt")
XCEPTION_LABELS_STATE = os.path.join(XCEPTION_VIC_DIR, "labels_state.txt")

# ResNet-50 (Keras/TensorFlow .h5)
RESNET50_KERAS_MODEL = os.path.join(BASE_DIR, "models", "resnet50", "resnet50_stage2_conv5.h5")
RESNET50_CLASS_CSV   = os.path.join(os.path.dirname(RESNET50_KERAS_MODEL), "class_index.csv")

# Fallback class map (old JSON format) if CSV not available
RESNET_DIR          = os.path.dirname(RESNET50_KERAS_MODEL)
RESNET_CLASSES_JSON = os.path.join(RESNET_DIR, "class_names.json")


# ----------------------------
# Import timing registry
# ----------------------------
_IMPORT_REGISTRY = {}  # alias -> {'cold_time': float, 'imported_by': str}
_REGISTRY_LOCK = threading.Lock()

def _timed_import(module_spec: str, alias: str, imported_by: str) -> float:
    t0 = time.perf_counter()
    already = alias in _IMPORT_REGISTRY or module_spec in sys.modules
    try:
        importlib.import_module(module_spec)
    finally:
        dt = time.perf_counter() - t0
        if not already:
            with _REGISTRY_LOCK:
                if alias not in _IMPORT_REGISTRY:
                    _IMPORT_REGISTRY[alias] = {'cold_time': dt, 'imported_by': imported_by}
    return dt

def _cold_cost(alias: str) -> float:
    with _REGISTRY_LOCK:
        rec = _IMPORT_REGISTRY.get(alias)
        return float(rec['cold_time']) if rec else 0.0

# ----------------------------
# Helpers (label parsing & formatting)
# ----------------------------
def _clean_spaces(s: str) -> str:
    return " ".join(s.replace("_", " ").split()).strip()

def _title(s: str) -> str:
    return " ".join([w.capitalize() if (w.isupper() or w.islower()) else w for w in s.split()])

def _prefer_parenthetical_for_crop(s: str) -> str:
    if "(" in s and ")" in s:
        inner = s[s.find("(")+1:s.find(")")].strip()
        if inner:
            return _title(inner)
    return _title(s)

def split_combined_label(name: str) -> Tuple[str, str]:
    """Split labels like 'Corn_(maize)__Common_rust' or 'Apple___Cedar_apple_rust' into (crop, state)."""
    raw = name.strip().strip("_")
    if "___" in raw:
        a, b = raw.split("___", 1)
    elif "__" in raw:
        a, b = raw.split("__", 1)
    else:
        import re
        parts = re.split(r"_{2,}", raw, maxsplit=1)
        if len(parts) == 1:
            crop = _prefer_parenthetical_for_crop(_clean_spaces(parts[0]))
            return crop, ""
        a, b = parts[0], parts[1]
    crop = _prefer_parenthetical_for_crop(_clean_spaces(a))
    state = _title(_clean_spaces(b))
    return crop, state

def _norm_key(s: str) -> str:
    return "".join(ch for ch in s.lower().replace("_", " ").strip() if ch.isalnum() or ch == " ")

def fmt_pct(p: float) -> str:
    return f"{p*100:.2f}%"

def pick_torch_device_preference():
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            return 'mps'
    except Exception:
        pass
    return 'cpu'

# ----------------------------
# Model Wrappers
# ----------------------------
class YoloClassifier:
    def __init__(self, model_path: str):
        self.pkg_time_this_call = 0.0
        self.pkg_time_this_call += _timed_import('torch', 'torch', 'YOLOv11')
        self.pkg_time_this_call += _timed_import('ultralytics', 'ultralytics', 'YOLOv11')

        from ultralytics import YOLO
        start = time.perf_counter()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO weights not found at: {model_path}")
        self.model = YOLO(model_path)
        self.device_pref = pick_torch_device_preference()
        self.model_init_time = time.perf_counter() - start
        self.pkg_cold_total = _cold_cost('torch') + _cold_cost('ultralytics')

    @property
    def name(self):
        return "YOLOv11"

    def classes(self):
        return self.model.names

    def predict(self, image_path: str, target_crop: Optional[str] = None, target_state: Optional[str] = None):
        start = time.perf_counter()
        device_arg = 0 if self.device_pref == 'cuda' else ('mps' if self.device_pref == 'mps' else 'cpu')
        results = self.model.predict(image_path, device=device_arg, verbose=False)
        infer_secs = time.perf_counter() - start

        res = results[0]
        probs = res.probs
        names = res.names

        top1_idx = int(probs.top1)
        top1_conf = float(probs.top1conf)
        top1_name = names.get(top1_idx, str(top1_idx))
        top_crop, top_state = split_combined_label(top1_name)

        # Target confidences (sum across all classes with same crop/state)
        target_crop_conf = None
        target_state_conf = None
        try:
            prob_vec = probs.data
            if hasattr(prob_vec, "cpu"):
                prob_vec = prob_vec.cpu().numpy()
            elif not isinstance(prob_vec, np.ndarray):
                prob_vec = prob_vec.numpy()
            if target_crop:
                want = _norm_key(target_crop)
                target_crop_conf = float(sum(prob_vec[idx] for idx, nm in names.items()
                                             if _norm_key(split_combined_label(nm)[0]) == want))
            if target_state:
                want = _norm_key(target_state)
                target_state_conf = float(sum(prob_vec[idx] for idx, nm in names.items()
                                              if _norm_key(split_combined_label(nm)[1]) == want))
        except Exception:
            pass

        top5_idx = [int(i) for i in probs.top5]
        top5_conf = [float(c) for c in probs.top5conf]
        top5 = [(names.get(i, str(i)), c) for i, c in zip(top5_idx, top5_conf)]

        return {
            'model': self.name,
            'device': self.device_pref,
            'infer_secs': infer_secs,
            'top1_name': top1_name,
            'top1_conf': top1_conf,
            'top5': top5,
            'parsed_top1': {'crop': top_crop, 'state': top_state},
            'target_crop_conf': target_crop_conf,
            'target_state_conf': target_state_conf,
            'pkg_time_this_call': self.pkg_time_this_call,
            'pkg_cold_total': self.pkg_cold_total,
            'model_init_time': self.model_init_time,
        }

class XceptionClassifier:
    EXPECTED_OUTPUT_NAMES = ["crop_output", "state_output"]

    def __init__(self, tflite_path: str, labels_crop_path: str, labels_state_path: str):
        # Labels
        if not os.path.isfile(labels_crop_path) or not os.path.isfile(labels_state_path):
            raise FileNotFoundError("Xception labels not found. Expected:\n  {}\n  {}".format(labels_crop_path, labels_state_path))
        self.crop_labels  = self._load_labels(labels_crop_path)
        self.state_labels = self._load_labels(labels_state_path)
        self.class_names = list(self.crop_labels)  # for dropdowns

        # ---- package imports (prefer full TF; fallback to tflite-runtime) ----
        self.pkg_time_this_call = 0.0
        Interpreter = None
        used_tf = False

        # Try full TensorFlow: use tf.lite.Interpreter (works across many builds)
        try:
            self.pkg_time_this_call += _timed_import('tensorflow', 'tensorflow', 'Xception')
            import tensorflow as tf  # noqa
            Interpreter = tf.lite.Interpreter  # type: ignore[attr-defined]
            used_tf = True
        except Exception:
            Interpreter = None
            used_tf = False

        # Fallback: tflite-runtime
        if Interpreter is None:
            try:
                self.pkg_time_this_call += _timed_import('tflite_runtime.interpreter', 'tflite_runtime', 'Xception')
                from tflite_runtime.interpreter import Interpreter  # type: ignore
                used_tf = False
            except Exception as e:
                raise SystemExit(
                    "No TFLite interpreter found.\n"
                    "Install one of:\n"
                    "  pip install tensorflow  (then uses tf.lite.Interpreter)\n"
                    "  or\n"
                    "  pip install tflite-runtime"
                ) from e

        self.pkg_cold_total = (_cold_cost('tensorflow') if used_tf else _cold_cost('tflite_runtime'))
        self.device_descr = "TensorFlow Lite (TF runtime)" if used_tf else "TensorFlow Lite (tflite-runtime)"

        # ---- load interpreter safely (copy to temp to avoid path quirks) ----
        import shutil, tempfile
        if not os.path.isfile(tflite_path) or os.stat(tflite_path).st_size < 1024:
            raise ValueError(f"Xception TFLite file is missing or too small: {tflite_path}")
        tmpdir = tempfile.mkdtemp(prefix="tflite_")
        tmp_model = os.path.join(tmpdir, "model.tflite")
        shutil.copy2(tflite_path, tmp_model)

        start = time.perf_counter()
        try:
            self.interpreter = None
            try:
                self.interpreter = Interpreter(model_path=tmp_model)
            except Exception:
                with open(tmp_model, "rb") as f:
                    self.interpreter = Interpreter(model_content=f.read())
        except Exception as e:
            raise RuntimeError(f"Failed to construct TFLite interpreter: {e}") from e

        self.interpreter.allocate_tensors()

        in_details = self.interpreter.get_input_details()
        if len(in_details) != 1:
            raise RuntimeError(f"Expected 1 input tensor, found {len(in_details)}")
        self.in0 = in_details[0]
        self.input_shape = list(self.in0["shape"])
        if len(self.input_shape) != 4 or self.input_shape[0] != 1:
            raise RuntimeError(f"Unexpected input shape for Xception TFLite: {self.input_shape}")
        self.H, self.W = int(self.input_shape[1]), int(self.input_shape[2])
        self.input_dtype = self.in0["dtype"]
        self.is_float_input = (self.input_dtype == np.float32)
        self.in_scale, self.in_zp = self.in0.get("quantization", (0.0, 0))

        self.out_details = self.interpreter.get_output_details()
        if len(self.out_details) < 2:
            raise RuntimeError(f"Expected 2 outputs (crop/state), found {len(self.out_details)}")
        self.crop_idx, self.state_idx = self._map_heads(self.out_details, self.crop_labels, self.state_labels)

        def vec_size(od):
            shp = list(od["shape"]); batch = shp[0] if shp else 1
            return int(np.prod(shp) // batch)
        if vec_size(self.out_details[self.crop_idx]) != len(self.crop_labels):
            raise RuntimeError("Crop label count does not match model output size.")
        if vec_size(self.out_details[self.state_idx]) != len(self.state_labels):
            raise RuntimeError("State label count does not match model output size.")

        self.model_init_time = time.perf_counter() - start

    @property
    def name(self):
        return "Xception (TFLite)"

    def predict(self, image_path: str, target_crop: Optional[str] = None, target_state: Optional[str] = None):
        start_prep = time.perf_counter()
        img = Image.open(image_path).convert("RGB").resize((self.W, self.H), Image.LANCZOS)
        arr = np.array(img)
        if self.is_float_input:
            x = arr.astype(np.float32)[None, ...]  # raw 0..255 float (matches your Dart raw255)
        else:
            if self.input_dtype == np.uint8:
                x = arr.astype(np.uint8)[None, ...]
            elif self.input_dtype == np.int8:
                if self.in_scale == 0.0:
                    raise RuntimeError("Int8 input but zero quantization scale.")
                real = arr.astype(np.float32)[None, ...]
                q = np.round(real / self.in_scale + self.in_zp)
                info = np.iinfo(np.int8); q = np.clip(q, info.min, info.max)
                x = q.astype(np.int8)
            else:
                raise RuntimeError(f"Unsupported input dtype: {self.input_dtype}")
        prep_secs = time.perf_counter() - start_prep

        start = time.perf_counter()
        self.interpreter.set_tensor(self.in0["index"], x)
        self.interpreter.invoke()
        infer_secs = time.perf_counter() - start

        crop_vec  = self._get_real_output(self.out_details[self.crop_idx]).squeeze(0)
        state_vec = self._get_real_output(self.out_details[self.state_idx]).squeeze(0)

        crop_probs  = self._softmax_if_needed(crop_vec)
        state_probs = self._softmax_if_needed(state_vec)

        def topk(probs, labels, k=5):
            idx = probs.argsort()[-k:][::-1]
            return [(labels[i], float(probs[i])) for i in idx]

        ci = int(crop_probs.argmax())
        si = int(state_probs.argmax())

        crop_out = {
            "top1_name": self.crop_labels[ci],
            "top1_conf": float(crop_probs[ci]),
            "top3": topk(crop_probs, self.crop_labels, 3),
            "top5": topk(crop_probs, self.crop_labels, 5),
        }
        state_out = {
            "top1_name": self.state_labels[si],
            "top1_conf": float(state_probs[si]),
            "top3": topk(state_probs, self.state_labels, 3),
            "top5": topk(state_probs, self.state_labels, 5),
        }

        target_crop_conf = None
        if target_crop:
            idx = {n.lower(): i for i, n in enumerate(self.crop_labels)}.get(target_crop.lower())
            if idx is not None:
                target_crop_conf = float(crop_probs[idx])
        target_state_conf = None
        if target_state:
            idx = {n.lower(): i for i, n in enumerate(self.state_labels)}.get(target_state.lower())
            if idx is not None:
                target_state_conf = float(state_probs[idx])

        return {
            'model': self.name,
            'device': self.device_descr,
            'prep_secs': prep_secs,
            'infer_secs': infer_secs,
            'top1_name': crop_out["top1_name"],
            'top1_conf': crop_out["top1_conf"],
            'top5': crop_out["top5"],
            'crop': crop_out,
            'state': state_out,
            'target_crop_conf': target_crop_conf,
            'target_state_conf': target_state_conf,
            'pkg_time_this_call': self.pkg_time_this_call,
            'pkg_cold_total': self.pkg_cold_total,
            'model_init_time': self.model_init_time,
        }

    # --- helpers ---
    def _load_labels(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]

    def _map_heads(self, outs, crop_labels, state_labels):
        def find_name(want: str):
            w = want.lower()
            for i, od in enumerate(outs):
                name = (od.get("name") or "").lower()
                if w in name:
                    return i
            return -1
        crop_idx = find_name(self.EXPECTED_OUTPUT_NAMES[0])
        state_idx = find_name(self.EXPECTED_OUTPUT_NAMES[1])
        if crop_idx == -1 or state_idx == -1:
            for i, od in enumerate(outs):
                n = (od.get("name") or "").lower()
                if "crop" in n:  crop_idx = i
                if "state" in n: state_idx = i

        def vec_size(i):
            shp = list(outs[i]["shape"]); batch = shp[0] if shp else 1
            return int(np.prod(shp) // batch)
        if crop_idx == -1 or state_idx == -1:
            for i in range(len(outs)):
                if vec_size(i) == len(crop_labels):  crop_idx  = i
                if vec_size(i) == len(state_labels): state_idx = i
        if crop_idx == -1 or state_idx == -1:
            crop_idx, state_idx = 0, 1
        return crop_idx, state_idx

    def _get_real_output(self, odetail: dict):
        arr = self.interpreter.get_tensor(odetail["index"])
        if np.issubdtype(odetail["dtype"], np.integer):
            scale, zp = odetail.get("quantization", (0.0, 0))
            if scale == 0.0:
                raise RuntimeError("Integer output but zero quantization scale.")
            return scale * (arr.astype(np.float32) - zp)
        return arr.astype(np.float32)

    def _softmax_if_needed(self, x):
        x = x.astype(np.float32)
        s = x.sum(axis=-1, keepdims=True)
        if np.all(x >= -1e-6) and np.all(x <= 1.0 + 1e-6) and np.all(np.abs(s - 1.0) < 1e-3):
            return x
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)

class ResNetClassifier:
    """
    Keras/TensorFlow ResNet-50 classifier (.h5), using tf.keras.applications.resnet50 preprocess.
    Replaces the old PyTorch ResNet with identical interface for the UI.
    """
    def __init__(self, keras_model_path: str, class_csv_path: str, json_fallback_path: str):
        # time TensorFlow import
        self.pkg_time_this_call = 0.0
        self.pkg_time_this_call += _timed_import('tensorflow', 'tensorflow', 'ResNetKeras')

        from tensorflow.keras.models import load_model
        from tensorflow.keras.applications.resnet50 import preprocess_input  # noqa
        self.preprocess_input = preprocess_input

        # Load class map: CSV (idx,label) no header → fallback JSON
        labels = None
        if os.path.isfile(class_csv_path):
            import csv
            id_to_class = {}
            with open(class_csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue
                    try:
                        idx = int(row[0]); lbl = row[1].strip()
                        id_to_class[idx] = lbl
                    except Exception:
                        continue
            if id_to_class:
                max_idx = max(id_to_class.keys())
                labels = [id_to_class.get(i, f"class_{i}") for i in range(max_idx + 1)]
        if labels is None:
            if not os.path.isfile(json_fallback_path):
                raise FileNotFoundError(
                    f"Class map not found. Provide CSV at {class_csv_path} or JSON at {json_fallback_path}"
                )
            with open(json_fallback_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                labels = [str(x) for x in data]
            elif isinstance(data, dict):
                try:
                    items = sorted(((int(k), v) for k, v in data.items()), key=lambda x: x[0])
                    labels = [str(v) for _, v in items]
                except Exception:
                    labels = [str(v) for v in data.values()]
            else:
                raise ValueError("Unsupported class map format (expect CSV or JSON).")
        self.class_names = labels

        if not os.path.isfile(keras_model_path):
            raise FileNotFoundError(f"Keras ResNet-50 model not found: {keras_model_path}")

        start = time.perf_counter()
        self.model = load_model(keras_model_path)
        self.model_init_time = time.perf_counter() - start
        self.pkg_cold_total = _cold_cost('tensorflow')
        self.device_pref = 'TensorFlow (Keras)'

    @property
    def name(self):
        return "ResNet50 (Keras)"

    def predict(self, image_path: str, target_crop: Optional[str] = None, target_state: Optional[str] = None):
        from tensorflow.keras.utils import img_to_array

        img = Image.open(image_path).convert('RGB').resize((224, 224), Image.LANCZOS)
        x = img_to_array(img)[None, ...]
        x = self.preprocess_input(x)

        start = time.perf_counter()
        probs = self.model.predict(x, verbose=0)[0].astype(float)
        infer_secs = time.perf_counter() - start

        top1_idx = int(np.argmax(probs))
        top1_conf = float(probs[top1_idx])
        top1_name = self.class_names[top1_idx]
        top_crop, top_state = split_combined_label(top1_name)

        top5_idx = np.argsort(probs)[::-1][:5]
        top5 = [(self.class_names[int(i)], float(probs[int(i)])) for i in top5_idx]

        target_crop_conf = None
        if target_crop:
            want = _norm_key(target_crop)
            target_crop_conf = float(sum(probs[i] for i, nm in enumerate(self.class_names)
                                         if _norm_key(split_combined_label(nm)[0]) == want))
        target_state_conf = None
        if target_state:
            want = _norm_key(target_state)
            target_state_conf = float(sum(probs[i] for i, nm in enumerate(self.class_names)
                                          if _norm_key(split_combined_label(nm)[1]) == want))

        return {
            'model': self.name,
            'device': self.device_pref,
            'infer_secs': infer_secs,
            'top1_name': top1_name,
            'top1_conf': top1_conf,
            'top5': top5,
            'parsed_top1': {'crop': top_crop, 'state': top_state},
            'target_crop_conf': target_crop_conf,
            'target_state_conf': target_state_conf,
            'pkg_time_this_call': self.pkg_time_this_call,
            'pkg_cold_total': self.pkg_cold_total,
            'model_init_time': self.model_init_time,
        }

# ----------------------------
# UI App
# ----------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Plant Classifier – YOLO / Xception / ResNet")
        self.configure(bg='white')
        try:
            self.attributes('-fullscreen', True)
        except Exception:
            try: self.state('zoomed')
            except Exception: pass
        self.bind('<Escape>', lambda e: self.attributes('-fullscreen', False))
        self.bind('<F11>', lambda e: self.attributes('-fullscreen', not self.attributes('-fullscreen')))
        self.bind('<Control-q>', lambda e: self.on_close())
        self.bind('<Command-q>', lambda e: self.on_close())
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.current_image_path = None
        self.target_crop_var = tk.StringVar(value="")
        self.target_state_var = tk.StringVar(value="")

        self.models = ('YOLOv11', 'Xception', 'ResNet')
        self.hist = {m: {'infer': [], 'conf': []} for m in self.models}
        self.pkg_cold_totals = {m: 0.0 for m in self.models}
        self.model_init_times = {m: 0.0 for m in self.models}

        self.yolo = None
        self.xception = None
        self.resnet = None

        self._build_topbar()
        self._build_body()
        self._build_metrics_area()
        self._build_log()

        threading.Thread(target=self._load_all_models, daemon=True).start()

    def _build_topbar(self):
        bar = tk.Frame(self, bg='white', highlightbackground='black', highlightthickness=1)
        bar.pack(fill='x', side='top')

        tk.Button(bar, text="Open Image", command=self.open_image,
                  bg='white', fg='black', relief='flat', padx=10, pady=8,
                  highlightbackground='black', highlightthickness=1).pack(side='left', padx=8, pady=8)

        tk.Label(bar, text="Target Crop:", bg='white', fg='black').pack(side='left', padx=(12, 4))
        self.target_crop_entry = ttk.Combobox(bar, textvariable=self.target_crop_var, width=24)
        self.target_crop_entry.pack(side='left', padx=(0, 10), pady=8)

        tk.Label(bar, text="Target State:", bg='white', fg='black').pack(side='left', padx=(12, 4))
        self.target_state_entry = ttk.Combobox(bar, textvariable=self.target_state_var, width=24)
        self.target_state_entry.pack(side='left', padx=(0, 10), pady=8)

        tk.Button(bar, text="Predict (All Models)", command=self.predict_all,
                  bg='white', fg='black', relief='flat', padx=12, pady=8,
                  highlightbackground='black', highlightthickness=1).pack(side='left', padx=8, pady=8)

        tk.Button(bar, text="Clear Log", command=self.clear_log,
                  bg='white', fg='black', relief='flat', padx=10, pady=8,
                  highlightbackground='black', highlightthickness=1).pack(side='right', padx=8, pady=8)

        tk.Button(bar, text="Exit", command=self.on_close,
                  bg='white', fg='black', relief='flat', padx=10, pady=8,
                  highlightbackground='black', highlightthickness=1).pack(side='right', padx=8, pady=8)

    def _build_body(self):
        body = tk.Frame(self, bg='white')
        body.pack(fill='both', expand=True, padx=12, pady=12)

        self.panel_frames = {}
        self._create_model_panel(body, 'YOLOv11')
        self._create_model_panel(body, 'Xception')
        self._create_model_panel(body, 'ResNet')

        body.grid_columnconfigure(0, weight=1, uniform='col')
        body.grid_columnconfigure(1, weight=1, uniform='col')
        body.grid_columnconfigure(2, weight=1, uniform='col')

        self.panel_frames['YOLOv11'].grid(row=0, column=0, sticky='nsew', padx=8, pady=8)
        self.panel_frames['Xception'].grid(row=0, column=1, sticky='nsew', padx=8, pady=8)
        self.panel_frames['ResNet'].grid(row=0, column=2, sticky='nsew', padx=8, pady=8)

    def _create_model_panel(self, parent, model_name: str):
        frame = tk.Frame(parent, bg='white', highlightbackground='black', highlightthickness=2)
        tk.Label(frame, text=model_name, bg='white', fg='black', font=('Segoe UI', 16, 'bold')).pack(fill='x', pady=(8, 4))
        info = tk.Text(frame, height=9, bg='white', fg='black', relief='flat')
        info.pack(fill='x', padx=8); info.insert('1.0', f"Loading {model_name}...\n"); info.configure(state='disabled')
        img_holder = tk.Label(frame, bg='white'); img_holder.pack(expand=True, fill='both', padx=8, pady=8)
        caption = tk.Label(frame, text='Prediction: —', bg='white', fg='black', font=('Segoe UI', 12))
        caption.pack(fill='x', padx=8, pady=(0, 8))
        self.panel_frames[model_name] = frame
        frame._info = info; frame._img = img_holder; frame._caption = caption

    # ----- Metrics area (3 charts) -----
    def _build_metrics_area(self):
        self.metrics_frame = tk.Frame(self, bg='white', highlightbackground='black', highlightthickness=1)
        self.metrics_frame.pack(fill='x', padx=12, pady=(0, 8))

        self.metrics_frame.grid_columnconfigure(0, weight=1, uniform='m')
        self.metrics_frame.grid_columnconfigure(1, weight=1, uniform='m')
        self.metrics_frame.grid_columnconfigure(2, weight=1, uniform='m')

        # 1) Inference time — mean ± std (rolling)
        self.fig_time = Figure(figsize=(4.0, 2.2), dpi=100)
        self.ax_time = self.fig_time.add_subplot(111)
        self.ax_time.set_title(f"Inference time — last {METRIC_WINDOW} calls (mean ± std)")
        self.ax_time.set_ylabel("Seconds")
        self.canvas_time = FigureCanvasTkAgg(self.fig_time, master=self.metrics_frame)
        self.canvas_time.get_tk_widget().grid(row=0, column=0, sticky='nsew', padx=6, pady=6)

        # 2) Confidence — mean ± std (rolling)
        self.fig_conf = Figure(figsize=(4.0, 2.2), dpi=100)
        self.ax_conf = self.fig_conf.add_subplot(111)
        self.ax_conf.set_title(f"Top-1 confidence — last {METRIC_WINDOW} calls (mean ± std)")
        self.ax_conf.set_ylabel("Probability")
        self.ax_conf.set_ylim(0.0, 1.0)
        self.canvas_conf = FigureCanvasTkAgg(self.fig_conf, master=self.metrics_frame)
        self.canvas_conf.get_tk_widget().grid(row=0, column=1, sticky='nsew', padx=6, pady=6)

        # 3) Load time (stacked)
        self.fig_load = Figure(figsize=(4.0, 2.2), dpi=100)
        self.ax_load = self.fig_load.add_subplot(111)
        self.ax_load.set_title("Model load time (sec): package cold total + model init")
        self.ax_load.set_ylabel("Seconds")
        self.canvas_load = FigureCanvasTkAgg(self.fig_load, master=self.metrics_frame)
        self.canvas_load.get_tk_widget().grid(row=0, column=2, sticky='nsew', padx=6, pady=6)

        self._refresh_plots()

    def _bar_with_error(self, ax, title, names, means, stds, ylim=None, fmt="{:.3f}"):
        ax.clear()
        ax.set_title(title)
        if ylim is not None:
            ax.set_ylim(*ylim)
        bars = ax.bar(names, means, yerr=stds, capsize=4)
        for rect, m in zip(bars, means):
            ax.text(rect.get_x() + rect.get_width()/2, rect.get_height(),
                    fmt.format(m), ha='center', va='bottom')

    def _rolling_stats(self, seq: List[float]) -> Tuple[float, float]:
        if not seq:
            return 0.0, 0.0
        w = seq[-METRIC_WINDOW:]
        if len(w) == 1:
            return float(w[0]), 0.0
        return float(np.mean(w)), float(np.std(w))

    def _refresh_plots(self):
        # Inference time (mean ± std)
        means_t, stds_t = [], []
        for m in self.models:
            mean, std = self._rolling_stats(self.hist[m]['infer'])
            means_t.append(mean); stds_t.append(std)
        self._bar_with_error(self.ax_time, f"Inference time — last {METRIC_WINDOW} calls (mean ± std)",
                             list(self.models), means_t, stds_t, ylim=None, fmt="{:.3f}")
        self.canvas_time.draw_idle()

        # Confidence (mean ± std)
        means_c, stds_c = [], []
        for m in self.models:
            mean, std = self._rolling_stats(self.hist[m]['conf'])
            means_c.append(mean); stds_c.append(std)
        self._bar_with_error(self.ax_conf, f"Top-1 confidence — last {METRIC_WINDOW} calls (mean ± std)",
                             list(self.models), means_c, stds_c, ylim=(0.0, 1.0), fmt="{:.2f}")
        self.canvas_conf.draw_idle()

        # Load times (stacked)
        self.ax_load.clear()
        self.ax_load.set_title("Model load time (sec): package cold total + model init")
        self.ax_load.set_ylabel("Seconds")
        names = list(self.models)
        pkg_vals = [self.pkg_cold_totals[m] for m in names]
        init_vals = [self.model_init_times[m] for m in names]
        bars1 = self.ax_load.bar(names, pkg_vals, label="Package cold total")
        bars2 = self.ax_load.bar(names, init_vals, bottom=pkg_vals, label="Model init")
        for rect, base, add in zip(bars2, pkg_vals, init_vals):
            total = base + add
            self.ax_load.text(rect.get_x() + rect.get_width()/2, total,
                              f"{total:.3f}", ha='center', va='bottom')
        self.ax_load.legend()
        self.canvas_load.draw_idle()

    def _build_log(self):
        log_frame = tk.Frame(self, bg='white', highlightbackground='black', highlightthickness=1)
        log_frame.pack(fill='x', side='bottom')
        tk.Label(log_frame, text='Log', bg='white', fg='black', font=('Segoe UI', 12, 'bold')).pack(anchor='w', padx=8)
        self.log_box = tk.Text(log_frame, height=10, bg='white', fg='black', relief='flat')
        self.log_box.pack(fill='x', padx=8, pady=(0, 8))
        self._log("App started. Load models in background…")

    def _log(self, text: str):
        def do():
            self.log_box.configure(state='normal')
            self.log_box.insert('end', text + "\n")
            self.log_box.see('end')
            self.log_box.configure(state='disabled')
        self.after(0, do)

    def _panel_log(self, model_name: str, text: str):
        def do():
            fr = self.panel_frames.get(model_name)
            if not fr: return
            info = fr._info
            info.configure(state='normal'); info.insert('end', text + "\n"); info.see('end'); info.configure(state='disabled')
        self.after(0, do)

    def _set_caption(self, model_name: str, text: str):
        def do():
            fr = self.panel_frames.get(model_name)
            if fr: fr._caption.config(text=text)
        self.after(0, do)

    def _set_panel_image(self, model_name: str, image_path: str, max_side: int = 420):
        def do():
            try:
                fr = self.panel_frames.get(model_name)
                if not fr: return
                img = Image.open(image_path).convert('RGB')
                w, h = img.size; scale = max(w, h) / float(max_side)
                if scale > 1: img = img.resize((int(w/scale), int(h/scale)), Image.LANCZOS)
                tkimg = ImageTk.PhotoImage(img)
                fr._img.configure(image=tkimg)
                if not hasattr(self, '_img_cache_dict'): self._img_cache_dict = {}
                self._img_cache_dict[model_name] = tkimg
            except Exception as e:
                self._panel_log(model_name, f"[Image Error] {e}")
        self.after(0, do)

    # Loading (sequential; with split timings)
    def _load_all_models(self):
        # YOLO
        try:
            start = time.perf_counter()
            self.yolo = YoloClassifier(YOLO_MODEL_PATH)
            secs = time.perf_counter() - start
            m = 'YOLOv11'
            self.pkg_cold_totals[m] = self.yolo.pkg_cold_total
            self.model_init_times[m] = self.yolo.model_init_time
            self._panel_log(m,
                "Load breakdown:\n"
                f"  Package import (this call): {self.yolo.pkg_time_this_call:.3f}s\n"
                f"  Package cold total (deps):  {self.yolo.pkg_cold_total:.3f}s\n"
                f"  Model init:                 {self.yolo.model_init_time:.3f}s\n"
                f"  Wrapper init (outer):       {secs:.3f}s"
            )
            self._log(f"[YOLO] pkg(this) {self.yolo.pkg_time_this_call:.3f}s | pkg(cold total) {self.yolo.pkg_cold_total:.3f}s | model {self.yolo.model_init_time:.3f}s")

            # Populate dropdowns from YOLO class names
            try:
                names = self.yolo.classes()
                crops, states = set(), set()
                for nm in (names.values() if isinstance(names, dict) else names):
                    c, s = split_combined_label(nm)
                    if c: crops.add(c)
                    if s: states.add(s)
                if crops: self._merge_target_crop_options(sorted(crops))
                if states: self._merge_target_state_options(sorted(states))
            except Exception:
                pass
        except Exception as e:
            self._panel_log('YOLOv11', "Load failed: {}\n{}".format(e, traceback.format_exc()))
            self._log("[YOLO] Load failed. See panel for details.")

        self.after(0, self._refresh_plots)

        # Xception
        try:
            start = time.perf_counter()
            self.xception = XceptionClassifier(XCEPTION_TFLITE, XCEPTION_LABELS_CROP, XCEPTION_LABELS_STATE)
            secs = time.perf_counter() - start
            m = 'Xception'
            self.pkg_cold_totals[m] = self.xception.pkg_cold_total
            self.model_init_times[m] = self.xception.model_init_time
            self._panel_log(m,
                "Load breakdown:\n"
                f"  Package import (this call): {self.xception.pkg_time_this_call:.3f}s\n"
                f"  Package cold total (deps):  {self.xception.pkg_cold_total:.3f}s\n"
                f"  Model init:                 {self.xception.model_init_time:.3f}s\n"
                f"  Wrapper init (outer):       {secs:.3f}s"
            )
            self._log(f"[Xception] pkg(this) {self.xception.pkg_time_this_call:.3f}s | pkg(cold total) {self.xception.pkg_cold_total:.3f}s | model {self.xception.model_init_time:.3f}s")
            # populate dropdowns
            self._merge_target_crop_options(self.xception.class_names)
            self._merge_target_state_options(self.xception.state_labels)
        except Exception as e:
            self._panel_log('Xception', "Load failed: {}\n{}".format(e, traceback.format_exc()))
            self._log("[Xception] Load failed. See panel for details.")

        self.after(0, self._refresh_plots)

        # ResNet (Keras)
        try:
            start = time.perf_counter()
            self.resnet = ResNetClassifier(RESNET50_KERAS_MODEL, RESNET50_CLASS_CSV, RESNET_CLASSES_JSON)
            secs = time.perf_counter() - start
            m = 'ResNet'
            self.pkg_cold_totals[m] = self.resnet.pkg_cold_total
            self.model_init_times[m] = self.resnet.model_init_time
            self._panel_log(m,
                "Load breakdown:\n"
                f"  Package import (this call): {self.resnet.pkg_time_this_call:.3f}s\n"
                f"  Package cold total (deps):  {self.resnet.pkg_cold_total:.3f}s\n"
                f"  Model init:                 {self.resnet.model_init_time:.3f}s\n"
                f"  Wrapper init (outer):       {secs:.3f}s"
            )
            self._log(f"[ResNet] pkg(this) {self.resnet.pkg_time_this_call:.3f}s | pkg(cold total) {self.resnet.pkg_cold_total:.3f}s | model {self.resnet.model_init_time:.3f}s")

            # populate dropdowns
            crops, states = set(), set()
            for nm in self.resnet.class_names:
                c, s = split_combined_label(nm)
                if c: crops.add(c)
                if s: states.add(s)
            if crops: self._merge_target_crop_options(sorted(crops))
            if states: self._merge_target_state_options(sorted(states))
        except Exception as e:
            self._panel_log('ResNet', "Load failed: {}\n{}".format(e, traceback.format_exc()))
            self._log("[ResNet] Load failed. See panel for details.")

        self.after(0, self._refresh_plots)

    # Options helpers
    def _merge_target_crop_options(self, more: List[str]):
        def do():
            existing = list(self.target_crop_entry["values"]) if self.target_crop_entry["values"] else []
            merged = sorted(set(existing + [str(o) for o in more if o]))
            self.target_crop_entry["values"] = merged
        self.after(0, do)

    def _merge_target_state_options(self, more: List[str]):
        def do():
            existing = list(self.target_state_entry["values"]) if self.target_state_entry["values"] else []
            merged = sorted(set(existing + [str(o) for o in more if o]))
            self.target_state_entry["values"] = merged
        self.after(0, do)

    # Actions
    def open_image(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff")]
        )
        if not path: return
        self.current_image_path = path
        for model_name in ('YOLOv11', 'Xception', 'ResNet'):
            self._set_panel_image(model_name, path)
            self._set_caption(model_name, 'Prediction: —')
        self._log("Opened image: {}".format(os.path.basename(path)))

    def predict_all(self):
        if not self.current_image_path:
            messagebox.showinfo("No image", "Please open an image first.")
            return
        threading.Thread(target=self._predict_all_thread, daemon=True).start()

    def _predict_all_thread(self):
        image_path = self.current_image_path
        target_crop = self.target_crop_var.get().strip() or None
        target_state = self.target_state_var.get().strip() or None

        if self.yolo is not None:
            try:
                self._panel_log('YOLOv11', "Running prediction…")
                start = time.perf_counter()
                out = self.yolo.predict(image_path, target_crop, target_state)
                total = time.perf_counter() - start
                self._render_result('YOLOv11', out, total, target_crop, target_state)
                self._update_metrics('YOLOv11', out)
            except Exception as e:
                self._panel_log('YOLOv11', "Prediction error: {}\n{}".format(e, traceback.format_exc()))

        if self.xception is not None:
            try:
                self._panel_log('Xception', "Running prediction…")
                start = time.perf_counter()
                out = self.xception.predict(image_path, target_crop, target_state)
                total = time.perf_counter() - start
                self._render_result('Xception', out, total, target_crop, target_state)
                self._update_metrics('Xception', out)
            except Exception as e:
                self._panel_log('Xception', "Prediction error: {}\n{}".format(e, traceback.format_exc()))

        if self.resnet is not None:
            try:
                self._panel_log('ResNet', "Running prediction…")
                start = time.perf_counter()
                out = self.resnet.predict(image_path, target_crop, target_state)
                total = time.perf_counter() - start
                self._render_result('ResNet', out, total, target_crop, target_state)
                self._update_metrics('ResNet', out)
            except Exception as e:
                self._panel_log('ResNet', "Prediction error: {}\n{}".format(e, traceback.format_exc()))

    # Metrics logging
    def _update_metrics(self, model_name: str, out: dict):
        infer = float(out.get('infer_secs', 0.0))
        # For Xception, we track crop-head confidence for "Top-1"
        conf = float(out['crop'].get('top1_conf', 0.0)) if model_name == 'Xception' and isinstance(out.get('crop'), dict) else float(out.get('top1_conf', 0.0))
        self.hist[model_name]['infer'].append(infer)
        self.hist[model_name]['conf'].append(conf)
        self.hist[model_name]['infer'] = self.hist[model_name]['infer'][-50:]
        self.hist[model_name]['conf']  = self.hist[model_name]['conf'][-50:]
        self.after(0, self._refresh_plots)

    # Render results text
    def _render_result(self, model_name: str, out: dict, total_secs: float, target_crop: Optional[str], target_state: Optional[str]):
        if model_name == 'Xception' and isinstance(out.get('crop'), dict) and isinstance(out.get('state'), dict):
            crop_top = out['crop']['top1_name']; crop_p = out['crop']['top1_conf']
            state_top = out['state']['top1_name']; state_p = out['state']['top1_conf']
            caption = f"Crop: {crop_top} ({fmt_pct(crop_p)}) | State: {state_top} ({fmt_pct(state_p)})"
            self._set_caption(model_name, caption)

            lines = []
            lines.append("Device: {}".format(out.get('device', '—')))
            if 'prep_secs' in out: lines.append("Prep time: {:.3f}s".format(out['prep_secs']))
            lines.append("Inference time: {:.3f}s".format(out.get('infer_secs', 0.0)))
            lines.append("Load breakdown:")
            lines.append("  Package import (this call): {:.3f}s".format(out.get('pkg_time_this_call', 0.0)))
            lines.append("  Package cold total (deps):  {:.3f}s".format(out.get('pkg_cold_total', 0.0)))
            lines.append("  Model init:                 {:.3f}s".format(out.get('model_init_time', 0.0)))
            lines.append("Total call time: {:.3f}s".format(total_secs))
            lines.append("")
            lines.append("Crop – Top-5:")
            for i, (name, conf) in enumerate(out['crop']['top5'], start=1):
                lines.append("  {}. {:<30} {}".format(i, name, fmt_pct(conf)))
            lines.append("")
            lines.append("State – Top-5:")
            for i, (name, conf) in enumerate(out['state']['top5'], start=1):
                lines.append("  {}. {:<30} {}".format(i, name, fmt_pct(conf)))

            if target_crop:
                tc = out.get('target_crop_conf'); lines.append("")
                lines.append(f"Target Crop '{target_crop}': {fmt_pct(tc) if tc is not None else '—'}")
            if target_state:
                ts = out.get('target_state_conf')
                lines.append(f"Target State '{target_state}': {fmt_pct(ts) if ts is not None else '—'}")

            self._panel_log(model_name, "\n".join(lines))
            self._log(f"[Xception] Crop: {crop_top} ({fmt_pct(crop_p)}) | State: {state_top} ({fmt_pct(state_p)}) | Infer: {out.get('infer_secs', 0.0):.3f}s")
            return

        # YOLO / ResNet (combined labels)
        combined = out.get('top1_name', '—')
        crop_pred, state_pred = split_combined_label(combined)
        caption = f"Crop: {crop_pred} ({fmt_pct(out.get('top1_conf', 0.0))}) | State: {state_pred or '—'}"
        self._set_caption(model_name, caption)

        lines = []
        lines.append("Device: {}".format(out.get('device', '—')))
        if 'prep_secs' in out: lines.append("Prep time: {:.3f}s".format(out['prep_secs']))
        lines.append("Inference time: {:.3f}s".format(out.get('infer_secs', 0.0)))
        lines.append("Load breakdown:")
        lines.append("  Package import (this call): {:.3f}s".format(out.get('pkg_time_this_call', 0.0)))
        lines.append("  Package cold total (deps):  {:.3f}s".format(out.get('pkg_cold_total', 0.0)))
        lines.append("  Model init:                 {:.3f}s".format(out.get('model_init_time', 0.0)))
        lines.append("Total call time: {:.3f}s".format(total_secs))
        lines.append("")
        lines.append("Top-5 (combined classes):")
        for i, (name, conf) in enumerate(out.get('top5', []), start=1):
            c, s = split_combined_label(name)
            pretty = f"{c} | {s}" if s else c
            lines.append("  {}. {:<40} {}".format(i, pretty, fmt_pct(conf)))

        if target_crop:
            tc = out.get('target_crop_conf'); lines.append("")
            lines.append(f"Target Crop '{target_crop}': {fmt_pct(tc) if tc is not None else '—'} (sum over all states)")
        if target_state:
            ts = out.get('target_state_conf')
            lines.append(f"Target State '{target_state}': {fmt_pct(ts) if ts is not None else '—'} (sum over all crops)")

        self._panel_log(model_name, "\n".join(lines))
        self._log("[{}] Crop: {} | State: {} | Infer: {:.3f}s".format(
            model_name, crop_pred, state_pred or '—', out.get('infer_secs', 0.0)))

    def clear_log(self):
        self.log_box.configure(state='normal')
        self.log_box.delete('1.0', 'end')
        self.log_box.configure(state='disabled')

    def on_close(self):
        try:
            self.destroy()
        except Exception:
            try: self.quit()
            finally: pass

if __name__ == "__main__":
    App().mainloop()
