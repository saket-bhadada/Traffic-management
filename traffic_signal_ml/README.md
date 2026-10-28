# Adaptive Traffic Signal Control with Emergency Preemption

An end-to-end Machine Learning and deterministic traffic signal control system designed for a standard four-way Indian intersection (**chowk**).

Four synchronized camera approaches (`N`, `E`, `S`, `W`) are sampled in lockstep at **1 FPS**. At each second $t$, the system:
1. Detects standard and emergency vehicles inside per-lane polygon ROIs using YOLOv8 (`yolov8n` default, configurable switch for `yolov8s`).
2. Builds the 8-dimensional state vector:
   $$X_t = [C_N, C_E, C_S, C_W, E_N, E_E, E_S, E_W]$$
   and logs every tick to CSV for replay.
3. Feeds a sliding window of the last $W=10$ seconds of $X_t$ into a **2-layer LSTM** with a linear head to predict per-lane queue growth rate $\frac{dC}{dt}$.
4. Passes counts and predicted growth rates to a **deterministic, ML-free controller** that dynamically recomputes green duration every second and executes an explicit 5-state emergency preemption state machine.

---

## Project Layout

```text
traffic_signal_ml/
  configs/
    datasets.yaml       # Roboflow dataset specs, split ratios, YOLOv8n/s switch
    class_map.yaml      # 2-class mapping (0: standard_vehicle, 1: emergency_vehicle)
    lanes.yaml          # Polygon ROIs per approach & normalize_by_area flag
    controller.yaml     # Adaptive timer constants, preemption thresholds, LSTM & sim config
  src/
    data/
      download.py       # Downloads 5 Roboflow Universe datasets via ROBOFLOW_API_KEY
      unify_labels.py   # Remaps source labels to {0, 1}, drops unmapped, logs counts
      build_splits.py   # Leak-free group splitting (.rf.<hash> aware) + emergency stratification
    detection/
      train_yolo.py     # Trains YOLOv8n / YOLOv8s and exports per-class test metrics
      detector.py       # Single Detector class returning (boxes, class_ids, confidences)
    features/
      lane_density.py   # Polygon ROI filtering and optional area normalization
      feature_vector.py # Builds X_t = [C_N, C_E, C_S, C_W, E_N, E_E, E_S, E_W] & logs to CSV
    prediction/
      dataset.py        # Sliding-window dataset (W=10 -> dC/dt per lane)
      lstm_model.py     # 2-layer PyTorch LSTM + linear head & OnlineQueuePredictor
      train_lstm.py     # Chronological train/val/test training on real or synthetic logs
    control/
      state_machine.py  # Explicit 5-state machine (Zero ML imports)
      controller.py     # Deterministic TrafficSignalController + transition CSV logger
    sim/
      synthetic_traffic.py # Bursty platoon arrivals, lulls, and emergency vehicle generator
      intersection_sim.py  # 1-second-step queue & discharge simulator
      baseline_fixed.py    # Fixed-timer (30s green, 3s amber, 1s all-red) baseline
    feeds/
      base.py              # FeedSource abstract interface
      video_file_feed.py   # 4-file lockstep 1 FPS VideoFileFeed with shared timestamp
      rtsp_feed.py         # RTSPFeed live camera stub
    evaluation/
      metrics.py           # Wait time, idle green, throughput, emergency time-to-green
      compare.py           # Side-by-side evaluation runner saving CSV comparison
    dashboard/
      app.py               # Streamlit live 4-feed visualization & preemption dashboard
  tests/
    test_state_machine.py
    test_features_and_feeds.py
    test_prediction_and_sim.py
  main.py
  README.md
  requirements.txt
```

---

## 1. Datasets & Licenses

The data pipeline (`src/data/download.py`, `src/data/unify_labels.py`, `src/data/build_splits.py`) is configured for the following five Roboflow Universe datasets:

| Dataset Identifier | Target Classes Supplied | License | Source URL |
| :--- | :--- | :--- | :--- |
| `roboflow-100/vehicles-q0x2v` | Car, Bus, Truck, Motorcycle | **CC BY 4.0** | [Link](https://universe.roboflow.com/roboflow-100/vehicles-q0x2v) |
| `vehicle-mscoco/vehicles-coco` | Car, Bus, Truck, Motorcycle | **CC BY 4.0** | [Link](https://universe.roboflow.com/vehicle-mscoco/vehicles-coco) |
| `indian-road-dataset/indian-roads-detection` | Car, Bus, Truck, Multi-axle Truck, Auto-rickshaw, Motorcycle | **CC BY 4.0** | [Link](https://universe.roboflow.com/indian-road-dataset/indian-roads-detection) |
| `ph-ambulance-dataset-with-augmentations/ph-ambulances` | Ambulance, Standard Vehicles | **CC BY 4.0** | [Link](https://universe.roboflow.com/ph-ambulance-dataset-with-augmentations/ph-ambulances) |
| `himank-vpetc/ambulance-4bova` | Ambulance, Fire Engine | **CC BY 4.0** | [Link](https://universe.roboflow.com/himank-vpetc/ambulance-4bova) |

### Leak-Free Split Construction
`src/data/build_splits.py` strips Roboflow augmentation suffixes (`.rf.<hex_hash>` and `_aug*`) via regex to group all augmented variants under their canonical source image stem before splitting (`70% train / 15% val / 15% test`), guaranteeing zero data leakage across splits and stratifying groups so `emergency_vehicle` examples are present in every split.

---

## 2. Detector & LSTM Training Status

- **YOLOv8 Detector (`src/detection/train_yolo.py`, `src/detection/detector.py`)**:
  - Supports `yolov8n` (default) and `yolov8s` via `configs/datasets.yaml` (`detector.model_variant`) or CLI flag `--model yolov8s`.
  - **Note on Dataset & Training**: As requested, live Roboflow dataset downloading and YOLOv8 weight training were **not** executed because the dataset has not been provided yet (`.env` has been created with `ROBOFLOW_API_KEY=your_roboflow_api_key_here`). Once your key or local dataset is placed in `data/splits/data.yaml`, run `python src/detection/train_yolo.py` to produce `artifacts/detector_best.pt` and `artifacts/detector_test_metrics.json` (reporting per-class Precision, Recall, and mAP50 for `standard_vehicle` and `emergency_vehicle`). Until trained weights are present, `Detector` automatically uses a deterministic color/contour fallback for synthetic demo videos.
- **2-Layer LSTM Queue Growth Predictor (`src/prediction/train_lstm.py`)**:
  - **Trained on Synthetic Data**: Because no real-world intersection CSV logs were present initially, the 2-layer LSTM (`input_size=8`, `hidden_size=64`, `num_layers=2`, `output_size=4`, window $W=10\text{s}$) was **trained on synthetic traffic data** generated by `src/sim/synthetic_traffic.py` (3,000 seconds with varied arrival rates, bursts, lulls, and emergency vehicle events, split chronologically `70/15/15`).
  - **Saved Weights**: `artifacts/lstm_predictor.pt`
  - **Validation / Test Metrics**: Best Validation MSE = `0.3944`, Test MSE = `0.4643`, Test MAE = `0.4231` veh/s.

---

## 3. Simulation Comparison: Adaptive + Preemption vs. Fixed-Timer Baseline

Both controllers were evaluated over an identical seeded **1,800-second (30-minute)** Indian chowk traffic trace (`seed=42`, saturation discharge rate = `1.5 veh/s`, `3s` Amber, `1s` All-Red, 11 emergency vehicle arrivals). Full results are saved to `artifacts/sim_comparison.csv`:

| Controller | Average Wait Time (s) | Idle Green Time (s) | Throughput (vehicles) | Emergency Time-to-Green (s) | Emergency Events |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Fixed-Timer Baseline (30s)** | 150.43 | 577.0 | 1,475 | 38.09 | 11 |
| **Adaptive + Emergency Preemption** | **54.45** | **328.0** | **1,668** | **5.18** | 11 |
| **Improvement** | **-63.8% wait** | **-43.2% wasted green** | **+13.1% throughput** | **-86.4% emergency delay** | — |

---

## 4. Entry Points & Commands

Activate the Python 3.11 virtual environment and run:

```powershell
# 1. Run the 1-second-step intersection simulator & baseline comparison
.\.venv\Scripts\python.exe traffic_signal_ml/main.py --mode sim

# 2. Run the 4-way synchronized video detector + LSTM + controller pipeline
.\.venv\Scripts\python.exe traffic_signal_ml/main.py --mode video

# 3. Launch the interactive Streamlit dashboard
.\.venv\Scripts\streamlit.exe run traffic_signal_ml/src/dashboard/app.py

# 4. Run the full pytest suite
.\.venv\Scripts\pytest.exe traffic_signal_ml/tests -v
```

---

## 5. Assumptions Made

1. **Right-of-Way & Clockwise Chowk Phase Order**: Each phase (`N -> E -> S -> W -> N`) grants protected green to the active approach while the other three approaches display red, matching standard 4-phase Indian signalized intersections (`chowks`).
2. **Adaptive Green Recomputation**: During `ADAPTIVE_GREEN`, the target green duration $\text{clamp}(\text{base} + k_1 C_{\text{lane}} + k_2 \max(dC/dt, 0), 10, 60)$ is recomputed every second using live counts and LSTM predictions, and remaining green time is updated as $\max(0, \text{target\_green} - \text{elapsed\_in\_green})$.
3. **Post-Preemption Safety Clearance**: When an emergency vehicle clears the intersection ($E_j == 0$) during `EMERGENCY_GREEN`, the controller transitions through standard `AMBER` (3s) and `ALL_RED` (1s) clearance intervals before resuming the clockwise cycle at phase $(i+1)$.
4. **Simultaneous Multi-Lane Emergencies**: If emergency vehicles are present on multiple approaches simultaneously, the active green approach is prioritized first (if it has an emergency vehicle) to avoid unnecessary phase switching; otherwise the next emergency approach in clockwise order is selected.
5. **Fixed-Timer Baseline Configuration**: The baseline uses a standard `30s` fixed green per approach with the exact same `3s` Amber and `1s` All-Red clearance intervals as the adaptive controller.
6. **Bounding Box ROI Anchor**: A detected vehicle is counted inside a lane's polygon ROI if the bottom-center of its bounding box (`(0.5*(x1+x2), y2)`) lies inside or on the boundary of the polygon (`configs/lanes.yaml`).
