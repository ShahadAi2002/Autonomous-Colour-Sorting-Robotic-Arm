#  Autonomous Colour-Sorting Robotic Arm

A cognitive robotics project built in **Webots** simulation, where a **UR5 robotic arm** autonomously sorts objects by colour using a **YOLOv8** perception model, a **Decision Tree** classifier, and an **adaptive confidence threshold** mechanism.

---

##  Overview

The robot follows a **sense → think → act** loop:
1. **Perceives** an object via camera using YOLOv8 (detects colour: red, blue, or yellow)
2. **Decides** whether to accept or reject it using a trained Decision Tree
3. **Acts** by moving to and pressing the matching coloured button (or the reject button)

The system also **self-regulates** its confidence threshold over time — raising it when detections are strong, lowering it after repeated rejections.

---

##  Repository Structure

```
autonomous-colour-sorting-arm/
│
├── camera_test.py          # Main Webots robot controller (state machine + full pipeline)
├── yolo_training.ipynb     # YOLOv8 training notebook + dataset preparation
├── decision_model.pkl      # Trained Decision Tree model
├── label_encoder.pkl       # Label encoder for Decision Tree output
└── README.md
```

---

##  System Architecture

The pipeline flows through three cognitive modules:

| Module | Component | Role |
|---|---|---|
| Perception | YOLOv8 | Detects object colour + confidence + bounding box size |
| Decision | Decision Tree + Adaptive Threshold | Accept or reject based on 4 features |
| Action | FSM Controller | Moves arm to correct button and presses it |

**Decision Tree input features:**
- `confidence` — YOLO certainty score
- `required_conf` — current adaptive threshold
- `size` — bounding box pixel area
- `confidence_gap` — confidence − required_conf

---

##  Adaptive Threshold Mechanism

The confidence threshold starts at **0.75** and adjusts automatically:

- **Reward** → if confidence > 0.90 consistently, threshold increases by `+0.01` (more selective)
- **Penalty** → after 3 consecutive rejections, threshold decreases by `-0.05` (more lenient)
- **Uncertainty logging** → if YOLO confidence is between 0.40 and the threshold, the image is saved as `needs_learning_N.jpg` for future retraining

---

##  State Machine

The robot cycles through 4 states per object:

```
INSPECT → MOVE_TO_BUTTON → PRESS_BUTTON → RETURN
```

---





##  Future Improvements

- **Pick & place** — physically move objects into colour-coded bins instead of pressing buttons
- **9-class model** — split by colour + shape (e.g. red_circle, blue_square, yellow_cylinder)
- **Sim-to-Real** — deploy on a physical UR5 arm
- **Multi-arm** — split the task across two arms for higher throughput

---


