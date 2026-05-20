from controller import Robot
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import sys
import joblib


robot = Robot()
timestep = int(robot.getBasicTimeStep())

# YOLO model
model = YOLO("best.pt")

# Decision Tree model
decision_model = joblib.load("decision_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ========= DEVICES =========
camera = robot.getDevice("camera")
camera.enable(timestep)


# capture img
def capture_image():
    img = camera.getImage()
    width = camera.getWidth()
    height = camera.getHeight()

    img = np.frombuffer(img, np.uint8)
    img = img.reshape((height, width, 4))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img


# joint names
joint_names = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint"
]

motors = []
for name in joint_names:
    motors.append(robot.getDevice(name))

# ========= POSES =========
inspection_pose = [0, -1.2, 1.6, -1.4, -1.57, 0]

red_button_pose    = [0.7, -0.6, 1.3, 0.9, 1.5, 1]
blue_button_pose   = [-0.5, -0.7, 1.5, 0.9, 1.5, 1]
yellow_button_pose = [-1.05, -0.5, 1, 0.9, 1.5, 1]
reject_button_pose = [1.7, -0.6, 1.4, 0.9, 1.5, 1]

pressing_pose = [0.0, 0.15, -0.15, 0.1, 0.0, 0.0]


# ========= HELPERS =========
def move_to_pose(pose):
    for motor, position in zip(motors, pose):
        motor.setPosition(position)


def wait(duration):
    start = robot.getTime()
    while robot.step(timestep) != -1:
        if robot.getTime() - start > duration:
            break


def get_target_pose(label):
    label = label.lower()
    if "red" in label:
        return red_button_pose
    elif "blue" in label:
        return blue_button_pose
    elif "yellow" in label:
        return yellow_button_pose
    else:
        return reject_button_pose


# ========= YOLO PERCEPTION =========
def get_perception():
    img = capture_image()
    results = model(img)

    if len(results[0].boxes) == 0:
        return {"label": "none", "confidence": 0.0, "size": 0.0}

    # Select highest confidence detection
    box = max(results[0].boxes, key=lambda b: float(b.conf[0]))

    cls_id = int(box.cls[0])
    confidence = float(box.conf[0])
    label = model.names[cls_id]

    x1, y1, x2, y2 = box.xyxy[0]
    width = x2 - x1
    height = y2 - y1
    area = width * height

    if area < 400:
        return {"label": "none", "confidence": 0.0, "size": 0.0}

    return {"label": label, "confidence": confidence, "size": float(area)}

   


# ========= DECISION TREE DECISION =========
def make_decision(confidence, required_conf, size):

    confidence_gap = confidence - required_conf

    # fix sklearn feature-name warning
    X = pd.DataFrame([{
        "confidence": confidence,
        "required_conf": required_conf,
        "size": size,
        "confidence_gap": confidence_gap
    }])

    prediction = decision_model.predict(X)[0]
    decision = label_encoder.inverse_transform([prediction])[0]

    return decision, confidence_gap


# ========= STATE MACHINE =========
move_to_pose(inspection_pose)

state = "INSPECT"
target_pose = None

# Adaptive Threshold
adaptive_conf = 0.75
consecutive_rejects = 0   # tracks reject streak, not the whole count
image_counter = 0

# Basic counters
total_items = 0
accepted_items = 0
rejected_items = 0

while robot.step(timestep) != -1:

    if state == "INSPECT":

        print("Inspecting object")

        #### 1- YOLO Perception ####
        data = get_perception()

        label = data["label"]
        confidence = data["confidence"]
        size = data["size"]

        if label != "none":
            total_items += 1

        #### 2- Decision Tree Decision ####
        decision, confidence_gap = make_decision(confidence, adaptive_conf, size)

        print(
            "label:", label,
            "| confidence:", round(confidence, 3),
            "| required:", adaptive_conf,
            "| size:", round(size, 1),
            "| gap:", round(confidence_gap, 3),
            "| ML decision:", decision
        )

        #### 3- Action based on DT model ####

        if decision == "accept":
            target_pose = get_target_pose(label)
            state = "MOVE_TO_BUTTON"
            accepted_items += 1

            # Reset reject streak on accept so penalty does not fire unfairly
            consecutive_rejects = 0

            # Adaptive reward
            if confidence > 0.90 and adaptive_conf < 0.85:
                adaptive_conf = round(adaptive_conf + 0.01, 2)
                print("adaptive_conf increased to:", adaptive_conf)

        else:
            # All reject logic and adaptive penalty live inside else only
            print("Decision Tree rejected the object -> moving to reject button")

            # Save only  uncertain images
            if 0.40 <= confidence <= adaptive_conf:
                print("uncertain prediction -> reject + save image")
                img = capture_image()
                cv2.imwrite(f"needs_learning_{image_counter}.jpg", img)
                image_counter += 1
            else:
                print("low confidence -> moving to reject button")

            target_pose = reject_button_pose
            state = "MOVE_TO_BUTTON"
            rejected_items += 1
            consecutive_rejects += 1

            # Penalty fires on consecutive reject streak only
            if consecutive_rejects >= 3 and adaptive_conf > 0.50:
                adaptive_conf = round(adaptive_conf - 0.05, 2)
                print("too many consecutive rejects, adaptive_conf lowered to:", adaptive_conf)
                consecutive_rejects = 0   # reset streak after adjustment

    elif state == "MOVE_TO_BUTTON":
        print("Moving to matching colour button")
        move_to_pose(target_pose)
        wait(2)
        state = "PRESS_BUTTON"

    elif state == "PRESS_BUTTON":
        print("Pressing matching button")
        pressed_pose = [a + b for a, b in zip(target_pose, pressing_pose)]
        move_to_pose(pressed_pose)
        wait(1)
        move_to_pose(target_pose)
        wait(1)
        state = "RETURN"

    elif state == "RETURN":
        print("Returning")
        move_to_pose(inspection_pose)
        wait(2)
        print(f"  Items seen: {total_items} | Accepted: {accepted_items} | Rejected: {rejected_items} | Uncertain images saved: {image_counter}")
        state = "INSPECT"