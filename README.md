# Landmine Detection and Ground Marking : Team MEOW

INDIGINOUS MILITARY TECHNOLOGY USING QUADRUPED ROBOTIC SWARM
VIT Vellore

-----

## Problem Statement

Landmine detection remains a high-risk, time-consuming, and manpower-intensive military operation. More than 3,000 Indians have been victims of landmine explosions in recent years. Existing approaches all fall short in different ways:

- **Manual de-mining** — soldiers physically sweeping terrain with detectors, putting lives directly at risk, especially under time pressure.
- **Drone-based detection** — bulky, computationally expensive, requires manual control for precision, can be spotted and tracked by enemy radar, and is unreliable in fog/smog.
- **Minesweepers** — mechanical rollers that trigger mines rather than identify them; bulky, easily spotted, and unusable for stealth operations.

None of these are well-suited for situations demanding stealth, speed, and safety at the same time.

## Our Solution

A **quadruped robotic swarm** designed to autonomously scan terrain, detect landmines using vision-based machine learning, and mark confirmed locations for safe clearance — without putting a single soldier at risk.

- Lightweight build means the robot does **not trigger** mine pressure plates while crossing terrain.
- Each unit analyzes its camera feed in real time to identify landmine-likely visual and environmental anomalies.
- Confirmed detections are marked on the ground with a **fluorescent marking**, visible even in low light, so clearance teams know exactly where to act.
- Designed to be deployed as a **swarm**, scaling coverage across a hazard zone faster than manual sweeping.

### System Architecture

```
Deploy quadruped swarm → Quadruped robots in motion (inverse kinematics)
        → Environment scanning (camera feed: thermal / night vision + sensor data)
        → Machine learning model
              ├── Classify terrain anomaly → Disturbances / soil damage / heat signs
              └── Detection of landmines
        → Identify areas with landmines → Fluorescent marking
        → Report data to base station → Create hazard map
        → AI model insights (camera feed + stats) → Human verification for next action
```

- **ML approach:** Transfer learning using **MobileNetV2**, achieving ~95–96% detection accuracy in testing.
- **Hardware:** 3D-printed/carbon-fibre quadruped frame, SG90 high-torque servos, Raspberry Pi / NVIDIA Jetson Nano for onboard compute and camera feed.

## Simulation

Before field-testing the physical hardware, we built a full software simulation of the robot's core behavior — locomotion, onboard vision, and detection — so the complete pipeline can be demonstrated end-to-end in a safe, repeatable environment.

**What the simulation does:**

- **Physics-based quadruped locomotion** — models the real robot's parallel five-bar leg linkage (8 actuated motors + passive knee joints) with a scripted trot gait, walking autonomously from a start point to an end point across the terrain.
- **Live onboard camera feed** — a continuously updating first-person camera view, streamed in real time in its own window, mirroring the actual Raspberry Pi camera module on the physical robot.
- **Real image-based landmine detection** — uses OpenCV ORB (Oriented FAST and Rotated BRIEF) feature matching to compare each live camera frame against reference landmine photographs, genuinely identifying visual matches rather than scripting fake detections.
- **Detection response behavior** — on detecting a landmine, the robot halts to confirm and report the exact (x, y) coordinates (on-screen and in console), marks the spot with a bright fluorescent-yellow ground marker for day/night visibility, then resumes at increased speed to move clear of the hazard.
- **Path integrity check** — an off-path decoy mine is included in the scene, positioned well outside the robot's route, to demonstrate that detection is genuinely tied to the robot's proximity and path rather than firing indiscriminately.

**Tech stack used for the simulation:**
- **PyBullet** — physics engine and 3D rendering for the robot, terrain, and environment
- **OpenCV** — real feature-based image recognition for detection
- **Anaconda / Conda** — used to manage the Python environment, since PyBullet requires prebuilt binaries that aren't available via pip on Windows
- **Python** — orchestrates gait control, navigation, camera capture, and detection logic
