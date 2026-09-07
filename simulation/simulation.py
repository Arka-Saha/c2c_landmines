import pybullet as p
import pybullet_data
import time
import math
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

# ============================================================
# SETUP
# ============================================================

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

plane = p.loadURDF("plane.urdf")

# Tint the ground a sandy/dirt tone instead of the default blue-and-white
# checkerboard, to look more like actual battlefield/minefield terrain.
p.changeVisualShape(plane, -1, rgbaColor=[0.76, 0.68, 0.48, 1.0])

START_X = 0
END_X = 10

# Three robots on parallel paths, 2 units apart perpendicular to the
# direction of travel (left robot at y=-2, center at y=0, right at y=+2)
ROBOT_Y_OFFSETS = {"left": -2.0, "center": 0.0, "right": 2.0}

p.resetDebugVisualizerCamera(
    cameraDistance=11.0,
    cameraYaw=50,
    cameraPitch=-45,
    cameraTargetPosition=[(START_X + END_X) / 2, 0, 0],
)

# --- Night mode: press 'N' during the simulation to toggle ---
NIGHT_MODE = False
DAY_GROUND = [0.76, 0.68, 0.48, 1.0]
NIGHT_GROUND = [0.12, 0.12, 0.16, 1.0]


def apply_night_mode(is_night):
    p.changeVisualShape(plane, -1, rgbaColor=NIGHT_GROUND if is_night else DAY_GROUND)


apply_night_mode(NIGHT_MODE)

MOTORS = {
    "front_rightR": 0,
    "front_rightL": 3,
    "front_leftR": 6,
    "front_leftL": 9,
    "back_rightR": 12,
    "back_rightL": 15,
    "back_leftR": 18,
    "back_leftL": 21,
}

DIAGONAL_PHASE = {
    "front_right": 0.0,
    "back_left": 0.0,
    "front_left": math.pi,
    "back_right": math.pi,
}

# --- Gait tuning constants ---
AMPLITUDE = 0.2
FREQUENCY = 1.5
RL_PHASE_OFFSET = math.pi / 4
BASE_ANGLE = 0.0

NORMAL_SPEED = 1.4
BOOST_SPEED = 1.3
BOOST_DURATION = 2.0
STOP_DURATION = 4.0
DETECTION_RANGE = 0.8

# --- Obstacle avoidance tuning ---
AVOID_LOOKAHEAD = 1.2      # start steering around an obstacle this far before reaching it
AVOID_CLEAR_MARGIN = 1.0   # keep steering until this far past the obstacle
OBSTACLE_LANE_THRESHOLD = 0.6  # how close (in y) an obstacle must be to a robot's lane to matter
AVOID_OFFSET = 0.5         # how far sideways a robot shifts to go around an obstacle
LATERAL_KP = 1.5           # proportional gain: how eagerly it steers toward its target y
MAX_LATERAL_SPEED = 0.6    # cap on sideways speed (keeps it a gentle divert, not a swerve)

# ============================================================
# LOAD THE SWARM
# ============================================================

robots = {}
for name, y_offset in ROBOT_Y_OFFSETS.items():
    body_id = p.loadURDF("quadruped/quadruped.urdf", [START_X, y_offset, 0.5])
    robots[name] = {
        "body_id": body_id,
        "y_offset": y_offset,
        "state": "MOVING",       # MOVING | STOPPED
        "stop_end_time": 0.0,
        "boost_end_time": 0.0,
        "reached_destination": False,
    }

# ============================================================
# MINEFIELD: original 3 near the center path + 6 more scattered
# near the left/right paths so each robot can encounter its own.
# Visual-only (no collision) so robots never physically hit them.
# ============================================================

IMG_FOLDER = r"C:\Users\ddhan\OneDrive\Documents\c2c hackathon\img data"
image_files = sorted(os.listdir(IMG_FOLDER))


def cycle_image(i):
    return image_files[i % len(image_files)]


MINE_LAYOUT = [
    # near the center path (y ~ 0)
    {"id": "mine_c1", "image": cycle_image(0), "position": [2.5, 0.0, 0.03]},
    {"id": "mine_c2", "image": cycle_image(1), "position": [5.5, 0.4, 0.03]},
    {"id": "mine_c3", "image": cycle_image(2), "position": [8.0, -0.3, 0.03]},
    # near the left path (y ~ -2)
    {"id": "mine_l1", "image": cycle_image(0), "position": [2.0, -1.7, 0.03]},
    {"id": "mine_l2", "image": cycle_image(1), "position": [5.0, -2.3, 0.03]},
    {"id": "mine_l3", "image": cycle_image(2), "position": [8.5, -1.9, 0.03]},
    # near the right path (y ~ +2)
    {"id": "mine_r1", "image": cycle_image(0), "position": [3.5, 2.2, 0.03]},
    {"id": "mine_r2", "image": cycle_image(1), "position": [6.0, 1.8, 0.03]},
    {"id": "mine_r3", "image": cycle_image(2), "position": [9.0, 2.3, 0.03]},
]

mine_position_lookup = {}

for mine in MINE_LAYOUT:
    texture_path = os.path.join(IMG_FOLDER, mine["image"])
    mine_visual = p.createVisualShape(
        shapeType=p.GEOM_CYLINDER, radius=0.15, length=0.05, rgbaColor=[1, 1, 1, 1]
    )
    mine_body = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=mine_visual,
        basePosition=mine["position"],
    )
    texture_id = p.loadTexture(texture_path)
    p.changeVisualShape(mine_body, -1, textureUniqueId=texture_id)
    mine_position_lookup[mine["id"]] = mine["position"]

# --- Decoy mine, well off every robot's path (all paths sit within y = -2..+2) ---
DECOY_IMAGE = cycle_image(0)
DECOY_POSITION = [4.5, 4.8, 0.03]
decoy_texture_path = os.path.join(IMG_FOLDER, DECOY_IMAGE)
decoy_visual = p.createVisualShape(
    shapeType=p.GEOM_CYLINDER, radius=0.15, length=0.05, rgbaColor=[1, 1, 1, 1]
)
decoy_body = p.createMultiBody(
    baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=decoy_visual, basePosition=DECOY_POSITION
)
decoy_texture_id = p.loadTexture(decoy_texture_path)
p.changeVisualShape(decoy_body, -1, textureUniqueId=decoy_texture_id)

# ============================================================
# OBSTACLES: trees and stones placed directly in each robot's
# path. Visual-only (no collision) — avoidance is handled by
# scripted lateral steering below, the same safe approach used
# for the mines, so nothing can cause a physics-flip on contact.
# ============================================================

OBSTACLES = [
    {"type": "tree", "position": [3.0, 0.05, 0]},    # center lane
    {"type": "stone", "position": [7.0, -0.1, 0]},   # center lane
    {"type": "stone", "position": [4.5, -2.1, 0]},   # left lane
    {"type": "tree", "position": [7.5, 2.1, 0]},      # right lane
]

for obs in OBSTACLES:
    ox, oy, oz = obs["position"]
    if obs["type"] == "tree":
        trunk_visual = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER, radius=0.08, length=0.6, rgbaColor=[0.45, 0.28, 0.1, 1.0]
        )
        p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=trunk_visual,
            basePosition=[ox, oy, 0.3],
        )
        canopy_visual = p.createVisualShape(
            shapeType=p.GEOM_SPHERE, radius=0.35, rgbaColor=[0.13, 0.5, 0.13, 1.0]
        )
        p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=canopy_visual,
            basePosition=[ox, oy, 0.7],
        )
    else:  # stone
        stone_visual = p.createVisualShape(
            shapeType=p.GEOM_SPHERE, radius=0.22, rgbaColor=[0.5, 0.5, 0.5, 1.0]
        )
        p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=stone_visual,
            basePosition=[ox, oy, 0.18],
        )

# ============================================================
# REAL DETECTION: OpenCV ORB feature matching against reference photos
# ============================================================

orb = cv2.ORB_create(nfeatures=800)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

reference_descriptors = []
for filename in image_files:
    path = os.path.join(IMG_FOLDER, filename)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    kp, des = orb.detectAndCompute(img, None)
    if des is not None:
        reference_descriptors.append((filename, des))

print(f"Loaded {len(reference_descriptors)} reference landmine images for detection.")
print(f"Swarm of {len(robots)} robots deployed. Minefield contains {len(MINE_LAYOUT)} mines + 1 decoy.")

MATCH_THRESHOLD = 8
already_reported = set()  # mine ids that have already been reported (shared across the whole swarm)


def check_for_landmine(camera_frame_bgr, only_filename=None):
    gray = cv2.cvtColor(camera_frame_bgr, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)
    if des is None:
        return False, 0

    candidates = reference_descriptors
    if only_filename is not None:
        candidates = [(name, d) for name, d in reference_descriptors if name == only_filename]

    best_count = 0
    for filename, ref_des in candidates:
        matches = bf.match(des, ref_des)
        good_matches = [m for m in matches if m.distance < 70]
        if len(good_matches) > best_count:
            best_count = len(good_matches)

    return best_count >= MATCH_THRESHOLD, best_count


def get_camera_image(body_id, width, height, renderer):
    base_pos, base_orn = p.getBasePositionAndOrientation(body_id)
    cam_eye = [base_pos[0], base_pos[1], base_pos[2] + 0.3]
    cam_target = [base_pos[0] + 1.0, base_pos[1], base_pos[2]]

    view_matrix = p.computeViewMatrix(cam_eye, cam_target, [0, 0, 1])
    proj_matrix = p.computeProjectionMatrixFOV(fov=70, aspect=1.0, nearVal=0.05, farVal=5.0)

    width, height, rgb_img, _, _ = p.getCameraImage(
        width=width, height=height, viewMatrix=view_matrix, projectionMatrix=proj_matrix, renderer=renderer
    )
    rgb_array = np.reshape(rgb_img, (height, width, 4))[:, :, :3].astype(np.uint8)
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


# ============================================================
# LIVE CAMERA WINDOWS — one per robot, arranged across the top
# ============================================================

DISPLAY_SCALE = 4
window_positions = {"left": 20, "center": 460, "right": 900}
for name in robots:
    window_name = f"Camera Feed - {name.upper()}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 128 * DISPLAY_SCALE, 96 * DISPLAY_SCALE)
    cv2.moveWindow(window_name, window_positions[name], 20)
    robots[name]["window_name"] = window_name

# --- Plain mission dashboard: just the numbers, no styling ---
DASHBOARD_WINDOW = "Mission Dashboard"
DASHBOARD_WIDTH = 480
DASHBOARD_HEIGHT = 260
cv2.namedWindow(DASHBOARD_WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(DASHBOARD_WINDOW, DASHBOARD_WIDTH, DASHBOARD_HEIGHT)
cv2.moveWindow(DASHBOARD_WINDOW, 20, 460)


def draw_dashboard():
    canvas = np.zeros((DASHBOARD_HEIGHT, DASHBOARD_WIDTH, 3), dtype=np.uint8)
    lines = [
        f"Robots deployed: {len(robots)}",
        f"Mines detected: {len(already_reported)}",
        f"Mission time: {time.time() - start_time:.1f}s",
        f"Left  x: {robots['left'].get('last_x', 0):.2f}",
        f"Center x: {robots['center'].get('last_x', 0):.2f}",
        f"Right x: {robots['right'].get('last_x', 0):.2f}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            canvas, line, (15, 40 + i * 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )
    cv2.imshow(DASHBOARD_WINDOW, canvas)

# ============================================================
# FINAL MISSION MAP: plots detected landmines and obstacles once
# every robot in the swarm has reached the destination
# ============================================================

def show_final_map():
    fig, ax = plt.subplots(figsize=(9, 6))

    landmine_xs = [mine_position_lookup[mid][0] for mid in already_reported]
    landmine_ys = [mine_position_lookup[mid][1] for mid in already_reported]
    ax.scatter(
        landmine_xs, landmine_ys, marker="^", color="red", s=180,
        label="Detected Landmine", zorder=5, edgecolors="black",
    )

    tree_xs = [o["position"][0] for o in OBSTACLES if o["type"] == "tree"]
    tree_ys = [o["position"][1] for o in OBSTACLES if o["type"] == "tree"]
    stone_xs = [o["position"][0] for o in OBSTACLES if o["type"] == "stone"]
    stone_ys = [o["position"][1] for o in OBSTACLES if o["type"] == "stone"]
    ax.scatter(tree_xs, tree_ys, marker="o", color="green", s=160, label="Tree", zorder=5, edgecolors="black")
    ax.scatter(stone_xs, stone_ys, marker="s", color="gray", s=140, label="Stone", zorder=5, edgecolors="black")

    # Reference lines showing each robot's patrol lane
    for name, y in ROBOT_Y_OFFSETS.items():
        ax.axhline(y=y, color="blue", linestyle="--", alpha=0.25)
        ax.text(END_X + 0.2, y, name, color="blue", va="center", fontsize=9)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("X coordinate (m)")
    ax.set_ylabel("Y coordinate (m)")
    ax.set_title("Mission Summary: Detected Landmines & Obstacles")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, END_X + 2.5)
    ax.set_ylim(-4, 5)
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN LOOP
# ============================================================

start_time = time.time()
step_counter = 0
CAMERA_UPDATE_INTERVAL = 4  # ~60 updates/sec at 240Hz sim rate

while True:
    t = time.time() - start_time
    now = time.time()

    # Press 'N' at any time to toggle night mode and see the fluorescent
    # markers' visibility advantage in low light
    keys = p.getKeyboardEvents()
    if ord("n") in keys and keys[ord("n")] & p.KEY_WAS_TRIGGERED:
        NIGHT_MODE = not NIGHT_MODE
        apply_night_mode(NIGHT_MODE)
        print(f"Night mode {'ON' if NIGHT_MODE else 'OFF'}")

    for name, robot in robots.items():
        body_id = robot["body_id"]

        # --- Gait: drive all 8 leg motors ---
        for leg_side, joint_index in MOTORS.items():
            if leg_side.endswith("R"):
                leg_group = leg_side[:-1]
                side_offset = 0.0
            else:
                leg_group = leg_side[:-1]
                side_offset = RL_PHASE_OFFSET
            phase = DIAGONAL_PHASE[leg_group]
            angle = BASE_ANGLE + AMPLITUDE * math.sin(2 * math.pi * FREQUENCY * t + phase + side_offset)
            p.setJointMotorControl2(
                bodyUniqueId=body_id, jointIndex=joint_index, controlMode=p.POSITION_CONTROL,
                targetPosition=angle, force=35,
            )

        base_pos, _ = p.getBasePositionAndOrientation(body_id)
        robot["last_x"] = base_pos[0]
        robot["last_y"] = base_pos[1]

        if not robot["reached_destination"] and base_pos[0] >= END_X:
            robot["reached_destination"] = True
            print(f"[{name.upper()}] Reached destination at ({base_pos[0]:.2f}, {base_pos[1]:.2f})")

        if robot["state"] == "STOPPED" and now >= robot["stop_end_time"]:
            robot["state"] = "MOVING"
            robot["boost_end_time"] = now + BOOST_DURATION

        if robot["reached_destination"] or robot["state"] == "STOPPED":
            current_speed = 0.0
        elif now < robot["boost_end_time"]:
            current_speed = BOOST_SPEED
        else:
            current_speed = NORMAL_SPEED

        current_vel, current_ang_vel = p.getBaseVelocity(body_id)

        # --- Obstacle avoidance: steer sideways around anything in this
        # robot's own lane, then settle back onto the original path ---
        home_lane_y = robot["y_offset"]
        target_y = home_lane_y
        for obs in OBSTACLES:
            ox, oy, oz = obs["position"]
            if abs(oy - home_lane_y) > OBSTACLE_LANE_THRESHOLD:
                continue  # this obstacle isn't in this robot's lane at all
            if (ox - AVOID_LOOKAHEAD) <= base_pos[0] <= (ox + AVOID_CLEAR_MARGIN):
                # Steer to whichever side of the obstacle is away from lane center
                steer_sign = -1.0 if oy >= home_lane_y else 1.0
                target_y = home_lane_y + steer_sign * AVOID_OFFSET
                break  # only handle one obstacle at a time

        lateral_error = target_y - base_pos[1]
        vy = max(-MAX_LATERAL_SPEED, min(MAX_LATERAL_SPEED, LATERAL_KP * lateral_error))

        p.resetBaseVelocity(
            body_id, linearVelocity=[current_speed, vy, current_vel[2]], angularVelocity=current_ang_vel
        )

        # --- Camera + detection (throttled) ---
        if step_counter % CAMERA_UPDATE_INTERVAL == 0:
            frame = get_camera_image(body_id, 128, 96, p.ER_TINY_RENDERER)
            display_frame = cv2.resize(
                frame, (128 * DISPLAY_SCALE, 96 * DISPLAY_SCALE), interpolation=cv2.INTER_NEAREST
            )
            cv2.imshow(robot["window_name"], display_frame)

            if not robot["reached_destination"] and robot["state"] == "MOVING":
                nearby_mine = None
                for mine in MINE_LAYOUT:
                    if mine["id"] in already_reported:
                        continue
                    mx, my, mz = mine["position"]
                    if abs(base_pos[0] - mx) < DETECTION_RANGE and abs(base_pos[1] - my) < DETECTION_RANGE:
                        nearby_mine = mine
                        break

                if nearby_mine is not None:
                    detection_frame = get_camera_image(body_id, 320, 240, p.ER_TINY_RENDERER)
                    found, match_count = check_for_landmine(detection_frame, only_filename=nearby_mine["image"])

                    if found:
                        already_reported.add(nearby_mine["id"])
                        mine_pos = mine_position_lookup[nearby_mine["id"]]

                        print(
                            f"[{name.upper()}] LANDMINE DETECTED — {nearby_mine['id']} "
                            f"({match_count} keypoint matches) at coordinates "
                            f"x={mine_pos[0]:.2f}, y={mine_pos[1]:.2f}"
                        )

                        if HAS_WINSOUND:
                            winsound.Beep(1000, 200)  # 1kHz beep, 200ms — audible detection cue

                        p.addUserDebugText(
                            f"({mine_pos[0]:.2f}, {mine_pos[1]:.2f})",
                            [mine_pos[0], mine_pos[1], mine_pos[2] + 0.5],
                            textColorRGB=[1, 0, 0],
                            textSize=0.75,
                        )

                        # Fluorescent ground marker
                        marker_visual = p.createVisualShape(
                            shapeType=p.GEOM_CYLINDER, radius=0.35, length=0.01,
                            rgbaColor=[1.0, 1.0, 0.0, 1.0],
                        )
                        p.createMultiBody(
                            baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=marker_visual,
                            basePosition=[mine_pos[0], mine_pos[1], 0.005],
                        )

                        robot["state"] = "STOPPED"
                        robot["stop_end_time"] = now + STOP_DURATION

    if step_counter % CAMERA_UPDATE_INTERVAL == 0:
        draw_dashboard()

    cv2.waitKey(1)
    step_counter += 1
    p.stepSimulation()
    time.sleep(1 / 240)

    if all(r["reached_destination"] for r in robots.values()):
        print("All robots have reached the destination. Generating mission summary map...")
        cv2.destroyAllWindows()
        show_final_map()
        break
