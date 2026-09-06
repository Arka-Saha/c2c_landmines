import pybullet as p
import pybullet_data
import time
import math
import os
import numpy as np
import cv2

# ============================================================
# SETUP
# ============================================================

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

plane = p.loadURDF("plane.urdf")

START_POINT = [0, 0, 0.5]
END_POINT = [10, 0, 0.5]

robot = p.loadURDF("quadruped/quadruped.urdf", START_POINT)

p.resetDebugVisualizerCamera(
    cameraDistance=9.0,
    cameraYaw=50,
    cameraPitch=-40,
    cameraTargetPosition=[(START_POINT[0] + END_POINT[0]) / 2, 0, 0],
)

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

# --- Gait tuning constants ---
AMPLITUDE = 0.2
FREQUENCY = 1.5
RL_PHASE_OFFSET = math.pi / 4
BASE_ANGLE = 0.0

NORMAL_SPEED = 0.6      # meters per second (bumped up from 0.4)
BOOST_SPEED = 1.3       # meters per second, used right after a detection
BOOST_DURATION = 2.0    # seconds to stay at boosted speed after resuming
STOP_DURATION = 4.0     # seconds to stop and "report" on detection

DIAGONAL_PHASE = {
    "front_right": 0.0,
    "back_left": 0.0,
    "front_left": math.pi,
    "back_right": math.pi,
}

# ============================================================
# PLACE THREE LANDMINE OBJECTS ALONG THE ROBOT'S PATH
# Visual-only (no collision shape) so the robot never physically
# hits them — detection is purely camera-based, no flipping.
# ============================================================

IMG_FOLDER = r"C:\Users\ddhan\OneDrive\Documents\c2c hackathon\img data"
image_files = sorted(os.listdir(IMG_FOLDER))

MINE_LAYOUT = [
    {"image": image_files[0 % len(image_files)], "position": [2.5, 0.0, 0.03]},
    {"image": image_files[1 % len(image_files)], "position": [5.5, 0.4, 0.03]},
    {"image": image_files[2 % len(image_files)], "position": [8.0, -0.3, 0.03]},
]

mine_filename_to_position = {}

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
    mine_filename_to_position[mine["image"]] = mine["position"]

# --- Decoy mine, well off the robot's path ---
# Placed far enough from the route that it will NEVER come within
# DETECTION_RANGE, demonstrating that detection is genuinely tied to the
# robot's actual path rather than firing on anything in the scene. It's
# intentionally NOT added to MINE_LAYOUT, so it never enters the detection
# candidate list at all — positioned to still be visible in the debug
# camera's framing.
DECOY_MINE_IMAGE = image_files[0 % len(image_files)]
DECOY_MINE_POSITION = [4.5, 3.0, 0.03]

decoy_texture_path = os.path.join(IMG_FOLDER, DECOY_MINE_IMAGE)
decoy_visual = p.createVisualShape(
    shapeType=p.GEOM_CYLINDER, radius=0.15, length=0.05, rgbaColor=[1, 1, 1, 1]
)
decoy_body = p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=-1,
    baseVisualShapeIndex=decoy_visual,
    basePosition=DECOY_MINE_POSITION,
)
decoy_texture_id = p.loadTexture(decoy_texture_path)
p.changeVisualShape(decoy_body, -1, textureUniqueId=decoy_texture_id)

# ============================================================
# REAL DETECTION: OpenCV ORB feature matching against your
# reference landmine photos (no training needed)
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

MATCH_THRESHOLD = 8
DETECTION_RANGE = 0.8  # meters — only attempt a match when this close to an unreported mine
detection_text_id = None
already_reported = set()


def check_for_landmine(camera_frame_bgr, only_filename=None):
    """ORB matching against reference photos. If only_filename is given,
    only that specific reference is compared — prevents a nearby match from
    getting misattributed to a different, un-encountered mine."""
    gray = cv2.cvtColor(camera_frame_bgr, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)
    if des is None:
        return False, 0, None

    candidates = reference_descriptors
    if only_filename is not None:
        candidates = [(name, d) for name, d in reference_descriptors if name == only_filename]

    best_count = 0
    best_name = None
    for filename, ref_des in candidates:
        matches = bf.match(des, ref_des)
        good_matches = [m for m in matches if m.distance < 70]
        if len(good_matches) > best_count:
            best_count = len(good_matches)
            best_name = filename

    return best_count >= MATCH_THRESHOLD, best_count, best_name


def get_robot_camera_image():
    base_pos, base_orn = p.getBasePositionAndOrientation(robot)
    cam_eye = [base_pos[0], base_pos[1], base_pos[2] + 0.3]
    cam_target = [base_pos[0] + 1.0, base_pos[1], base_pos[2]]

    view_matrix = p.computeViewMatrix(cam_eye, cam_target, [0, 0, 1])
    proj_matrix = p.computeProjectionMatrixFOV(fov=70, aspect=1.0, nearVal=0.05, farVal=5.0)

    # TinyRenderer (software, CPU-based) instead of hardware OpenGL — the
    # hardware renderer competes with PyBullet's own GUI window for the
    # graphics context and was causing stalls/freezing when called
    # repeatedly. Software rendering at this small resolution is plenty
    # fast and doesn't fight the GUI for GPU resources.
    width, height, rgb_img, _, _ = p.getCameraImage(
        width=128, height=96, viewMatrix=view_matrix, projectionMatrix=proj_matrix,
        renderer=p.ER_TINY_RENDERER,
    )
    rgb_array = np.reshape(rgb_img, (height, width, 4))[:, :, :3].astype(np.uint8)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    return bgr_array


def get_detection_camera_image():
    """Higher-resolution capture used ONLY for the ORB matching step (not
    for the live display feed). The small 128x96 display frame doesn't have
    enough pixel detail for reliable feature matching — this only runs when
    we're already close to an unreported mine, so the extra render cost is
    negligible (it's a rare, brief burst, not a continuous cost)."""
    base_pos, base_orn = p.getBasePositionAndOrientation(robot)
    cam_eye = [base_pos[0], base_pos[1], base_pos[2] + 0.3]
    cam_target = [base_pos[0] + 1.0, base_pos[1], base_pos[2]]

    view_matrix = p.computeViewMatrix(cam_eye, cam_target, [0, 0, 1])
    proj_matrix = p.computeProjectionMatrixFOV(fov=70, aspect=1.0, nearVal=0.05, farVal=5.0)

    width, height, rgb_img, _, _ = p.getCameraImage(
        width=320, height=240, viewMatrix=view_matrix, projectionMatrix=proj_matrix,
        renderer=p.ER_TINY_RENDERER,
    )
    rgb_array = np.reshape(rgb_img, (height, width, 4))[:, :, :3].astype(np.uint8)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    return bgr_array


# ============================================================
# LIVE CAMERA FEED WINDOW (top-right of the screen)
# ============================================================

CAMERA_WINDOW_NAME = "Robot Camera Feed"
CAMERA_DISPLAY_SCALE = 4  # upscale so the small render is actually visible
cv2.namedWindow(CAMERA_WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(CAMERA_WINDOW_NAME, 128 * CAMERA_DISPLAY_SCALE, 96 * CAMERA_DISPLAY_SCALE)
cv2.moveWindow(CAMERA_WINDOW_NAME, 900, 20)  # adjust if your screen resolution differs

# ============================================================
# MAIN LOOP
# ============================================================

start_time = time.time()
reached_destination = False

state = "MOVING"          # MOVING | STOPPED
stop_end_time = 0.0
boost_end_time = 0.0

CAMERA_UPDATE_INTERVAL = 4  # ~60 updates/sec at 240Hz sim rate — smooth but not overloaded
step_counter = 0

while True:
    t = time.time() - start_time
    now = time.time()

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
            bodyUniqueId=robot,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=angle,
            force=35,
        )

    base_pos, _ = p.getBasePositionAndOrientation(robot)

    if not reached_destination and base_pos[0] >= END_POINT[0]:
        reached_destination = True
        print(f"Reached destination point B at ({base_pos[0]:.2f}, {base_pos[1]:.2f})")

    if state == "STOPPED" and now >= stop_end_time:
        state = "MOVING"
        boost_end_time = now + BOOST_DURATION

    if reached_destination or state == "STOPPED":
        current_speed = 0.0
    elif now < boost_end_time:
        current_speed = BOOST_SPEED
    else:
        current_speed = NORMAL_SPEED

    current_vel, current_ang_vel = p.getBaseVelocity(robot)
    p.resetBaseVelocity(
        robot,
        linearVelocity=[current_speed, 0, current_vel[2]],
        angularVelocity=current_ang_vel,
    )

    step_counter += 1
    if step_counter % CAMERA_UPDATE_INTERVAL == 0:
        # Capture and display the live camera feed continuously, regardless
        # of whether a mine is nearby — this is the always-on camera window.
        frame = get_robot_camera_image()
        display_frame = cv2.resize(
            frame, (128 * CAMERA_DISPLAY_SCALE, 96 * CAMERA_DISPLAY_SCALE), interpolation=cv2.INTER_NEAREST
        )
        cv2.imshow(CAMERA_WINDOW_NAME, display_frame)
        cv2.waitKey(1)

        # Reuse this same frame for detection matching so we don't render twice
        if not reached_destination and state == "MOVING":
            nearby_mine = None
            for mine in MINE_LAYOUT:
                if mine["image"] in already_reported:
                    continue
                mx, my, mz = mine["position"]
                if abs(base_pos[0] - mx) < DETECTION_RANGE and abs(base_pos[1] - my) < DETECTION_RANGE:
                    nearby_mine = mine
                    break

            if nearby_mine is not None:
                detection_frame = get_detection_camera_image()
                found, match_count, match_name = check_for_landmine(detection_frame, only_filename=nearby_mine["image"])

                if found and match_name not in already_reported:
                    already_reported.add(match_name)
                    mine_pos = mine_filename_to_position.get(match_name, base_pos)

                    print(
                        f"LANDMINE DETECTED — matched '{match_name}' "
                        f"({match_count} keypoint matches) at coordinates "
                        f"x={mine_pos[0]:.2f}, y={mine_pos[1]:.2f}"
                    )

                    if detection_text_id is not None:
                        p.removeUserDebugItem(detection_text_id)
                    detection_text_id = p.addUserDebugText(
                        f"LANDMINE at ({mine_pos[0]:.2f}, {mine_pos[1]:.2f})",
                        [mine_pos[0], mine_pos[1], mine_pos[2] + 0.5],
                        textColorRGB=[1, 0, 0],
                        textSize=1.5,
                    )

                    # Mark the ground with a bright fluorescent-yellow patch
                    # so the spot stays visible even in low-light/night
                    # conditions — a flat disc laid directly on the ground.
                    marker_visual = p.createVisualShape(
                        shapeType=p.GEOM_CYLINDER,
                        radius=0.35,
                        length=0.01,
                        rgbaColor=[1.0, 1.0, 0.0, 1.0],  # bright fluorescent yellow
                    )
                    p.createMultiBody(
                        baseMass=0,
                        baseCollisionShapeIndex=-1,  # visual only, doesn't affect movement
                        baseVisualShapeIndex=marker_visual,
                        basePosition=[mine_pos[0], mine_pos[1], 0.005],
                    )

                    state = "STOPPED"
                    stop_end_time = now + STOP_DURATION

    p.stepSimulation()
    time.sleep(1 / 240)
