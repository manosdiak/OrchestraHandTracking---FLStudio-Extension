"""
Hand-gesture MIDI controller
-----------------------------
Webcam -> MediaPipe Hands -> gesture logic -> MIDI -> virtual MIDI port -> FL Studio

Mapping:
  RIGHT hand vertical    -> global volume (continuous, CC7, broadcast to all instrument ports)
  RIGHT hand horizontal  -> instrument   (discrete zones, switches MIDI output channel)
  LEFT hand vertical   -> key/transpose(discrete zones, semitone shift)
  LEFT hand horizontal -> note         (discrete zones, Note On/Off, arpeggiator-style)

Setup:
  pip install opencv-python mediapipe mido python-rtmidi

  Download the hand-tracking model once (recent mediapipe pip releases have a
  known, still-open bug where the old mp.solutions API doesn't load at all --
  see https://github.com/google-ai-edge/mediapipe/issues/6261 -- so this
  script uses the newer, actively maintained Tasks API instead, which needs
  this model file):
      curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
  (or just paste that URL into a browser and save it as hand_landmarker.task
  in the same folder as this script)

  Windows: install loopMIDI (https://www.tobias-erichsen.de/software/loopmidi.html),
           create portS named e.g. "HandMIDI-0", "HandMIDI-1", etc., then this script will find them.
  Mac:     Audio MIDI Setup -> MIDI Studio -> double-click IAC Driver -> check
           "Device is online". This script opens a virtual port directly.
  Linux:   `sudo modprobe snd-virmidi` or just let mido open a virtual port.

  In FL Studio: Options > MIDI Settings > enable each HandMIDI-N port and give
  it a matching PORT number. For each Channel Rack instrument, open its own
  plugin settings > MIDI tab > set Input Port to that same number. To make the
  volume truly global, either link all target mixer faders to the same CC7
  source, or (preferably) link CC7 to the Master mixer fader. This script sends
  the same CC7 value to every instrument port so the control is shared.
"""

import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import mido
import time

# ---------------------------------------------------------------------------
# Configuration - tune these to taste
# ---------------------------------------------------------------------------

NUM_INSTRUMENTS = 6          # left/right zones for instrument select
NUM_KEY_STEPS = 12           # semitone range for key/transpose (right hand Y)
SCALE = [0, 2, 4, 5, 7, 9, 11]   # major scale intervals (right hand X picks a scale degree)
BASE_NOTE = 60               # C4
SMOOTHING = 0.35             # 0-1, higher = more responsive / less smooth
HYSTERESIS = 0.03            # fraction of screen width/height a hand must cross
                              # past a zone boundary before switching zones (anti-flicker)
CAM_INDEX = 0
MIDI_PORT_NAME = "HandMIDI"  # loopMIDI port name on Windows; on Mac/Linux mido
                              # will open a virtual port with this name directly
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# ---------------------------------------------------------------------------
# MIDI setup -- one virtual port PER INSTRUMENT (see setup notes above: FL
# Studio routes instruments reliably by MIDI PORT, not MIDI channel, so each
# instrument gets its own loopMIDI/IAC cable named HandMIDI-0, HandMIDI-1, ...)
# ---------------------------------------------------------------------------

def open_midi_port(name):
    available = mido.get_output_names()
    for full_name in available:
        if name.lower() in full_name.lower():
            return mido.open_output(full_name)
    try:
        return mido.open_output(name, virtual=True)
    except Exception as e:
        raise RuntimeError(
            f"Could not open or create a MIDI port named '{name}'. "
            f"On Windows, create it first in loopMIDI. Available ports: {available}"
        ) from e


midi_outs = {i: open_midi_port(f"{MIDI_PORT_NAME}-{i}") for i in range(NUM_INSTRUMENTS)}

# ---------------------------------------------------------------------------
# MediaPipe setup (Tasks API -- see note in the setup instructions above about
# why this doesn't use the older mp.solutions.hands interface)
# ---------------------------------------------------------------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. Download it with:\n"
        f"  curl -L -o hand_landmarker.task "
        f"https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)
landmarker = mp_vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(CAM_INDEX)
frame_timestamp_ms = 0


def zone_index(value, num_zones):
    """value in [0,1] -> integer zone index in [0, num_zones-1]"""
    idx = int(value * num_zones)
    return max(0, min(num_zones - 1, idx))


class HandState:
    """Tracks smoothed position + current discrete zone (with hysteresis) for one hand."""

    def __init__(self, num_zones_x, num_zones_y):
        self.x = None
        self.y = None
        self.zone_x = 0
        self.zone_y = 0
        self.num_zones_x = num_zones_x
        self.num_zones_y = num_zones_y

    def update(self, raw_x, raw_y):
        if self.x is None:
            self.x, self.y = raw_x, raw_y
        else:
            self.x += (raw_x - self.x) * SMOOTHING
            self.y += (raw_y - self.y) * SMOOTHING

        new_zone_x = zone_index(self.x, self.num_zones_x)
        new_zone_y = zone_index(self.y, self.num_zones_y)

        changed_x = False
        changed_y = False

        if new_zone_x != self.zone_x:
            boundary = new_zone_x / self.num_zones_x
            if abs(self.x - boundary) > HYSTERESIS:
                self.zone_x = new_zone_x
                changed_x = True

        if new_zone_y != self.zone_y:
            boundary = new_zone_y / self.num_zones_y
            if abs(self.y - boundary) > HYSTERESIS:
                self.zone_y = new_zone_y
                changed_y = True

        return changed_x, changed_y


left_state = HandState(NUM_INSTRUMENTS, 1)      # x=instrument zones, y unused (continuous)
right_state = HandState(len(SCALE), NUM_KEY_STEPS)  # x=note zones, y=key zones

current_instrument = 0
current_key_shift = 0
current_note_playing = None
current_note_port = None
last_volume_sent = None

print("Running. Press 'q' in the video window to quit.")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)  # mirror view, more intuitive to use
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        frame_timestamp_ms += 33  # ~30fps; must be monotonically increasing
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        if result.hand_landmarks and result.handedness:
            for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                # Because we flipped the frame, MediaPipe's Left/Right label
                # (computed on the flipped image) already matches the user's
                # actual left/right hand as seen in a mirror.
                label = handedness[0].category_name  # "Left" or "Right"

                wrist = landmarks[0]  # landmark index 0 = wrist
                x, y = wrist.x, wrist.y  # both in [0,1]

                # simple landmark dots (no drawing_utils available in Tasks API)
                h, w, _ = frame.shape
                for lm in landmarks:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)

                if label == "Left":
                    # vertical -> volume (continuous), sent to whichever
                    # instrument's port is currently selected
                    volume = int((1 - y) * 127)  # y=0 is top of frame -> hand up = louder
                    volume = max(0, min(127, volume))
                    if last_volume_sent is None or abs(volume - last_volume_sent) >= 2:
                        # Global volume: broadcast the same CC7 value to every
                        # instrument port instead of only the selected instrument.
                        for out in midi_outs.values():
                            out.send(
                                mido.Message('control_change', control=7, value=volume, channel=0)
                            )
                        last_volume_sent = volume

                    # horizontal -> instrument (discrete): picks which port
                    # subsequent notes/CC are sent to
                    changed_x, _ = left_state.update(x, y)
                    if changed_x:
                        new_instrument = left_state.zone_x

                        # If a note is currently sounding, stop it on the OLD
                        # instrument port before switching to the new one.
                        if current_note_playing is not None and current_note_port is not None:
                            midi_outs[current_note_port].send(
                                mido.Message(
                                    'note_off',
                                    note=current_note_playing,
                                    channel=0
                                )
                            )
                            current_note_playing = None
                            current_note_port = None

                        current_instrument = new_instrument
                        print(f"Instrument zone -> {current_instrument} (port {MIDI_PORT_NAME}-{current_instrument})")

                elif label == "Right":
                    changed_x, changed_y = right_state.update(x, y)

                    if changed_y:
                        # zone centered around 0 -> map to a symmetric transpose range
                        current_key_shift = right_state.zone_y - (NUM_KEY_STEPS // 2)
                        print(f"Key shift -> {current_key_shift} semitones")

                    if changed_x:
                        degree = right_state.zone_x
                        note = BASE_NOTE + SCALE[degree] + current_key_shift
                        out = midi_outs[current_instrument]

                        if current_note_playing is not None and current_note_port is not None:
                            # Send note-off to the port that actually owns the
                            # currently sounding note.
                            midi_outs[current_note_port].send(
                                mido.Message('note_off', note=current_note_playing, channel=0)
                            )

                        out.send(mido.Message('note_on', note=note, velocity=100, channel=0))
                        current_note_playing = note
                        current_note_port = current_instrument
                        print(f"Note on -> {note} (instrument {current_instrument})")

        cv2.imshow("Hand MIDI Controller", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    if current_note_playing is not None and current_note_port is not None:
        midi_outs[current_note_port].send(
            mido.Message('note_off', note=current_note_playing, channel=0)
        )
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    for out in midi_outs.values():
        out.close()