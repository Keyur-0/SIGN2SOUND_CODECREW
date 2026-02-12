import cv2
import numpy as np
import mediapipe as mp


class HandLandmarkExtractor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # store previous frame landmarks for motion calculation
        self.prev_landmarks = []
        self.last_finger_states = None

    def _landmarks_to_vector(self, hand_landmarks):
        vec = []
        for lm in hand_landmarks.landmark:
            vec.extend([lm.x, lm.y, lm.z])
        return np.array(vec, dtype=np.float32)

    def _motion_score(self, current, previous):
        """
        Motion = mean L2 distance between frames
        """
        if previous is None:
            return 0.0
        return np.mean(np.linalg.norm(current - previous, axis=1))

    def _dist(self, p1, p2):
        return np.linalg.norm(p1[:2] - p2[:2])
    
    def _finger_states(self, pts):
        """
        Returns binary finger extension states:
        [thumb, index, middle, ring, pinky]
        """
        wrist = pts[0]

        fingers = {
            "thumb":  (pts[4], pts[2]),
            "index":  (pts[8], pts[6]),
            "middle": (pts[12], pts[10]),
            "ring":   (pts[16], pts[14]),
            "pinky":  (pts[20], pts[18]),
        }

        states = []
        for tip, pip in fingers.values():
            states.append(1.0 if tip[1] < pip[1] else 0.0)

        return np.array(states, dtype=np.float32)

    def extract(self, frame):
        """
        Returns: (126,) feature vector
        [ dominant_hand (63) | secondary_hand (63 or zeros) ]
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        # default: no hands
        output = np.zeros(126, dtype=np.float32)

        if not results.multi_hand_landmarks:
            self.prev_landmarks = []
            return output

        hand_vectors = []
        hand_points = []

        for hand in results.multi_hand_landmarks:
            vec = self._landmarks_to_vector(hand)
            pts = vec.reshape(21, 3)
            finger_states = self._finger_states(pts)
            self.last_finger_states = finger_states
            vec = vec  # DO NOT concatenate into model input
            thumb_tip  = pts[4]
            index_tip  = pts[8]
            middle_tip = pts[12]
            ring_tip   = pts[16]
            pinky_tip  = pts[20]

            wrist = pts[0]
            hand_vectors.append(vec)
            hand_points.append(pts)

        # ---------- ONE HAND ----------
        if len(hand_vectors) == 1:
            dominant = hand_vectors[0]
            self.prev_landmarks = [hand_points[0]]
            output[:63] = dominant
            return output

        # ---------- TWO HANDS ----------
        # compute motion score for each hand
        motion_scores = []

        for i, pts in enumerate(hand_points):
            prev = self.prev_landmarks[i] if i < len(self.prev_landmarks) else None
            motion_scores.append(self._motion_score(pts, prev))

        # dominant = hand with MORE motion
        dominant_idx = int(np.argmax(motion_scores))
        secondary_idx = 1 - dominant_idx

        # --- geometry features ---
        # thumb_index_dist  = self._dist(thumb_tip, index_tip)
        # index_middle_dist = self._dist(index_tip, middle_tip)
        # middle_ring_dist  = self._dist(middle_tip, ring_tip)

        # # palm size (for scale normalization)
        # palm_size = self._dist(wrist, middle_tip)

        # # normalize
        # thumb_index_dist  /= palm_size + 1e-6
        # index_middle_dist /= palm_size + 1e-6
        # middle_ring_dist  /= palm_size + 1e-6

        # geometry = np.array([
        #     thumb_index_dist,
        #     index_middle_dist,
        #     middle_ring_dist
        # ], dtype=np.float32)
        # dominant = hand_vectors[dominant_idx]
        # secondary = hand_vectors[secondary_idx]

        # output[:63] = dominant
        # output[63:] = secondary

        # # dominant = hand_vectors[dominant_idx]
        # dominant = np.concatenate([hand_vectors[dominant_idx], geometry])
        # secondary = hand_vectors[secondary_idx]

        # output[:63] = dominant
        # output[63:] = secondary

        # self.prev_landmarks = hand_points
        # return output
        # ---------- TWO HANDS ----------
        dominant = hand_vectors[dominant_idx]
        secondary = hand_vectors[secondary_idx]

        output[:63] = dominant
        output[63:] = secondary

        self.prev_landmarks = hand_points
        return output