import mediapipe as mp
import cv2

# Camera source
url = "" #Url of IP cam
cap = cv2.VideoCapture(0)   # change to 'url' if using IP cam

# Mediapipe setup
mphands = mp.solutions.hands
hands = mphands.Hands()
mpdraw = mp.solutions.drawing_utils

# Fingertip landmarks
tips = [8, 12, 16, 20]

while True:
    suc, frame = cap.read()
    if not suc:
        break

    # ==================== Preprocessing ====================
    frame = cv2.flip(frame, 1)
    sf = 0.7
    h, w = frame.shape[:2]
    h2, w2 = int(h * sf), int(w * sf)
    frame = cv2.resize(src=frame, dsize=(w2, h2), interpolation=cv2.INTER_CUBIC)

    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # ==================== Finger Counting ====================
    left_fingers = 0
    right_fingers = 0

    if results.multi_hand_landmarks:
        for handlms, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mpdraw.draw_landmarks(frame, handlms, mphands.HAND_CONNECTIONS)
            label = hand_handedness.classification[0].label

            llm = []
            fingers = 0
            for id, lm in enumerate(handlms.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                llm.append([id, cx, cy])

            # Thumb check
            if label == "Right":
                if llm[4][1] < llm[3][1]:
                    fingers += 1
            else:  # Left hand
                if llm[4][1] > llm[3][1]:
                    fingers += 1

            # Other fingers
            for idtip in tips:
                if llm[idtip][2] < llm[idtip - 2][2]:
                    fingers += 1

            if label == "Left":
                left_fingers = fingers
            else:
                right_fingers = fingers

        cv2.putText(frame, f'Fingers left : {left_fingers}', (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 23), 3)
        cv2.putText(frame, f'Fingers right : {right_fingers}', (400, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 23), 3)
        cv2.putText(frame, f'Total fingers : {right_fingers + left_fingers}', (200, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 245), 3)

    # ==================== Show Frame ====================
    cv2.imshow("V", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
