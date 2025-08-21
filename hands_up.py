import mediapipe as mp
import cv2

cap = cv2.VideoCapture(0)
mppose = mp.solutions.pose
pose = mppose.Pose()
mpdraw = mp.solutions.drawing_utils

while True:
  rec, frame = cap.read()
  frame = cv2.flip(frame, 1)

  rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  result = pose.process(rgb_frame)

  if result.pose_landmarks:
    mpdraw.draw_landmarks(frame, result.pose_landmarks, mppose.POSE_CONNECTIONS)

    llm = []
    hands = 0


    for id, lm in enumerate(result.pose_landmarks.landmark):
      h, w, c = frame.shape  # get frame shape
      cx, cy = int(lm.x * w), int(lm.y * h)  # convert to pixel coordinates
      llm.append([id, cx, cy])


    if llm[19][2] < llm[11][2]:
      hands += 1
    if llm[20][2] < llm[12][2]:
      hands += 1
    else:
      hands += 0

    if hands == 1:
      cv2.putText(frame, f"{hands} Hand is up", (50, 100),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 23), 3)
    elif hands == 2:
      cv2.putText(frame, f"{hands} Hands are up", (50, 100),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 23), 3)


  cv2.imshow("Video", frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break