import cv2
import mediapipe as mp
import time

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

CTime = 0
PTime = 0

while True:
    success, frame = cap.read()

    if frame is None:
        break

    RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(RGBframe)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

            for id, lm in enumerate(handLms.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)

                cv2.circle(frame, (cx, cy), 3, (255, 0, 0), cv2.FILLED)

    CTime = time.time()
    fps = 1 / (CTime - PTime)
    PTime = CTime

    cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("image", frame)
    if cv2.waitKey(1) == ord("q"):
        break



cv2.destroyAllWindows()
cap.release()
