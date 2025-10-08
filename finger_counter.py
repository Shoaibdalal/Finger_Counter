import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.75)

fingertips = [4, 8, 12, 16, 20]

cap = cv2.VideoCapture(0)

bg_color = (10, 10, 10)
box_color = (0, 255, 100)
text_color = (255, 255, 255)

def count_fingers(hand_landmarks, hand_label):
    fingers = 0
    lm = hand_landmarks.landmark

    if hand_label == "Right":
        if lm[fingertips[0]].x < lm[fingertips[0] - 1].x:
            fingers += 1
    else:
        if lm[fingertips[0]].x > lm[fingertips[0] - 1].x:
            fingers += 1

    for tip_id in fingertips[1:]:
        if lm[tip_id].y < lm[tip_id - 2].y:
            fingers += 1

    return fingers

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    total_fingers = 0

    if result.multi_hand_landmarks and result.multi_handedness:
        for landmarks, hand_handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            label = hand_handedness.classification[0].label  
            count = count_fingers(landmarks, label)
            total_fingers += count

            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                                   mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2))

    cv2.rectangle(frame, (20, 20), (300, 120), bg_color, -1)
    cv2.rectangle(frame, (20, 20), (300, 120), box_color, 3)

    cv2.putText(frame, f"Fingers: {total_fingers}", (40, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 2, text_color, 4)

    cv2.imshow("Attractive Finger Counter", frame)

    if cv2.waitKey(1) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()
