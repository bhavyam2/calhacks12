import cv2
for idx in range(4):
    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    ok, frame = cap.read()
    print(f"Index {idx}: Opened={cap.isOpened()} FrameOK={ok}")
    cap.release()