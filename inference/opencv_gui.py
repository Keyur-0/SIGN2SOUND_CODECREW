import cv2
import numpy as np

WINDOW_NAME = "SIGN2SOUND"
current_text = ""

def main():
    
    print("OpenCV GUI started")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        frame = np.ones((400, 700, 3), dtype=np.uint8) * 255

        cv2.putText(frame, "SIGN2SOUND - GUI Ready", (80,200), cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,0),2)

        cv2.putText(frame, current_text, (50, 320),cv2.FONT_HERSHEY_SIMPLEX,0.7,(20, 20, 20),2)

        cv2.putText(frame, "Press 'q' to quit", (220,260), cv2.FONT_HERSHEY_SIMPLEX,0.6,(50,50,50),1)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()