from detector import *

if __name__ == "__main__":
    cap = cv2.VideoCapture("test2.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("output.mp4", fourcc, 20, (1280, 720))

    while cap.isOpened():
        (ret, frame) = cap.read()

        if ret == True:
            frame = lane_detection(frame)
            cv2.imshow("frame", frame)
            out.write(frame)
        if cv2.waitKey(1) == ord("q") or ret == False:
            break

    cap.release()
    cv2.destroyAllWindows()