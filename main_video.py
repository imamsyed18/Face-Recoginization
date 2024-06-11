import cv2
from simple_facerec import SimpleFacerec


def display_message_box(message):
    cv2.namedWindow("Message", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Message", 600, 200)
    cv2.putText(message, "Press any key to continue...", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Message", message)
    cv2.waitKey(0)
    cv2.destroyWindow("Message")

def capture_and_process(cap, sfr):
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame")
            break

        face_locations, face_names = sfr.detect_known_faces(frame)
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break



def main():
    try:

        sfr = SimpleFacerec()
        sfr.load_encoding_images("images/")


        cap = cv2.VideoCapture(0)


        if not cap.isOpened():
            print("Error: Unable to open camera")
            return


        capture_and_process(cap, sfr)

       
        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print("Error:", str(e))
        display_message_box("An error occurred. Please check the console for details.")


if __name__ == "__main__":
    main()



