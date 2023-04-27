import cv2
import video.process as process
import numpy as np
import video.rectangle_analysis as rectangle_analysis

def video_contours():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contours = process.get_contours_sobel(image)

        cv2.imshow('Contours', contours.astype(np.uint8))

        if cv2.waitKey(50) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def video_rectangle(draw_arrow=True):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contours = process.get_contours_sobel(image)
        rects = process.find_rectangle(contours)

        rects = np.int0(rects)
        for rect in rects:
            pts = rect.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            if draw_arrow:
                normale = 50*rectangle_analysis.find_normal(rect, 1)
                midpoint = np.mean(rect, axis=0)
                midpoint = midpoint.astype(np.int32)
                normale = normale.astype(np.int32)
                cv2.arrowedLine(frame, midpoint, midpoint+normale[0:2], (0, 255, 0), thickness=2, tipLength=0.2)

        cv2.imshow('Contours', frame)

        if cv2.waitKey(100) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
