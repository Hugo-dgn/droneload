import cv2
import droneload.video.process as process
import numpy as np
import droneload.video.rectangle_analysis as rectangle_analysis

def video_contours():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        pool_size = 1
        if pool_size > 1:
            image = image[::pool_size, ::pool_size]

        contours = process.get_contours_sobel(image)
        cv2.imshow('Contours', np.kron(contours, np.ones((pool_size, pool_size))))

        if cv2.waitKey(50) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def video_rectangle(draw_arrow=True):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        _frame = frame.copy()
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        pool_size = 1
        if pool_size > 1:
            image = image[::pool_size, ::pool_size]

        contours = process.get_contours_sobel(image)

        rects = pool_size*process.find_rectangle(contours)

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
    
    cv2.imwrite("data/rect_dected.jpg", frame)
    cv2.imwrite("data/rect_not_dected.jpg", _frame)
    
    cap.release()
    cv2.destroyAllWindows()

def find_rectanle(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    contours = process.get_contours_sobel(image)

    rects = process.find_rectangle(contours)

    for rect in rects:
        pts = rect.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
    cv2.imshow('Contours', frame)
