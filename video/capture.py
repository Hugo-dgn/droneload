import cv2
import video.process as process
import numpy as np
import video.rectangle_analysis as rectangle_analysis
import locals


def video_contours():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = image[::locals.pool_size, ::locals.pool_size]

        contours = process.get_contours_sobel(image)
        cv2.imshow('Contours', np.kron(contours, np.ones((locals.pool_size, locals.pool_size))))

        if cv2.waitKey(50) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def video_rectangle(draw_arrow=True):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = image[::locals.pool_size, ::locals.pool_size]

        contours = process.get_contours_sobel(image)

        rects = locals.pool_size*process.find_rectangle(contours, locals.tol)

        for rect in rects:
            pts = rect.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            if draw_arrow:
                normale = 50*rectangle_analysis.find_normal(rect, locals.alpha)
                midpoint = np.mean(rect, axis=0)
                midpoint = midpoint.astype(np.int32)
                normale = normale.astype(np.int32)
                cv2.arrowedLine(frame, midpoint, midpoint+normale[0:2], (0, 255, 0), thickness=2, tipLength=0.2)
        cv2.imshow('Contours', frame)
        if cv2.waitKey(100) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

    cv2.imwrite('data/rect.jpg', frame)
