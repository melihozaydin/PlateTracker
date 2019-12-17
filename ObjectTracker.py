import sys
import cv2
import os
import darknetcv3 as dn
import numpy as np
import time
import serial

"""
results= tuple
results= ((label,conf,(x,y,width,height)), (label,conf,(x,y,width,height)), (label,conf,(x,y,width,height)))
"""

configPath = "cfg/silah-yolov3-tiny.cfg"
weightPath = "bin/silah-yolov3-tiny_final.weights"
metaPath = "data/silah-yolov3-tiny-obj.data"
thresh = 0.35

'''
configPath = "cfg/yolov3.cfg"
weightPath = "bin/yolov3.weights"
metaPath = "cfg/coco.data"
'''

# detect = ['Insan', 'Sirt Cantasi', 'Semsiye', 'El Cantasi', 'Kiravat', 'Bavul', 'Sise']
detect = ['silah']

colors = [tuple(255 * np.random.rand(3)) for i in range(5)]
capture = cv2.VideoCapture(1)

midScreenWindow = 15  # hizalama toleransi

port = '/dev/ttyACM0'
ard = serial.Serial(port, 115200, timeout=100)
time.sleep(1)


def move(step, _dir):

    data = str(step) + ', '+ str(_dir) + '\n'
    ard.write(data.encode("utf-8"))
    ard.flush()
"""
    print('Data', data)
    print('Byte Data ', bytes(data, 'utf-8'))
"""


if __name__ == '__main__':

    i = 0

    while (capture.isOpened()):
        stime = time.time()
        ret, frame = capture.read()
        i = i + 40
        if ret:
            midScreenX = (frame.shape[1]/2)
            midBox = None

            results = dn.performDetect(
                frame, thresh, configPath, weightPath, metaPath, showImage=False)

            for color, result in zip(colors, results):
                if (result[0] in detect):
                    tl = (int(result[2][0] - (result[2][2]/2)),
                          int(result[2][1] - (result[2][3]/2)))  # topleft x,y
                    br = (int(result[2][0] + (result[2][2]/2)),
                          int(result[2][1] + (result[2][3]/2)))  # br x,y
                    label = result[0]
                    confidence = result[1]
                    text = '{}: {:.0f}%'.format(label, confidence * 100)

                    frame = cv2.rectangle(frame, tl, br, color, 7)
                    frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX,
                                        1, (0, 0, 0), 2)

                    midBox = (result[2][0], result[2][1])

            if midBox is not None and i > -1:
                i = 0
                midBoxX = midBox[0]

                # Right
                if(midBoxX > (midScreenX - midScreenWindow)):
                    print(str(midBoxX) + " > " + str(midScreenX) + " : Pan Right : ")
                    move(1, 1)


                # Left
                elif(midBoxX < (midScreenX + midScreenWindow)):
                    print(str(midBoxX) + " < " + str(midScreenX)
                    + " : Pan Left : ")
                    move(1, 0)

                else:
                    print(str(midBoxX) + " ~ " + str(midScreenX) + " : ")

            cv2.imshow('frame', frame)
            print('FPS {:.1f}'.format(1 / (time.time() - stime)))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                ard.close()
                break
        else:
            capture.release()
            cv2.destroyAllWindows()
            break
