import cv2
import numpy as np
import pytesseract

port = '/dev/ttyACM0'
ard = serial.Serial(port, 115200, timeout=100)
time.sleep(1)

def move(step, _dir):

    data = str(step) + ', '+ str(_dir) + '\n'
    ard.write(data.encode("utf-8"))
    ard.flush()

def yolo_inference(input_image, classesFile="classes.names", modelConfiguration="darknet-yolov3.cfg", modelWeights="lapi.weights", confThreshold=0.5):
    
    nmsThreshold = 0.4  #Non-maximum suppression threshold
    inpWidth = 416  #608     #Width of network's input image
    inpHeight = 416 #608     #Height of network's input image
    
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
    
    # Sets the input to the network
    net.setInput(blob)
    
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    OutputNames = [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # Runs the forward pass to get output of the output layers
    outs = net.forward(OutputNames)
    
    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    time = t * 1000.0 / cv2.getTickFrequency()
    
    # POSTPROCESS
    frameHeight = input_image.shape[0]
    frameWidth = input_image.shape[1]

    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        # print("out.shape : ", out.shape)
        for detection in out:
            #if detection[4]>0.001:
            scores = detection[5:]
            classId = np.argmax(scores)
            #if scores[classId]>confThreshold:
            confidence = scores[classId]
            """
            if detection[4]>confThreshold:
                print('detection', detection[4], " - ", scores[classId], " - th : ", confThreshold)
                print('als', detection)
            """
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    
    # select bboxes accordingto nms supression
    output = []
    for i in indices:
        i = i[0] # get bbox id
        
        box = boxes[i] # get that bbox from the raw boxes list
         # Get the label for the class name and its confidence
        
        if classes:
            assert(classId < len(classes))
            label = '%s' % (classes[classId])

        # create a new result list [(left, top, width, height, confidence, ClassID, label), ...]
        output.append([box[0], box[1], box[2], box[3], confidences[i], classIds[i], label])
    
    return output, time


def jaro_distance(ying, yang, long_tolerance=False, winklerize=False):
    if not isinstance(ying, str) or not isinstance(yang, str):
        raise TypeError('expected str or unicode, got %s' % type(s).__name__)

    ying_len = len(ying)
    yang_len = len(yang)

    if not ying_len or not yang_len:
        return 0.0

    min_len = max(ying_len, yang_len)
    search_range = (min_len // 2) - 1
    if search_range < 0:
        search_range = 0

    ying_flags = [False]*ying_len
    yang_flags = [False]*yang_len

    # looking only within search range, count & flag matched pairs
    common_chars = 0
    for i, ying_ch in enumerate(ying):
        low = i - search_range if i > search_range else 0
        hi = i + search_range if i + search_range < yang_len else yang_len - 1
        for j in range(low, hi+1):
            if not yang_flags[j] and yang[j] == ying_ch:
                ying_flags[i] = yang_flags[j] = True
                common_chars += 1
                break

    # short circuit if no characters match
    if not common_chars:
        return 0.0

    # count transpositions
    k = trans_count = 0
    for i, ying_f in enumerate(ying_flags):
        if ying_f:
            for j in range(k, yang_len):
                if yang_flags[j]:
                    k = j + 1
                    break
            if ying[i] != yang[j]:
                trans_count += 1
    trans_count /= 2

    # adjust for similarities in nonmatched characters
    common_chars = float(common_chars)
    weight = ((common_chars/ying_len + common_chars/yang_len +
              (common_chars-trans_count) / common_chars)) / 3

    # winkler modification: continue to boost if strings are similar
    if winklerize and weight > 0.7 and ying_len > 3 and yang_len > 3:
        # adjust for up to first 4 chars in common
        j = min(min_len, 4)
        i = 0
        while i < j and ying[i] == yang[i] and ying[i]:
            i += 1
        if i:
            weight += i * 0.1 * (1.0 - weight)

        # optionally adjust for long strings
        # after agreeing beginning chars, at least two or more must agree and
        # agreed characters must be > half of remaining characters
        if (long_tolerance and min_len > 4 and common_chars > i+1 and
                2 * common_chars >= min_len + i):
            weight += ((1.0 - weight) * (float(common_chars-i-1) / float(ying_len+yang_len-i*2+2)))

    return weight


frame = cv2.imread('samples/6.jpg')
# 4 ve 6 tmm

print(frame.shape)
#run inference
# [(left, top, width, height, confidence, ClassID, label), ...]
results, time = yolo_inference(frame)

print(results, '\n', 'inf time:', time)

ROI = []
ocr_list = []
for result in results:
    
    left = result[0]
    top = result[1]
    width = int(result[2] + result[2]*0.02)
    height = int(result[3] + result[3]*0.0)
    
    # cut bboxes and put them in a list
    print(left, width , top, height)
    print(frame[top:top+height, left:left+width].shape)

    ROI.append(cv2.cvtColor(frame[top:top+height, left:left+width], cv2.COLOR_BGR2GRAY))
    cv2.imshow('detection result1', ROI[0])

    #draw boxes    
    cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 3)
    
    # put label text
    label = result[6]
    confidence = result[4]
    text = '{}: {:.0f}%'.format(label, confidence * 100)
    
    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (0, 0, 255), cv2.FILLED)
    frame = cv2.putText(frame, text, (left, top), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 0, 0), 2)

# apply OCR
for roi in ROI:
    ocr_list.append(pytesseract.image_to_string(roi, config='--psm 11'))

print('\n ----Found---- \n', ocr_list, jaro_distance(ocr_list[0], 'HR 26 BR 9044'))
cv2.imshow('detection result', frame)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
    
# if match threshold:
# Camera movements
# move(10,1)
