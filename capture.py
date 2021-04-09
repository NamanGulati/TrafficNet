import cv2
import numpy as np
import csv


model_idx_to_class = {0: 0,
 1: 1,
 2: 10,
 3: 11,
 4: 12,
 5: 13,
 6: 14,
 7: 15,
 8: 16,
 9: 17,
 10: 18,
 11: 19,
 12: 2,
 13: 20,
 14: 21,
 15: 22,
 16: 23,
 17: 24,
 18: 25,
 19: 26,
 20: 27,
 21: 28,
 22: 29,
 23: 3,
 24: 30,
 25: 31,
 26: 32,
 27: 33,
 28: 34,
 29: 35,
 30: 36,
 31: 37,
 32: 38,
 33: 39,
 34: 4,
 35: 40,
 36: 41,
 37: 42,
 38: 5,
 39: 6,
 40: 7,
 41: 8,
 42: 9}


labels_csv = csv.DictReader(open("signnames.csv"))
labels = {}
for row in labels_csv:
  labels[int(row['ClassId'])] = row['SignName']

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

net = cv2.dnn.readNetFromONNX("TrafficSignNetExport")


cap = cv2.VideoCapture(1)
cap.set(3,480)
cap.set(4,480)
cap.set(10,30)

while True:
    success,imgOriginal = cap.read()
    
    img = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
    img = clahe.apply(img)
    # img = imgOriginal
    img_clahe = img
    img = cv2.resize(img,(28,28))
    blob = cv2.dnn.blobFromImage(img,crop=False)
    blob = np.array(blob)
    blob = blob/255
    
    net.setInput(blob)
    preds = net.forward()
    biggest_pred_index = np.array(preds)[0].argmax()

    cv2.putText(img_clahe,f"prediction: {labels[model_idx_to_class[biggest_pred_index]]}",(20,35),fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=1,color=(0,0,255),thickness=3)
    cv2.imshow("input",img_clahe)
    # print("prediction:",labels[biggest_pred_index])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
