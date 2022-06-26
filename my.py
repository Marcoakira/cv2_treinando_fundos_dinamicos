import joblib
import numpy as np
import cv2
import imutils
from sklearn.neighbors import KNeighborsClassifier
from time import sleep




def get_flow(prevs, next):
    flow = cv2.calcOpticalFlowFarneback(prevs, next, None, .1, 3, 5, 1, 5, 2, 1)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def get_strip(historico):
    strip = np.array(historico)
    # try:
    #     strip.reshape(bgr.shape[0]*batch, bgr.shape[1],3)
    # except:
    #     pass

    strip = strip.reshape(bgr.shape[0]*batch, bgr.shape[1], 3)
    strip = imutils.resize(strip, height=frame.shape[0])
    return strip

#labels

# labels = ['-', 'cima', 'baixo', 'direita', 'esquerda']
labels = ['-', 'perto', 'longe', 'bonita mao', 'bonita mao']

historico =[]
batch = 20
escala = 3


inferencia = False

pred_list = []
mem = 2

x = []
y = []

count_0 = 0
count_1 = 0
count_2 = 0
count_3 = 0
count_4 = 0

model = KNeighborsClassifier(n_neighbors=5)

cap = cv2.VideoCapture(1)


ret, frame = cap.read()
frame = cv2.flip(frame,1)
frame = cv2.resize(frame, None, fx=1 / escala, fy=1 / escala)
#
prevs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
print(prevs.shape)
#
hsv = np.zeros_like(frame)
hsv[..., 1] = 255

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_copy = cv2.resize(frame, None, fx=1 / escala, fy=1 / escala)
    next = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)

    # print(next.shape)

    bgr = get_flow(prevs, next)
    historico.append(bgr)
    historico = historico[-batch:]

    # print(prevs.shape)

    if len(historico) == batch:
        strip = get_strip(historico)
        if inferencia:
            pred = model.predict(strip.reshape(1,-1))
            pred_list.append(pred)
            if pred_list[-mem:] == [pred] * mem:
                print(labels[int(pred)])
                cv2.putText(frame, labels[int(pred)],(30,60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,255), 4, cv2.LINE_AA)


    bgr = cv2.resize(bgr, None, fx=escala, fy=escala)
    final = cv2.hconcat([frame,bgr])

    if len(historico) == batch:
        final = cv2.hconcat([final, strip])

    cv2.imshow("Final", final)
    prevs = next

    k = cv2.waitKey(1)

    if k == ord('0'):
        count_0 += 1
        print(f"Item Adcionado a classe{labels[0]} - total {count_0} .")
        x.append(strip)
        y.append(0)

    if k == ord('1'):
        count_1 += 1
        print(f"Item Adcionado a classe{labels[1]} - total {count_1} .")
        x.append(strip)
        y.append(1)

    if k == ord('2'):
        count_2 += 1
        print(f"Item Adcionado a classe{labels[2]} - total {count_2} .")
        x.append(strip)
        y.append(2)

    if k == ord('3'):
        count_3 += 1
        print(f"Item Adcionado a classe{labels[3]} - total {count_3} .")
        x.append(strip)
        y.append(3)

    if k == ord('4'):
        count_4 += 1
        print(f"Item Adcionado a classe{labels[4]} - total {count_4} .")
        x.append(strip)
        y.append(4)

    if k == ord('t'):
        print("treinando o modelo KNN")
        X = np.array(x)
        X = X.reshape(len(y), -1)
        model.fit(X,y)
        print("modelo Treinado com sucesso!")
        joblib.dump(model,"modelo_treino10.pkl")

    if k == ord("r"):
        print("carregando model salvo:")
        model = joblib.load("modelo_treino10.pkl")
        inferencia = True


    if k == ord('q'):
        cap.release()
        exit()
