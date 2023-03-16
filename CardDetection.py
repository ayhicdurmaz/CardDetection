import cv2
import numpy as np


class Card:
    def __init__(self, _img, _points, _value, _symbol):
        self.points = _points
        self.card_image = _img
        self.value = _value
        self.symbol = _symbol


def autoTreshCalculate(img, sigma=0.33):
    median = np.median(img)
    lower = int(max(0, (1 - sigma) * median))
    upper = int(max(255, (1 + sigma) * median))
    return lower, upper


def findCardCorners(points):
    sample_index = np.argmax(points[:, 1])
    _corner_points = np.zeros((4, 2), dtype="float32")
    arr = np.zeros((3, 2))

    j = 0
    for i in range(0, len(points)):
        if i != sample_index:
            arr[j] = (np.sqrt(((points[sample_index, 0] - points[i, 0])
                      ** 2) + ((points[sample_index, 1] - points[i, 1]) ** 2)), i)
            j += 1

    near_index = int(arr[np.argmin(arr[:, 0]), 1])
    far_index = int(arr[np.argmax(arr[:, 0]), 1])
    mid_index = 6 - (sample_index + near_index + far_index)

    if points[near_index, 0] > points[sample_index, 0]:
        _corner_points[0] = points[near_index]
        _corner_points[1] = points[sample_index]
        _corner_points[2] = points[far_index]
        _corner_points[3] = points[mid_index]
    else:
        _corner_points[0] = points[sample_index]
        _corner_points[1] = points[near_index]
        _corner_points[2] = points[mid_index]
        _corner_points[3] = points[far_index]

    return _corner_points


def card2Canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lower_tresh, upper_tresh = autoTreshCalculate(gray, sigma=-0.3)
    _, tresh = cv2.threshold(gray, lower_tresh, upper_tresh, cv2.THRESH_BINARY)
    blur = cv2.medianBlur(tresh, 5)
    erode = cv2.erode(blur, (3,3), iterations=1)
    canny = cv2.Canny(erode, 127, 255)

    return canny


def getCards(img):
    conts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cont in conts:
        cont_area = cv2.contourArea(cont)
        approx = cv2.approxPolyDP(
            cont, 0.015 * cv2.arcLength(cont, True), True)
        if cont_area > 1000:
            if len(approx) == 4:
                # Unknown corners
                points = approx.reshape(4, 2)

                # Finded Corners
                corner_points = findCardCorners(points)

                # Set cards dimensions
                max_height = 300
                max_width = 200

                converted_points = np.float32([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]])

                matrix = cv2.getPerspectiveTransform(corner_points, converted_points)
                image_out = cv2.warpPerspective(frame, matrix, (max_width, max_height))
                
                cards.append(Card(image_out, points, 0, ''))


def cardParser(img, i):# will delete i
    card_corners = [frame_card[0:75, 0:40], cv2.flip(frame_card[225:300, 0:40], -1), frame_card[0:75, 160:200], cv2.flip(frame_card[225:300, 160:200], -1)]
    for corner in card_corners:
        corner = cv2.resize(corner, (80, 150))
        corner_gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
        _, tresh = cv2.threshold(corner_gray, 127, 255, cv2.THRESH_BINARY_INV)
        erode = cv2.erode(tresh, (3,3), iterations=1)
        conts, _ = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        points = [[],[],[],[]]
        for cont in conts:
            x,y,w,h = cv2.boundingRect(cont)
            if x == 0 or x+w == 80:
                continue
            elif y == 0 or y == 150:
                continue
            else:
                points[0].append(x)
                points[1].append(y)
                points[2].append(x+w)
                points[3].append(y+h)
        

        symbol = cv2.resize(corner_gray[max(points[3])-30:max(points[3]), min(points[0]): max(points[2])], (28,28))
        _, tresh_sym = cv2.threshold(symbol, 127,255,cv2.THRESH_BINARY_INV)
        val = cv2.resize(corner_gray[min(points[1]):min(points[1])+50, min(points[0]): max(points[2])], (28,28))
        _, tresh_val = cv2.threshold(val, 155,255,cv2.THRESH_BINARY)

        cv2.imwrite("val" + str(i) + ".png", tresh_val)
        print(i)
        i+=1
    return i


#frame = cv2.imread("a.png")
#frame = cv2.imread("WP.jpeg")
frame = cv2.imread("missing2.jpeg")

cards = []
frame_canny = card2Canny(frame)
getCards(frame_canny)

print(len(cards))

i=0
for card in cards:
    frame_card = card.card_image

    i = cardParser(frame_card, i)

    i+=1

cv2.imshow("Frame", frame_canny)
cv2.waitKey(0)
