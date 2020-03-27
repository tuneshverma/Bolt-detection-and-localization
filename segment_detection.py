import cv2
import math
import statistics


def radiuswithcenter(raw_image, dict, sorted_dict_keys, pos, area, draw=True):
    x_list = []
    y_list = []

    for x in range(len(dict[sorted_dict_keys[pos]])):
        # print(x)
        x_list.append((dict[sorted_dict_keys[pos]])[x][0][0])
        y_list.append((dict[sorted_dict_keys[pos]])[x][0][1])

    c_x = (statistics.mean(x_list))
    c_y = (statistics.mean(y_list))

    cord_x_org = x_list - c_x
    cord_y_org = y_list - c_y
    square_x = cord_x_org**2
    square_y = cord_y_org**2
    dist_square = square_x + square_y
    # print(square_x)
    max_distance_square = max(dist_square)
    # print(max_distance_square)
    radius = math.sqrt(max_distance_square)

    if 3.14*(radius**2) < area:
        if draw:
            cv2.circle(raw_image, (c_x, c_y), int(radius), (0, 255, 0), 2)
        return c_x, c_y, radius
    else:
        return False, False, False


def segment_detection(image):
    # image = cv2.imread('main_data/good_image/good_image2405_10_005.png')
    # image_resize = cv2.resize(image, (640, 640))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    edge = cv2.Canny(blur, 50, 200)
    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilation = cv2.dilate(edge, kernal, iterations=1)
    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernal)

    # cv2.imshow('opening', opening)
    # cv2.waitKey(0)

    _, contours, h = cv2.findContours(opening, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    dict = {}
    c = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        if (len(approx) > 8) & (len(approx) < 23) & (area < 1900) & (area > 30):
            dict[area] = contour
            c.append(contour)

    sorted_dict_keys = sorted(dict.keys())
    # print(sorted_dict_keys)
    return sorted_dict_keys, dict


# cv2.drawContours(image_resize, c, -1, (255, 0, 0), 2)
# cv2.imshow('contours', image_resize)
# cv2.waitKey(0)
