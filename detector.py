import matplotlib.pyplot as plt
import cv2
import numpy as np
from numpy.linalg import det


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def line_engineer(point1, point2):
    """
    This function takes two points as input and returns a tuple representation of a line in standard form (Ax+By = C).
    :param point1: Point number one as a tuple.
    :param point2: Point number two as a tuple.
    :returns: Coefficients of a line in standard form as a list (A, B, C)
    """
    a = -1 * (point2[1] - point1[1])
    b = (point2[0] - point1[0])
    c = b * point1[1] + a * point1[0]
    return [a, b, c]


def intersection_calculator(line1, line2):
    """
    This function calculates the intersection of two lines.
    :param line1: Coefficients of line as a tuple or list in standard form.
    :param line2: Coefficients of line as a tuple or list in standard form.
    :return: Intersection (point) of the two lines as a tuple.
    """
    matrix = []
    for i in range(3):
        matrix.append([line1[i], line2[i]])
    y = det([matrix[0], matrix[2]]) / det([matrix[0], matrix[1]])
    x = det([matrix[1], matrix[2]]) / (-1 * det([matrix[0], matrix[1]]))
    return int(x), int(y)


def base_point_finder(line1, line2, y=720):
    """
    This function calculates the base point of the suggested path by averaging the x coordinates of both detected
    lines at the highest y value.
    :param line1: Coefficients of equation of first line in standard form as a tuple.
    :param line2: Coefficients of equation of second line in standard form as a tuple.
    :param y: Height value from which base point should be calculated.
    :return: Tuple representation of the point from which the suggested path will originate.
    """
    x1 = (line1[2] - line1[1] * y) / (line1[0])
    x2 = (line2[2] - line2[1] * y) / (line2[0])
    x = (x1 + x2) / 2
    return int(x), int(y)


def draw_lines(original_image, houghp_transform_lines):
    blank_image = np.zeros_like(original_image)

    # get all values of theta for all detected lines
    theta_values = []
    for line in houghp_transform_lines:
        (x0, y0, x1, y1) = line[0]
        points = (x0, y0, x1, y1)
        theta = np.arctan((y1 - y0) / (x1 - x0))
        theta_values.append((theta, points))

    # remove all groups of parallel line, while leaving one
    theta_values.sort(key=lambda x: x[0])
    compare = None
    intersecting_lines = []
    for (i, (theta, points)) in enumerate(theta_values):
        if i == 0:
            compare = theta
            intersecting_lines.append(points)
            continue
        else:
            delta = compare - theta
            compare = theta
            # radians
            if -(np.pi / 12) <= delta <= (np.pi / 12):
                continue
            else:
                intersecting_lines.append(points)

    # iterate through intersecting_lines and plot all points
    if len(intersecting_lines) == 2:
        lines = []
        for (x0, y0, x1, y1) in intersecting_lines:
            cv2.line(blank_image, (x0, y0), (x1, y1), (0, 255, 255), thickness=5)
            lines.append(line_engineer((x0, y0), (x1, y1)))
        point = intersection_calculator(lines[0], lines[1])
        base = base_point_finder(lines[0], lines[1])
        cv2.arrowedLine(blank_image, base, point, (0, 0, 255), 5)
        cv2.putText(blank_image, "Suggested Path: Active", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        for (x0, y0, x1, y1) in intersecting_lines:
            cv2.line(blank_image, (x0, y0), (x1, y1), (0, 255, 255), thickness=5)
        cv2.putText(blank_image, "Suggested Path: Inactive", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    detected_image = cv2.addWeighted(original_image, 0.6, blank_image, 1, 0)

    if len(intersecting_lines) != 2:
        print("help", len(intersecting_lines))

    return detected_image


def lane_detection(image):
    img = image.copy()
    height = img.shape[0]
    width = img.shape[1]

    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_image, 50, 100)

    vertices = [(0, height), (width / 2, height / 2), (width, height)]
    cropped_image = region_of_interest(edges, np.array([vertices], np.int32))
    lanes_detected_image = draw_lines(img, cv2.HoughLinesP(cropped_image, 1, np.pi / 180, 27, minLineLength=20,
                                                           maxLineGap=25))

    return lanes_detected_image



