import math
import cv2
import numpy as np

def get_blob(labels, idBlob):
    blob = np.zeros_like(labels, dtype=np.uint8)
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i][j] == idBlob:
                blob[i][j] = 255
    return blob

def get_colored_blob(image, labels, idBlob, col):
    colors = [(204, 50, 153), (255,144,30), (0, 69, 255)]
    image[labels == idBlob] = [colors[col][0], colors[col][1], colors[col][2]]
    return image

def get_number_of_holes(blob):
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(255 - blob, 4)
    return len(np.unique(labels)) - 2


def get_blob_area(blob):
    contour = get_blob_contour(blob)
    return cv2.contourArea(contour)

def get_MER(angle, blob, centroid):
    contour_points = get_blob_contour(blob)
    alpha = -math.sin(angle)
    beta = math.cos(angle)

    major_axis = (alpha, -beta, beta * centroid[1] - alpha * centroid[0])
    minor_axis = (beta, alpha, -beta * centroid[0] - alpha * centroid[1])

    dist_point1 = float('-inf')
    dist_point2 = float('inf')
    dist_point3 = float('-inf')
    dist_point4 = float('inf')

    for p in contour_points:

        j = p[0][0]
        i = p[0][1]

        dist_M = (major_axis[0] * j + major_axis[1] * i + major_axis[2]) / math.sqrt(major_axis[0] ** 2 + major_axis[1] ** 2)
        dist_m = (minor_axis[0] * j + minor_axis[1] * i + minor_axis[2]) / math.sqrt(minor_axis[0] ** 2 + minor_axis[1] ** 2)

        if dist_M > dist_point1:
            c1 = (j, i)
            dist_point1 = dist_M

        if dist_M < dist_point2:
            c2 = (j, i)
            dist_point2 = dist_M

        if dist_m > dist_point3:
            c3 = (j, i)
            dist_point3 = dist_m

        if dist_m < dist_point4:
            c4 = (j, i)
            dist_point4 = dist_m


    # Declare a, b, a', b'
    a = alpha
    b = -beta
    aa = beta
    bb = alpha

    # lines thorugh c1,c2,c3 and c4
    cl1 = -(a * c1[0] + b * c1[1])
    cl2 = -(a * c2[0] + b * c2[1])
    cw1 = -(aa * c3[0] + bb * c3[1])
    cw2 = -(aa * c4[0] + bb * c4[1])

    # vertexes of the oriented MER
    jv1 = (b * cw1 - bb * cl1) / (a * bb - b * aa)
    iv1 = (aa * cl1 - a * cw1) / (a * bb - b * aa)
    v1 = (int(jv1), int(iv1))

    jv2 = (b * cw2 - bb * cl1) / (a * bb - b * aa)
    iv2 = (aa * cl1 - a * cw2) / (a * bb - b * aa)
    v2 = (int(jv2), int(iv2))

    jv3 = (b * cw1 - bb * cl2) / (a * bb - b * aa)
    iv3 = (aa * cl2 - a * cw1) / (a * bb - b * aa)
    v3 = (int(jv3), int(iv3))

    jv4 = (b * cw2 - bb * cl2) / (a * bb - b * aa)
    iv4 = (aa * cl2 - a * cw2) / (a * bb - b * aa)
    v4 = (int(jv4), int(iv4))

    length = np.linalg.norm(np.array(v1) - np.array(v2))
    width = np.linalg.norm(np.array(v1) - np.array(v3))

    return v1, v2, v3, v4, length, width



def get_axis_coordinates(angle, blob, centroid):
    contour_points = get_blob_contour(blob)
    alpha = -math.sin(angle)
    beta = math.cos(angle)

    major_axis = (alpha, -beta, beta * centroid[1] - alpha * centroid[0])
    minor_axis = (beta, alpha, -beta * centroid[0] - alpha * centroid[1])

    dist_point1 = float('inf')
    dist_point2 = float('inf')
    dist_point3 = float('inf')
    dist_point4 = float('inf')

    for p in contour_points:

        j = p[0][0]
        i = p[0][1]

        dist_M = (major_axis[0] * j + major_axis[1] * i + major_axis[2]) / math.sqrt(major_axis[0] ** 2 + major_axis[1] ** 2)
        dist_m = (minor_axis[0] * j + minor_axis[1] * i + minor_axis[2]) / math.sqrt(minor_axis[0] ** 2 + minor_axis[1] ** 2)

        if dist_m > 0 and abs(dist_M) < dist_point1:
            p1 = (j, i)
            dist_point1 = abs(dist_M)

        if dist_m < 0 and abs(dist_M) < dist_point2:
            p2 = (j, i)
            dist_point2 = abs(dist_M)

        if dist_M > 0 and abs(dist_m) < dist_point3:
            p3 = (j, i)
            dist_point3 = abs(dist_m)

        if dist_M < 0 and abs(dist_m) < dist_point4:
            p4 = (j, i)
            dist_point4 = abs(dist_m)

    return p1,p2,p3,p4

def is_rod(blob, centroid):
    if get_number_of_holes(blob) == 0:
        return False

    if get_number_of_holes(blob) > 2:
        return False

    if get_number_of_holes(blob) == 1:
        countours = get_blob_contour(blob)
        #haralick circularity
        dist = [np.linalg.norm(point - centroid) for point in countours]
        std = np.std(dist)
        mu = np.mean(dist)
        if mu / std > 5:
            return False

    return True


def getHoleFeatures(blob):
    contours = get_inner_contour(blob)
    centers = []
    diameters =[]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        diameter = 2 * math.sqrt(area/math.pi)
        x, y, w, h = cv2.boundingRect(cnt)
        centers.append((x+w/2, y+h/2))
        diameters.append(diameter)

    return centers,diameters

def get_barycenter_width(angle, blob, centroid):
    contour_points = get_blob_contour(blob)
    alpha = -math.sin(angle)
    beta = math.cos(angle)

    major_axis = (alpha, -beta, beta * centroid[1] - alpha * centroid[0])
    minor_axis = (beta, alpha, -beta * centroid[0] - alpha * centroid[1])

    dist_point1 = float('inf')
    dist_point2 = float('inf')

    for p in contour_points:

        j = p[0][0]
        i = p[0][1]

        dist_M = (major_axis[0] * j + major_axis[1] * i + major_axis[2]) / math.sqrt(major_axis[0]** 2 + major_axis[1]**2)
        dist_m = (minor_axis[0] * j + minor_axis[1] * i + minor_axis[2]) / math.sqrt(minor_axis[0]**2 + minor_axis[1]**2)

        if dist_M > 0 and abs(dist_m) < dist_point1:
            point1 = (j, i)
            dist_point1 = abs(dist_m)

        if dist_M < 0 and abs(dist_m) < dist_point2:
            point2 = (j, i)
            dist_point2 = abs(dist_m)

    dist = np.linalg.norm(np.array(point1) - np.array(point2))

    return [point1, point2, dist]

def get_angle(moments):
    # first derivative
    theta = 0.5 * math.atan(2 * moments['mu11'] / (moments['mu02'] - moments['mu20']))

    # second derivative
    theta2 = -(math.cos(2 * theta) * (moments['mu02'] - moments['mu20']) + math.sin(2 * theta) * 2 * moments['mu11'])

    # analysis of the second derivative
    if theta2 > 0:
        # major axis
        return theta
    else:
        # minor axis
        return theta + math.pi / 2

def get_blob_contour(blob):
    contours, hierarchy = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # RETR_EXTERNAL very useful if you want to find the outer corners
    return contours[0]

def get_inner_contour(blob):
    contours, hierarchy = cv2.findContours(blob, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    inners = []
    for cnt in contours:
        area = area = cv2.contourArea(cnt)
        if area < 900:
            inners.append(cnt)

    return inners

def median_filtered_image(binarized):
    iterations = 5
    median = binarized.copy()
    kernel_size = 3

    for i in range(iterations):
        median = cv2.medianBlur(median.copy(), kernel_size)

    return median

def separate_blobs(blob):
    contours, hierarchy = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        #check for connected blobs
        if cv2.contourArea(c) > 6000:

            peri = cv2.arcLength(c, True)
            cnt = cv2.approxPolyDP(c,0.005*peri, True)
            #convex hull
            hull = cv2.convexHull(cnt, returnPoints=False)
            #defect points
            defects = cv2.convexityDefects(cnt, hull)
            points = []
            #array of distances between 2 point and x,y
            dist = []
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                far = tuple(cnt[f][0])
                points.append(far)

            for p in points:
                for p2 in points:
                    if p!=p2:
                        dist.append([np.linalg.norm(np.array(p)-np.array(p2)), p, p2])

            # keep only even-index elements because distances are duplicated
            d = dist[::2]

            #keep only distances of previous array d
            a = [el[0] for el in d]

            #if distance < 40, we keep track of the corresponding coordinates
            coord = [d[i] for i in range(len(a)) if a[i] < 40]

            for co in coord :
                #draw black line to split connected blobs
                cv2.line(blob, co[1], co[2], (0,0,0), 2)
                # draw white line to  restore the shape of the rod in case of rod breakage during the splitting
                cv2.line(blob, np.array(co[1]) - 2, np.array(co[2]) - 2, (255, 255, 255), 2)
                cv2.line(blob, np.array(co[1])+2, np.array(co[2])+2, (255, 255, 255), 2)

    return blob

def print_info(centroid, angle, type, length, width, barycenter_width, holes_centers, holes_diameters):

    print("Blob type: ", type)
    print("Centroid: ", centroid[0], centroid[1])
    print("Angle: ", angle)
    print("Length: ", length)
    print("Width: ", width)
    print("Width at barycenter:", barycenter_width)

    for i in range(len(holes_centers)):
        print("Hole ", i+1, ": { center: ", holes_centers[i], ", diameter: ", holes_diameters[i], "}")



def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def draw_blobs_features(img,vertexes, axis_coord, centroid):

    cv2.line(img, vertexes[0], vertexes[1], (0, 255, 0), 2)
    cv2.line(img, vertexes[0], vertexes[2], (0, 255, 0), 2)
    cv2.line(img, vertexes[2], vertexes[3], (0, 255, 0), 2)
    cv2.line(img, vertexes[1], vertexes[3], (0, 255, 0), 2)

    cv2.line(img, axis_coord[0], axis_coord[1], (255, 255, 255), 2)
    cv2.line(img, axis_coord[2], axis_coord[3], (255, 255, 255), 2)

    cv2.circle(img, (int(centroid[0]), int(centroid[1])), 4, (255, 0, 0), -1)

    return img
