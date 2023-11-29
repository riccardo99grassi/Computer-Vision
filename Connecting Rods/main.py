import math
import numpy as np
import utils
import glob
import cv2

file_names = glob.glob("images/*.BMP")
file_names.sort()
images = [cv2.imread(file) for file in file_names]


count = 0
show = []
for img in images:
    print(file_names[count])
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    filtered = cv2.GaussianBlur(imgGray, (5, 5), 1) if count < 12 else utils.median_filtered_image(imgGray)

    # Binarize the image to separate foreground and background
    threshold, binarized = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binarized = utils.separate_blobs(binarized)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(binarized, 4)
    #detected blobs
    n_blob = np.unique(labels).size
    final_img = cv2.cvtColor(binarized.copy(), cv2.COLOR_GRAY2BGR)
    count_color = 0
    for i in range(1, n_blob):
        id_blob = i
        blob = utils.get_blob(labels, id_blob)
        centroid = [centroids[id_blob][0], centroids[id_blob][1]]

        if (utils.is_rod(blob, centroid)):
            type = "A" if utils.get_number_of_holes(blob) == 1 else "B"
            moments = cv2.moments(blob, True)
            angle = math.degrees(utils.get_angle(moments))

            v1, v2, v3, v4, length, width = utils.get_MER(math.radians(angle),blob, centroid)
            barycenter_width = utils.get_barycenter_width(math.radians(angle), blob, centroid)[2]

            holes_centers, holes_diameters = utils.getHoleFeatures(blob)

            final_img = utils.get_colored_blob(final_img, labels, id_blob, count_color)
            p1, p2, p3, p4 = utils.get_axis_coordinates(math.radians(angle),blob, centroid)
            final_img = utils.draw_blobs_features(final_img, [v1,v2,v3,v4], [p1,p2,p3,p4], centroid)

            utils.print_info(centroid, angle, type, length, width, barycenter_width, holes_centers, holes_diameters)
            count_color +=1
            print()

    show.append(final_img)
    count +=1

imgStack = utils.stackImages(1,([show[0],show[1],show[2],show[3],show[4]],
                                [show[5],show[6],show[7],show[8],show[9]],
                                [show[10],show[11],show[12],show[13],show[14]],
                             ))
cv2.imshow("Stack", imgStack)
cv2.waitKey(0)
