import cv2
import numpy as np
import glob
from skimage import io

class Stitch:
    def __init__(self, first_image, image_org):

        self.detector = cv2.SIFT_create(700)
        self.bf = cv2.BFMatcher()

        self.process_first_frame(first_image)
        self.output_img = image_org

        self.H_prev = np.eye(3)
        self.H_org = np.eye(3)

        # Initialize corners' coords
        corner_0 = np.array([0, 0])
        corner_1 = np.array([first_image.shape[1], 0])
        corner_2 = np.array([first_image.shape[1], first_image.shape[0]])
        corner_3 = np.array([0, first_image.shape[0]])

        self.corners = np.array([[corner_0, corner_1, corner_2, corner_3]], dtype=np.float32)
        print(self.corners)
        # ----------
        self.extreme = np.array([0, first_image.shape[0], 0, first_image.shape[1]])

    def process_first_frame(self, first_image):

        self.frame_prev = first_image
        frame_gray_prev = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
        self.kp_prev, self.des_prev = self.detector.detectAndCompute(frame_gray_prev, None)
        # self.des_prev /= (self.des_prev.sum(axis=1, keepdims=True) + 1e-7) 
        # self.des_prev = np.sqrt(self.des_prev)

    def pair_wise_match(self, des_cur, des_prev):

        pair_matches = self.bf.knnMatch(des_cur, des_prev, k=2)
        matches = []
        for m, n in pair_matches:
            if m.distance < 0.7*n.distance:
                matches.append(m)

        matches = sorted(matches, key=lambda x: x.distance)
        matches = matches[:min(len(matches),70)]

        return matches

    def findTransformation(self, image_1_kp, image_2_kp, matches):

        image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
        image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
        for i in range(0, len(matches)):
            image_1_points[i] = image_1_kp[matches[i].queryIdx].pt
            image_2_points[i] = image_2_kp[matches[i].trainIdx].pt

        H, mask = cv2.findHomography(image_1_points, image_2_points, cv2.RANSAC, ransacReprojThreshold=2.0)

        return H

    def transformed_coords(self, frame_cur, H):

        corner_0 = np.array([0, 0])
        corner_1 = np.array([frame_cur.shape[1], 0])
        corner_2 = np.array([frame_cur.shape[1], frame_cur.shape[0]])
        corner_3 = np.array([0, frame_cur.shape[0]])

        corners = np.array([[corner_0, corner_1, corner_2, corner_3]], dtype=np.float32)
        transformed_corners = cv2.perspectiveTransform(corners, H)

        transformed_corners = np.array(transformed_corners, dtype=np.int32)

        return transformed_corners

    def frame_process(self, frame_cur, frame_org):

        self.frame_cur = frame_cur
        self.frame_org = frame_org
        frame_gray_cur = cv2.cvtColor(frame_cur, cv2.COLOR_BGR2GRAY)
        self.kp_cur, self.des_cur = self.detector.detectAndCompute(frame_gray_cur, None)
        # self.des_cur /= (self.des_cur.sum(axis=1, keepdims=True) + 1e-7) 
        # self.des_cur = np.sqrt(self.des_cur)

        self.matches = self.pair_wise_match(self.des_cur, self.des_prev)

        if len(self.matches) < 4:
            return

        self.H = self.findTransformation(self.kp_cur, self.kp_prev, self.matches)
        self.H_new = np.matmul(self.H_org, self.H)
        self.H_dot = np.matmul(self.H_prev, self.H)
        transformed_corners = self.transformed_coords(self.frame_cur, self.H_new)
        self.corners = transformed_corners.copy()
        print(self.corners)

        #--------
        self.render(self.H_dot, self.frame_org)
        #--------

        # loop
        self.H_prev = self.H_dot
        self.H_org = self.H_new
        self.kp_prev = self.kp_cur
        self.des_prev = self.des_cur
        self.frame_prev = self.frame_cur
        print(self.frame_prev.shape)


    #------------------------------------------------------------------------------------------------------------------
    def draw_border(self, image, corners, color=(0, 0, 0)):
        # make image's border invisible
        for i in range(corners.shape[1]-1, -1, -1):
            cv2.line(image, tuple(corners[0, i, :]), tuple(
                corners[0, i-1, :]), thickness=5, color=color)
        return image

    #------------------------------------------------------------------------------------------------------------------
    # Those below functions are prepared for post processing

    def find_extreme(self, corner_cur):
        corners = corner_cur
        north = min(corners.T[1])
        south = max(corners.T[1])
        west = min(corners.T[0])
        east = max(corners.T[0])

        # ------
        if north < self.extreme[0]:
            self.extreme[0] = north
        if south > self.extreme[1]:
            self.extreme[1] = south
        if west < self.extreme[2]:
            self.extreme[2] = west
        if east > self.extreme[3]:
            self.extreme[3] = east

        return self.extreme

    def create_overlap_mask(self, warped_img):
        self.mask = np.zeros(warped_img.shape)
        self.mask[warped_img > 0] = 1
        self.mask[self.output_img <= 0] = 0
        return self.mask
    
    def distance_transform(self, warped_img):
        out_img = self.output_img.copy()
        img = warped_img.copy()
        out_img = cv2.cvtColor(out_img.astype('uint8'), cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2GRAY)
        
        out_img = cv2.distanceTransform(out_img, cv2.DIST_L2, 3)
        img = cv2.distanceTransform(img, cv2.DIST_L2, 3)
        cv2.normalize(out_img, out_img, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)
        
        # cv2.imshow('Distance Transform  out Image', out_img)
        # cv2.imshow('Distance Transform Image', img)
        
        mask = np.zeros(warped_img.shape)
        mask[warped_img > 0] = 1
        mask[self.output_img <= 0] = 0
        
        # Transform to 3 channels
        out_img3 = np.zeros(self.output_img.shape)
        out_img3[:,:,0] = out_img
        out_img3[:,:,1] = out_img
        out_img3[:,:,2] = out_img
        
        img3 = np.zeros(self.output_img.shape)
        img3[:,:,0] = img
        img3[:,:,1] = img
        img3[:,:,2] = img
        
        return out_img3, img3, mask

    def alpha_blending(self, warped_img, alpha = 0.3):
        # alpha = 0.3
        # self.create_overlap_mask(warped_img)
        # warped_img[self.mask > 0] = warped_img[self.mask > 0]*alpha
        # self.output_img[self.mask > 0] = self.output_img[self.mask > 0]*(1-alpha)
        # self.output_img += warped_img
        
        out_img, img, mask = self.distance_transform(warped_img)
        warped_img[mask > 0] = warped_img[mask > 0]*img[mask > 0] / (img[mask > 0] + out_img[mask > 0])
        self.output_img[mask > 0] = self.output_img[mask > 0]*out_img[mask > 0] / (img[mask > 0] + out_img[mask > 0])
        self.output_img += warped_img
        
        return self.output_img

    def render(self, H, frame_cur):
        # --------
        transformed_corners = self.transformed_coords(frame_cur, H)
        # -------------
        self.extreme = self.find_extreme(transformed_corners[0])

        # #-------------

        self.output_img_old = self.output_img

        self.output_img = np.zeros((min(4096, self.extreme[1] - self.extreme[0]),
                                    min(4096, self.extreme[3] - self.extreme[2]), 3))

        # ----------------------------------
        if self.extreme[0] < 0 or self.extreme[2] < 0:
            if self.output_img.shape[0] < 4096:
                h_offset = int(- self.extreme[0])
            else:
                h_offset = 0
            if self.output_img.shape[1] < 4096:
                w_offset = int(- self.extreme[2])
            else:
                w_offset = 0

            self.output_img[h_offset:h_offset + self.output_img_old.shape[0],
            w_offset:w_offset + self.output_img_old.shape[1], :] = self.output_img_old
            H_prev = np.eye(3)
            H_prev[0, 2] = w_offset
            H_prev[1, 2] = h_offset
            H = np.matmul(H_prev, H)
            self.H_dot = H
            transformed_corners = self.transformed_coords(frame_cur, H)
            if self.extreme[0] < 0:
                self.extreme[1] += h_offset
                self.extreme[0] += h_offset

            if self.extreme[2] < 0:
                self.extreme[3] += w_offset
                self.extreme[2] += w_offset

        else:
            self.output_img[0:self.output_img_old.shape[0], 0:self.output_img_old.shape[1], :] = self.output_img_old
        warped_img = cv2.warpPerspective(
            frame_cur, H, (self.output_img.shape[1], self.output_img.shape[0]), flags=cv2.INTER_LINEAR)

        warped_img = self.draw_border(warped_img, transformed_corners)
        #self.distance_transform(warped_img)
        self.output_img = self.alpha_blending(warped_img)
        # self.output_img[warped_img > 0] = warped_img[warped_img > 0]
        
        output_temp = np.copy(self.output_img)
        
        #output_temp = self.draw_border(output_temp, transformed_corners, color=(0, 0, 255))
        
        cv2.imshow('output',  (output_temp/255.))

        return self.output_img

    # def construct_final_image(self, data):
    #     self.output_img_final = np.zeros((min(2048, self.extreme[1] - self.extreme[0]),
    #                                       min(2048, self.extreme[3] - self.extreme[2]),3))

    #     w_offset = int(-self.extreme[2])
    #     h_offset = int(-self.extreme[0])
    #     print(w_offset, h_offset)

    #     self.output_img_final[h_offset:h_offset+data[0].shape[0],
    #                           w_offset:w_offset+data[0].shape[1], :] = data[0]

    #     i = 1
    #     first_matrix = True
    #     for matrix in self.matrix_list:
    #         if first_matrix:
    #             matrix[0, 2] = w_offset
    #             matrix[1, 2] = h_offset
    #             matrix_prev = matrix
    #             print(matrix)
    #             first_matrix = False
    #         else:
    #             matrix = np.matmul(matrix_prev, matrix)
    #             print('matrix: ',matrix)

    #             warped_img = cv2.warpPerspective(
    #             data[i], matrix, (self.output_img_final.shape[1], self.output_img_final.shape[0]), flags=cv2.INTER_LINEAR)
    #             #print('warped_img: ', warped_img.shape)

    #             #warped_img = self.draw_border(warped_img, transformed_corners)

    #             self.output_img_final[warped_img > 0] = warped_img[warped_img > 0]

    #             output_temp = np.copy(self.output_img_final)
    #             i +=1
    #             matrix_prev = matrix
    #             #output_temp = self.draw_border(output_temp, transformed_corners, color=(0, 0, 255))

    #             cv2.imshow('output_final',  output_temp/255.)

    #     return self.output_img_final

def main():

    video_path = 'Data/building.mp4'
    data = []
    cap = cv2.VideoCapture(video_path)
    is_first_frame = True
    i = 0
    cap.read()
    while cap.isOpened():
        ret, frame_cur = cap.read()
        if ret:
        #frame = frame_cur
            frame_cur = cv2.resize(frame_cur,(int(frame_cur.shape[1]/2),int(frame_cur.shape[0]/2)), interpolation = cv2.INTER_AREA)
        # frame = cv2.detailEnhance(frame_cur, sigma_r= 50, sigma_s= 0.45)
        # print(frame_cur.shape)
        # cv2.imshow('fr',  frame_cur)
        #data.append(frame_cur)
        if not ret:
            if is_first_frame:
                continue
            break

        if is_first_frame:
            video_mosaic = Stitch(frame_cur, frame_cur)
            is_first_frame = False
            continue
        
        # process each frame
        video_mosaic.frame_process(frame_cur, frame_cur)
        i += 1
        print(i)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #output = video_mosaic.blend(data)
    print(video_mosaic.output_img.shape)
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
    # cv2.imwrite('mosaicn.jpg', video_mosaic.output_img)
    # cv2.imwrite('fin.jpg', output)

if __name__ == "__main__":
    main()