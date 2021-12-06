import util.util as util
import cv2
import torch
import numpy as np
import face_alignment as F

def mask_tensor(self, image):
        image_numpy = util.tensor2im(image)
        util.save_image(image_numpy, '/home6/liuhy/contrastive-unpaired-translation/test/npmask.png' ) 
        image_cv2 = cv2.cvtColor(np.array(image_numpy), cv2.COLOR_BGR2RGB)
        image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)
        # print(type(image_cv2), image_cv2.shape)

        device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        align = F.FaceAlignment(landmarks_type=F.LandmarksType._3D, device=device)
        try:
            lms68 = align.get_landmarks(image_cv2)[0]
        except UserWarning:
            return []
        
        feature_id = [17, 22, 27, 36, 42, 48, 60, 68]
        feature_name = ['eyebrow1', 'eyebrow2', 'nose', 'eye1', 'eye2', 'lips', 'teeth']
        image_tensor = torch.ones_like(image)
        
        for i in range(len(feature_name)):
            xmin, ymin, xmax, ymax = self.return_BBox(feature_id[i], feature_id[i+1], lms68)
            xmin, ymin, xmax, ymax = int(xmin), int(xmax), int(ymin), int(ymax)
            print(feature_name[i],': ',xmin, ymin, xmax, ymax)
            for x in range(xmin, xmax + 1):
                for y in range(ymin, ymax + 1):
                    for j in range(3):
                        image_tensor[0][j][y][x] = 0
        
        image_ret = image_tensor * image
        image_numpy = util.tensor2im(image_ret)
        util.save_image(image_numpy, '/home6/liuhy/contrastive-unpaired-translation/test/testmask.png') 
        print("Current image saved...")

        exit(0)
        return image_ret