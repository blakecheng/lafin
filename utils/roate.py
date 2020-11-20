import cv2
import yaml

import os
import sys

import copy
import numpy as np
from math import cos, sin, atan2, asin, sqrt




def _parse_param(param, pose_noise=False, frontal=True, 
                     large_pose=False, yaw_pose=None, pitch_pose=None, roll_pose=None):
        """Work for both numpy and tensor"""
        p_ = param[:12].reshape(3, -1)
        p = p_[:, :3]
        s, R, t3d = P2sRt(p_)
        angle = matrix2angle(R)
        original_angle = angle[0]
        if yaw_pose is not None or pitch_pose is not None or roll_pose is not None:
            # angle[0] = yaw_pose if yaw_pose is not None
            # if yaw_pose is not None:
                # angle[0] = yaw_pose
                # flag = -1 if angle[0] < 0 else 1
            if yaw_pose is not None:
                angle[0] = yaw_pose
            
            if pitch_pose is not None:
                angle[1] = pitch_pose
                
            if roll_pose is not None:
                angle[2] = roll_pose

            p = angle2matrix(angle) * s
        else:
            if frontal:
                angle[0] = 0
                angle[1] = 0
                p = angle2matrix(angle) * s
            if pose_noise:
                if frontal:
                    if np.random.randint(0, 5):
                        p = random_p(s, angle)
                else:
                    p = random_p(s, angle)
            elif large_pose:
                if frontal:
                    if np.random.randint(0, 5):
                        p = assign_large(s, angle)
                else:
                    p = assign_large(s, angle)


        offset = p_[:, -1].reshape(3, 1)
        alpha_shp = param[12:52].reshape(-1, 1)
        alpha_exp = param[52:-4].reshape(-1, 1)
        box = param[-4:]
        return p, offset, alpha_shp, alpha_exp, box, original_angle
    
    
def affine_align(landmark=None, **kwargs):
        M = None
        src = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]], dtype=np.float32)
        src = src * 290 / 112
        src[:, 0] += 50
        src[:, 1] += 60
        src = src / 400 * render_size
        dst = landmark.astype(np.float32)
        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M2 = tform.params[0:2, :]
        with torch.cuda.device(current_gpu):
            M2 = torch.from_numpy(M2).float().cuda()
        return M2

def random_p(s, angle):
        if np.random.randint(0, 2) == 0:
            angle[0] += np.random.uniform(-0.965, -0.342, 1)[0]
            # angle[1] += np.random.uniform(-0.1, 0.1, 1)[0]
        else:
            angle[0] += np.random.uniform(0.342, 0.965, 1)[0]
            # angle[1] += np.random.uniform(-0.1, 0.1, 1)[0]
        angle[0] = max(-1.2, min(angle[0], 1.2))
        random_2 = np.random.uniform(-0.5, 0.5, 1)[0]
        angle[1] += random_2
        angle[1] = max(-1.0, min(angle[1], 1.0))
        p = angle2matrix(angle) * s
        return p
    
def P2sRt(P):
    ''' decompositing camera matrix P.
    Args:
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t2d: (2,). 2d translation.
    '''
    t3d = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t3d


def matrix2angle(R):
    ''' compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
    Args:
        R: (3,3). rotation matrix
    Returns:
        x: yaw
        y: pitch
        z: roll
    '''
    # assert(isRotationMatrix(R))

    if R[2, 0] != 1 and R[2, 0] != -1:
        x = -asin(max(-1, min(R[2, 0], 1)))
        y = atan2(R[2, 1] / cos(x), R[2, 2] / cos(x))
        z = atan2(R[1, 0] / cos(x), R[0, 0] / cos(x))

    else:  # Gimbal lock
        z = 0  # can be anything
        if R[2, 0] == -1:
            x = np.pi / 2
            y = z + atan2(R[0, 1], R[0, 2])
        else:
            x = -np.pi / 2
            y = -z + atan2(-R[0, 1], -R[0, 2])

    return [x, y, z]


def angle2matrix(angles):
    ''' get rotation matrix from three rotation angles(radian). The same as in 3DDFA.
    Args:
        angles: [3,]. x, y, z angles
        x: yaw.
        y: pitch.
        z: roll.
    Returns:
        R: 3x3. rotation matrix.
    '''
    # x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
    # x, y, z = angles[0], angles[1], angles[2]
    y, x, z = angles[0], angles[1], angles[2]

    # x
    Rx=np.array([[1,      0,       0],
                 [0, cos(x),  -sin(x)],
                 [0, sin(x),  cos(x)]])
    # y
    Ry=np.array([[ cos(y), 0, sin(y)],
                 [      0, 1,      0],
                 [-sin(y), 0, cos(y)]])
    # z
    Rz=np.array([[cos(z), -sin(z), 0],
                 [sin(z),  cos(z), 0],
                 [     0,       0, 1]])
    R = Rz.dot(Ry).dot(Rx)
    return R.astype(np.float32)



class face_3d:
    def __init__(self):
        import os
        cwd = os.getcwd()
        code_path = 'utils/F3DDFA_V2'
        sys.path.insert(0, code_path)  
        os.chdir(code_path)

        from FaceBoxes import FaceBoxes
        from TDDFA import TDDFA

        
        
        # load config
        cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)

        # Init FaceBoxes and TDDFA, recommend using onnx flag
        onnx_flag = False # or  True to use ONNX to speed up
        if onnx_flag:
            import os
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
            os.environ['OMP_NUM_THREADS'] = '4'
            
            from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
            from TDDFA_ONNX import TDDFA_ONNX
            face_boxes = FaceBoxes_ONNX()
            tddfa = TDDFA_ONNX(**cfg)
        else:
            tddfa = TDDFA(gpu_mode=False, **cfg)
            face_boxes = FaceBoxes()
            
        self.tddfa = tddfa
        self.face_boxes = face_boxes
        self.dense_flag = False
        os.chdir(cwd)
            
    def roate(self,img,yaw_pose=None, pitch_pose=None, roll_pose=None,frontal=True):
        if yaw_pose is not None:
            yaw_pose = yaw_pose*math.pi/180.0
        if pitch_pose is not None:
            pitch_pose = pitch_pose *math.pi/180.0
        if roll_pose is not None:
            roll_pose = roll_pose* math.pi/180.0
            
        boxes = self.face_boxes(img)
        param_lst, roi_box_lst = self.tddfa(img, boxes)
        new_param_lst = []
        for i in range(len(param_lst)):
            param = copy.deepcopy(param_lst[i])
            p_ = param[:12].reshape(3, -1)
            p, offset, alpha_shp, alpha_exp, box, original_angle = _parse_param(param_lst[i],yaw_pose=yaw_pose, pitch_pose=pitch_pose,roll_pose=roll_pose,frontal=frontal)
            p_[:, :3]= p
            new_param_lst.append(param)
        ver_lst = self.tddfa.recon_vers(new_param_lst, roi_box_lst, dense_flag=self.dense_flag)   
        return ver_lst

import math
import os
    
face=face_3d()
root= '/opt/mnt/cb/dataset/FFHQ/ffhq-lafin/images'
img_list = os.listdir(root)
img_fp = os.path.join(root,img_list[100])
img = cv2.imread(img_fp)
rotate_landmark_lst = face.roate(img,None,None,None,frontal=False)
