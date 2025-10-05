import os
import sys

import cv2

sys.path.insert(0, "..")
import importlib
import time

import numpy as np
import torch

from .functions import get_meanface
from .networks import Pip_resnet18, Pip_resnet50, Pip_resnet101
from torchvision import models, transforms


# current file
current_file = __file__

# package root (two levels up if inside subpackage)
package_root = os.path.abspath(os.path.join(__file__, "..", ".."))

class PixelInPixelNet:
    def __init__(self, experiment_info):
        experiment_name = experiment_info.split("/")[-1][:-3]
        data_name = experiment_info.split("/")[-2]
        config_path = f".experiments.{data_name}.{experiment_name}"

        my_config = importlib.import_module(config_path, package="PIPNet")
        Config = my_config.Config
        self.cfg = Config()
        self.cfg.experiment_name = experiment_name
        self.cfg.data_name = data_name

        if self.cfg.backbone == "resnet18":
            resnet18 = models.resnet18(pretrained=self.cfg.pretrained)
            net = Pip_resnet18(
                resnet18,
                self.cfg.num_nb,
                num_lms=self.cfg.num_lms,
                input_size=self.cfg.input_size,
                net_stride=self.cfg.net_stride,
            )
        elif self.cfg.backbone == "resnet50":
            resnet50 = models.resnet50(pretrained=self.cfg.pretrained)
            net = Pip_resnet50(
                resnet50,
                self.cfg.num_nb,
                num_lms=self.cfg.num_lms,
                input_size=self.cfg.input_size,
                net_stride=self.cfg.net_stride,
            )
        elif self.cfg.backbone == "resnet101":
            resnet101 = models.resnet101(pretrained=self.cfg.pretrained)
            net = Pip_resnet101(
                resnet101,
                self.cfg.num_nb,
                num_lms=self.cfg.num_lms,
                input_size=self.cfg.input_size,
                net_stride=self.cfg.net_stride,
            )
        else:
            print("No such backbone!")
            exit(0)

        if self.cfg.use_gpu:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
        self.net = net.to(device)

        save_dir = os.path.join(
            os.path.join(package_root, "PIPNet/snapshots"),
            self.cfg.data_name,
            self.cfg.experiment_name,
        )
        weight_file = os.path.join(save_dir, "epoch%d.pth" % (self.cfg.num_epochs - 1))
        state_dict = torch.load(weight_file, map_location=device)
        self.net.load_state_dict(state_dict)

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((self.cfg.input_size, self.cfg.input_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )

        meanface_indices, self.reverse_index1, self.reverse_index2, self.max_len = (
            get_meanface(
                os.path.join("data", self.cfg.data_name, "meanface.txt"),
                self.cfg.num_nb,
            )
        )

    def inference(self, frame, detections, device="cpu"):
        input_size = self.cfg.input_size
        frame_height, frame_width, _ = frame.shape
        det_box_scale = 1.2

        faces_landmarks = []
        for i in range(len(detections)):
            t1 = time.time()
            det_xmin = detections[i][0]
            det_ymin = detections[i][1]
            det_width = detections[i][2]
            det_height = detections[i][3]
            det_xmax = det_xmin + det_width - 1
            det_ymax = det_ymin + det_height - 1

            det_xmin -= int(det_width * (det_box_scale - 1) / 2)
            # remove a part of top area for alignment, see paper for details
            det_ymin += int(det_height * (det_box_scale - 1) / 2)
            det_xmax += int(det_width * (det_box_scale - 1) / 2)
            det_ymax += int(det_height * (det_box_scale - 1) / 2)
            det_xmin = max(det_xmin, 0)
            det_ymin = max(det_ymin, 0)
            det_xmax = min(det_xmax, frame_width - 1)
            det_ymax = min(det_ymax, frame_height - 1)
            det_width = det_xmax - det_xmin + 1
            det_height = det_ymax - det_ymin + 1
            cv2.rectangle(
                frame, (det_xmin, det_ymin), (det_xmax, det_ymax), (0, 0, 255), 2
            )
            det_crop = frame[det_ymin:det_ymax, det_xmin:det_xmax, :]
            det_crop = cv2.resize(det_crop, (input_size, input_size))
            inputs = Image.fromarray(det_crop[:, :, ::-1].astype("uint8"), "RGB")
            inputs = self.preprocess(inputs).unsqueeze(0)
            inputs = inputs.to(device)
            (
                lms_pred_x,
                lms_pred_y,
                lms_pred_nb_x,
                lms_pred_nb_y,
                outputs_cls,
                max_cls,
            ) = forward_pip(
                self.net,
                inputs,
                self.preprocess,
                input_size,
                self.cfg.net_stride,
                self.cfg.num_nb,
            )
            lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
            tmp_nb_x = lms_pred_nb_x[self.reverse_index1, self.reverse_index2].view(
                self.cfg.num_lms, self.max_len
            )
            tmp_nb_y = lms_pred_nb_y[self.reverse_index1, self.reverse_index2].view(
                self.cfg.num_lms, self.max_len
            )
            tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(
                -1, 1
            )
            tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(
                -1, 1
            )
            lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()
            lms_pred = lms_pred.cpu().numpy()
            lms_pred_merge = lms_pred_merge.cpu().numpy()
            print("landmark detection time in sec: ", time.time() - t1)
            face_landmarks = []
            for i in range(self.cfg.num_lms):
                x_pred = lms_pred_merge[i * 2] * det_width
                y_pred = lms_pred_merge[i * 2 + 1] * det_height
                cv2.circle(
                    frame,
                    (int(x_pred) + det_xmin, int(y_pred) + det_ymin),
                    1,
                    (0, 0, 255),
                    2,
                )
                face_landmarks.append((int(x_pred) + det_xmin, int(y_pred) + det_ymin))

            faces_landmarks.append(np.array(face_landmarks))

        return faces_landmarks
