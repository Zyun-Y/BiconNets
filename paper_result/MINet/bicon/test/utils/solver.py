import os
from pprint import pprint

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F
import network as network_lib
from loss.CEL import CEL
from utils.dataloader import create_loader
from utils.metric import cal_maxf, cal_pr_mae_meanf
import datetime
from utils.misc import (
    AvgMeter,
    construct_print,
    write_data_to_file,
)
from utils_bicon import *
from utils.pipeline_ops import (
    get_total_loss,
    make_optimizer,
    make_scheduler,
    resume_checkpoint,
    save_checkpoint,
)
from utils.recorder import TBRecorder, Timer, XLSXRecoder


class Solver:
    def __init__(self, exp_name: str, arg_dict: dict, path_dict: dict):
        super(Solver, self).__init__()
        self.exp_name = exp_name
        self.arg_dict = arg_dict
        self.path_dict = path_dict

        self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to_pil = transforms.ToPILImage()

        self.tr_data_path = self.arg_dict["rgb_data"]["tr_data_path"]
        self.te_data_list = self.arg_dict["rgb_data"]["te_data_list"]

        self.save_path = self.path_dict["save"]
        self.save_pre = self.arg_dict["save_pre"]

        if self.arg_dict["tb_update"] > 0:
            self.tb_recorder = TBRecorder(tb_path=self.path_dict["tb"])
        if self.arg_dict["xlsx_name"]:
            self.xlsx_recorder = XLSXRecoder(xlsx_path=self.path_dict["xlsx"])

        # 依赖与前面属性的属性
        self.tr_loader = create_loader(
            data_path=self.tr_data_path,
            training=True,
            size_list=self.arg_dict["size_list"],
            prefix=self.arg_dict["prefix"],
            get_length=False,
        )
        self.end_epoch = self.arg_dict["epoch_num"]
        self.iter_num = self.end_epoch * len(self.tr_loader)

        if hasattr(network_lib, self.arg_dict["model"]):
            self.net = getattr(network_lib, self.arg_dict["model"])().to(self.dev)
        else:
            raise AttributeError
        pprint(self.arg_dict)

        # if self.arg_dict["resume_mode"] == "test":
        #     # resume model only to test model.
        #     # self.start_epoch is useless
        #     resume_checkpoint(
        #         model=self.net, load_path=self.path_dict["final_state_net"], mode="onlynet",
        #     )
        #     return

        self.loss_funcs = [
            torch.nn.BCEWithLogitsLoss(reduction=self.arg_dict["reduction"]).to(self.dev)
        ]
        if self.arg_dict["use_aux_loss"]:
            self.loss_funcs.append(CEL().to(self.dev))

        self.opti = make_optimizer(
            model=self.net,
            optimizer_type=self.arg_dict["optim"],
            optimizer_info=dict(
                lr=self.arg_dict["lr"],
                momentum=self.arg_dict["momentum"],
                weight_decay=self.arg_dict["weight_decay"],
                nesterov=self.arg_dict["nesterov"],
            ),
        )
        self.sche = make_scheduler(
            optimizer=self.opti,
            total_num=self.iter_num if self.arg_dict["sche_usebatch"] else self.end_epoch,
            scheduler_type=self.arg_dict["lr_type"],
            scheduler_info=dict(
                lr_decay=self.arg_dict["lr_decay"], warmup_epoch=self.arg_dict["warmup_epoch"]
            ),
        )

        # AMP
        if self.arg_dict["use_amp"]:
            construct_print("Now, we will use the amp to accelerate training!")
            from apex import amp

            self.amp = amp
            self.net, self.opti = self.amp.initialize(self.net, self.opti, opt_level="O1")
        else:
            self.amp = None

        if self.arg_dict["resume_mode"] == "train":
            # resume model to train the model
            self.start_epoch = resume_checkpoint(
                model=self.net,
                optimizer=self.opti,
                scheduler=self.sche,
                amp=self.amp,
                exp_name=self.exp_name,
                load_path=self.path_dict["final_full_net"],
                mode="all",
            )
        else:
            # only train a new model.
            self.start_epoch = 0
        return

    

    def test(self):
        device = torch.device('cpu')
        self.net.load_state_dict(torch.load('/MINet_model.pth',map_location=device))
        self.net.to(self.dev)
        self.net.eval()

        total_results = {}
        for data_name, data_path in self.te_data_list.items():
            construct_print(f"Testing with testset: {data_name}")
            self.te_loader = create_loader(
                data_path=data_path,
                training=False,
                prefix=self.arg_dict["prefix"],
                get_length=False,
            )
            self.save_path = os.path.join(self.path_dict["save"], data_name)
            if not os.path.exists(self.save_path):
                construct_print(f"{self.save_path} do not exist. Let's create it.")
                os.makedirs(self.save_path)
            results = self.__test_process(save_pre=self.save_pre)

    def __test_process(self, save_pre):
        loader = self.te_loader

        pres = [AvgMeter() for _ in range(256)]
        recs = [AvgMeter() for _ in range(256)]
        meanfs = AvgMeter()
        maes = AvgMeter()

        tqdm_iter = tqdm(enumerate(loader), total=len(loader), leave=False)
        for test_batch_id, test_data in tqdm_iter:
            tqdm_iter.set_description(f"{self.exp_name}: te=>{test_batch_id + 1}")
            with torch.no_grad():
                in_imgs, in_mask_paths, in_names = test_data
                print(in_imgs.shape)
                in_imgs = in_imgs.to(self.dev, non_blocking=True)
                outputs = self.net(in_imgs)

                pred = bv_test(outputs)

                outputs_np = pred.cpu().detach()

            for item_id, out_item in enumerate(outputs_np):
                gimg_path = os.path.join(in_mask_paths[item_id])
                gt_img = Image.open(gimg_path).convert("L")
                out_img = self.to_pil(out_item).resize(gt_img.size, resample=Image.NEAREST)
                # gt_img = gt_img.resize([320,320], resample=Image.BILINEAR)
                if save_pre:
                    oimg_path = os.path.join(self.save_path, in_names[item_id] + ".png")
                    # gt_img.save(oimg_path)
                    out_img.save(oimg_path)

