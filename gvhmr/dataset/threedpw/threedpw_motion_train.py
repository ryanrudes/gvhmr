import numpy as np
import torch

from gvhmr.configs import MainStore, builds
from gvhmr.dataset.imgfeat_motion.base_dataset import ImgfeatMotionDatasetBase
from gvhmr.utils.assets import DATA_ROOT
from gvhmr.utils.geo.hmr_cam import resize_K
from gvhmr.utils.geo_transform import compute_cam_angvel
from gvhmr.utils.net_utils import get_valid_mask, repeat_to_max_len, repeat_to_max_len_dict
from gvhmr.utils.pylogger import Log
from gvhmr.utils.smplx_utils import make_smplx
from gvhmr.utils.video_io_utils import read_video_np, save_video
from gvhmr.utils.vis.renderer_utils import simple_render_mesh_background


class ThreedpwSmplDataset(ImgfeatMotionDatasetBase):
    def __init__(self, imgfeat_subdir="imgfeats/3dpw_train_smplx_refit", serve_crops=False):
        # Path. `imgfeat_subdir` selects which cached-feature backbone to train on (default HMR2; e.g.
        # "imgfeats/3dpw_train_dinov2" for a re-extracted DINOv2 set — see docs/EXTENSIBILITY.md Tier B).
        self.hmr4d_support_dir = DATA_ROOT / "3DPW/hmr4d_support"  # honours $GVHMR_DATA_ROOT (default inputs/)
        self.imgfeat_subdir = imgfeat_subdir
        # serve_crops: also decode the clip's raw crops (L,3,256,256) for in-loop LoRA backbone training
        # (ROADMAP Regime B). Default off -> the item dict is unchanged and the cached-feature path is
        # byte-identical. The crops reproduce what the extractor fed the ViT, so f_imgseq matches at LoRA-0.
        self.serve_crops = serve_crops
        self.dataset_name = "3DPW"

        # Setting
        self.min_motion_frames = 60
        self.max_motion_frames = 120
        super().__init__()

    def _load_dataset(self):
        self.train_labels = torch.load(self.hmr4d_support_dir / "train_3dpw_gt_labels.pt", weights_only=False)
        self.refit_smplx = torch.load(self.hmr4d_support_dir / "train_refit_smplx.pt", weights_only=False)
        if True:  # Remove clips that have obvious error
            update_list = {
                "courtyard_basketball_00_1": [(0, 300), (340, 468)],
                "courtyard_laceShoe_00_0": [(0, 620), (780, 931)],
                "courtyard_rangeOfMotions_00_1": [(0, 370), (410, 601)],
                "courtyard_shakeHands_00_1": [(0, 100), (120, 391)],
            }
            for k, v in update_list.items():
                self.refit_smplx[k]["valid_range_list"] = v

        self.f_img_folder = self.hmr4d_support_dir / self.imgfeat_subdir
        # Only keep sequences whose feature file exists — lets a partial re-extraction (e.g. a few vids
        # with a new backbone) train without regenerating the whole set.
        self.refit_smplx = {v: d for v, d in self.refit_smplx.items() if (self.f_img_folder / f"{v}.pt").exists()}
        Log.info(f"[{self.dataset_name}] Train ({len(self.refit_smplx)} seqs with '{self.imgfeat_subdir}' features)")

    def _get_idx2meta(self):
        # We expect to see the entire sequence during one epoch,
        # so each sequence will be sampled max(SeqLength // MotionFrames, 1) times
        seq_lengths = []
        self.idx2meta = []
        for vid in self.refit_smplx:
            valid_range_list = self.refit_smplx[vid]["valid_range_list"]
            for start, end in valid_range_list:
                seq_length = end - start
                num_samples = max(seq_length // self.max_motion_frames, 1)
                seq_lengths.append(seq_length)
                self.idx2meta.extend([(vid, start, end)] * num_samples)
        minutes = sum(seq_lengths) / 25 / 60
        Log.info(
            f"[{self.dataset_name}] has {minutes:.1f} minutes motion -> Resampled to {len(self.idx2meta)} samples."
        )

    def _load_data(self, idx):
        data = {}
        vid, range1, range2 = self.idx2meta[idx]

        # Random select a subset
        mlength = range2 - range1
        min_motion_len = self.min_motion_frames
        max_motion_len = self.max_motion_frames

        if mlength < min_motion_len:  # this may happen, the minimal mlength is around 30
            start = range1
            length = mlength
        else:
            effect_max_motion_len = min(max_motion_len, mlength)
            length = np.random.randint(min_motion_len, effect_max_motion_len + 1)  # [low, high)
            start = np.random.randint(range1, range2 - length + 1)
        end = start + length
        data["length"] = length
        data["meta"] = {"data_name": self.dataset_name, "idx": idx, "vid": vid, "start_end": (start, end)}

        # Select motion subset
        data["smplx_params_incam"] = {k: v[start:end] for k, v in self.refit_smplx[vid]["smplx_params_incam"].items()}
        data["K_fullimg"] = self.train_labels[vid]["K_fullimg"]
        data["T_w2c"] = self.train_labels[vid]["T_w2c"][start:end]

        # Img (as feature):
        f_img_dict = torch.load(self.f_img_folder / f"{vid}.pt", weights_only=False)
        data["bbx_xys"] = f_img_dict["bbx_xys"][start:end]  # (F, 3)
        data["f_imgseq"] = f_img_dict["features"][start:end].float()  # (F, 3)
        data["img_wh"] = f_img_dict["img_wh"]  # (2)
        data["kp2d"] = torch.zeros((end - start), 17, 3)  # (L, 17, 3)  # do not provide kp2d

        # Raw crops for in-loop backbone training — decode only this clip's frames, crop with the same
        # bbx the extractor used, so JointBackbone(crops) reproduces the cached f_imgseq at LoRA-0.
        if self.serve_crops:
            from gvhmr.utils.imgcrop import get_batch  # dpvo-free: safe to import in a forked worker

            video_path = self.hmr4d_support_dir / f"videos/{vid[:-2]}.mp4"
            crops, _ = get_batch(str(video_path), data["bbx_xys"], img_ds=0.5, start_frame=start, end_frame=end)
            data["crops"] = crops.float()  # (L, 3, 256, 256)

        return data

    def _process_data(self, data, idx):
        length = data["length"]

        smpl_params_c = data["smplx_params_incam"]
        smpl_params_w_zero = {k: torch.zeros_like(v) for k, v in smpl_params_c.items()}
        K_fullimg = data["K_fullimg"][None].repeat(length, 1, 1)
        cam_angvel = compute_cam_angvel(data["T_w2c"][:, :3, :3])

        max_len = self.max_motion_frames
        return_data = {
            "meta": data["meta"],
            "length": length,
            "smpl_params_c": smpl_params_c,
            "smpl_params_w": smpl_params_w_zero,
            "R_c2gv": torch.zeros(length, 3, 3),  # (F, 3, 3)
            "gravity_vec": torch.zeros(3),  # (3)
            "bbx_xys": data["bbx_xys"],  # (F, 3)
            "K_fullimg": K_fullimg,  # (F, 3, 3)
            "f_imgseq": data["f_imgseq"],  # (F, D)
            "kp2d": data["kp2d"],  # (F, 17, 3)
            "cam_angvel": cam_angvel,  # (F, 6)
            "mask": {
                "valid": get_valid_mask(max_len, length),
                "vitpose": False,
                "bbx_xys": True,
                "f_imgseq": True,
                "spv_incam_only": True,
            },
        }
        if "crops" in data:
            return_data["crops"] = data["crops"]  # (F, 3, 256, 256)

        if False:  # Debug, render incam
            start, end = data["meta"]["start_end"]
            vid = data["meta"]["vid"]

            ds = 0.5
            faces = smplx.faces
            smplx = make_smplx("supermotion")
            smplx_c_verts = smplx(**return_data["smpl_params_c"]).vertices
            K_render = resize_K(K_fullimg, ds)

            video_path = self.hmr4d_support_dir / f"videos/{vid[:-2]}.mp4"
            images = read_video_np(video_path, scale=ds, start_frame=start, end_frame=end)

            render_dict = {
                "K": K_render[:1],  # only support batch size 1
                "faces": faces,
                "verts": smplx_c_verts,
                "background": images,
            }
            img_overlay = simple_render_mesh_background(render_dict, VI=10)
            save_video(img_overlay, "tmp.mp4", crf=28)

        # Batchable
        return_data["smpl_params_c"] = repeat_to_max_len_dict(return_data["smpl_params_c"], max_len)
        return_data["smpl_params_w"] = repeat_to_max_len_dict(return_data["smpl_params_w"], max_len)
        return_data["R_c2gv"] = repeat_to_max_len(return_data["R_c2gv"], max_len)
        return_data["bbx_xys"] = repeat_to_max_len(return_data["bbx_xys"], max_len)
        return_data["K_fullimg"] = repeat_to_max_len(return_data["K_fullimg"], max_len)
        return_data["f_imgseq"] = repeat_to_max_len(return_data["f_imgseq"], max_len)
        return_data["kp2d"] = repeat_to_max_len(return_data["kp2d"], max_len)
        return_data["cam_angvel"] = repeat_to_max_len(return_data["cam_angvel"], max_len)
        if "crops" in return_data:  # pad the crops the same way (padded frames are masked downstream)
            return_data["crops"] = repeat_to_max_len(return_data["crops"], max_len)

        return return_data


# 3DPW
MainStore.store(name="v1", node=builds(ThreedpwSmplDataset), group="train_datasets/imgfeat_3dpw")
MainStore.store(  # serves raw crops for in-loop LoRA backbone training (ROADMAP Regime B stage 3)
    name="v1_crops",
    node=builds(ThreedpwSmplDataset, serve_crops=True),
    group="train_datasets/imgfeat_3dpw",
)
MainStore.store(  # re-extracted DINOv2 features (Tier B backbone swap)
    name="dinov2",
    node=builds(ThreedpwSmplDataset, imgfeat_subdir="imgfeats/3dpw_train_dinov2"),
    group="train_datasets/imgfeat_3dpw",
)
MainStore.store(  # re-extracted Sapiens features (Tier B backbone swap) — see docs/ROADMAP.md A1
    name="sapiens",
    node=builds(ThreedpwSmplDataset, imgfeat_subdir="imgfeats/3dpw_train_sapiens"),
    group="train_datasets/imgfeat_3dpw",
)
MainStore.store(  # Sapiens with 2x2 spatial pooling instead of GAP (4096-d) — the A1 pooling ablation.
    # GAP-Sapiens lost to HMR2 by 32mm (74.5 vs 42.8); it averages a 64x64 feature map into one vector,
    # destroying the spatial structure that IS the pose signal. grid2 keeps a coarse layout and strictly
    # contains GAP, so it isolates "is Sapiens bad, or was the pooling bad?".
    name="sapiens_grid2",
    node=builds(ThreedpwSmplDataset, imgfeat_subdir="imgfeats/3dpw_train_sapiens_grid2"),
    group="train_datasets/imgfeat_3dpw",
)
