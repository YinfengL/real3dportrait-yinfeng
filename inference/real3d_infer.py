import os
import sys
sys.path.append('./')
import torch
import torch.nn.functional as F
import torchshow as ts
import librosa
import random
import time
import numpy as np
import importlib
import tqdm
import copy
import cv2
import math
import traceback
import subprocess
import uuid
import hashlib
 
# common utils
from utils.commons.hparams import hparams, set_hparams
from utils.commons.tensor_utils import move_to_cuda, convert_to_tensor
from utils.commons.ckpt_utils import load_ckpt, get_last_checkpoint
 
# 3DMM-related utils
from deep_3drecon.deep_3drecon_models.bfm import ParametricFaceModel
from data_util.face3d_helper import Face3DHelper
from data_gen.utils.process_image.fit_3dmm_landmark import fit_3dmm_for_a_image
from data_gen.utils.process_video.fit_3dmm_landmark import fit_3dmm_for_a_video
from deep_3drecon.secc_renderer import SECC_Renderer
from data_gen.eg3d.convert_to_eg3d_convention import get_eg3d_convention_camera_pose_intrinsic
from data_gen.utils.process_image.extract_lm2d import extract_lms_mediapipe_job
 
# Face Parsing
from data_gen.utils.mp_feature_extractors.mp_segmenter import MediapipeSegmenter
from data_gen.utils.process_video.extract_segment_imgs import inpaint_torso_job, extract_background
 
# other inference utils
from inference.infer_utils import mirror_index, load_img_to_512_hwc_array, load_img_to_normalized_512_bchw_tensor
from inference.infer_utils import smooth_camera_sequence, smooth_features_xd, save_wav16k
 
from inference.edit_secc import blink_eye_for_secc
 
 
def read_first_frame_from_a_video(vid_name):
    cap = cv2.VideoCapture(vid_name)
    ret, frame_bgr = cap.read()
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_rgb
 
 
def analyze_weights_img(gen_output):
    # 仅用于调试，保持原样
    if 'image_raw' not in gen_output or 'weights_img' not in gen_output:
        return
    img_raw = gen_output['image_raw']
    mask_005_to_03 = torch.bitwise_and(gen_output['weights_img'] > 0.05, gen_output['weights_img'] < 0.3).repeat([1, 3, 1, 1])
    mask_005_to_05 = torch.bitwise_and(gen_output['weights_img'] > 0.05, gen_output['weights_img'] < 0.5).repeat([1, 3, 1, 1])
    mask_005_to_07 = torch.bitwise_and(gen_output['weights_img'] > 0.05, gen_output['weights_img'] < 0.7).repeat([1, 3, 1, 1])
    mask_005_to_09 = torch.bitwise_and(gen_output['weights_img'] > 0.05, gen_output['weights_img'] < 0.9).repeat([1, 3, 1, 1])
    mask_005_to_10 = torch.bitwise_and(gen_output['weights_img'] > 0.05, gen_output['weights_img'] < 1.0).repeat([1, 3, 1, 1])
 
    img_raw_005_to_03 = img_raw.clone()
    img_raw_005_to_03[~mask_005_to_03] = -1
 
 
def cal_face_area_percent(img_name):
    img = cv2.resize(cv2.imread(img_name)[:, :, ::-1], (512, 512))
    lm478 = extract_lms_mediapipe_job(img) / 512
    min_x = lm478[:, 0].min()
    max_x = lm478[:, 0].max()
    min_y = lm478[:, 1].min()
    max_y = lm478[:, 1].max()
    area = (max_x - min_x) * (max_y - min_y)
    return area
 
 
def crop_img_on_face_area_percent(img_name, out_name='temp/cropped_src_img.png', min_face_area_percent=0.2):
    try:
        os.makedirs(os.path.dirname(out_name), exist_ok=True)
    except:
        pass
    face_area_percent = cal_face_area_percent(img_name)
    if face_area_percent >= min_face_area_percent:
        print(f"face area percent {face_area_percent} larger than threshold {min_face_area_percent}, directly use the input image...", flush=True)
        cmd = f"cp {img_name} {out_name}"
        os.system(cmd)
        return out_name
    else:
        print(f"face area percent {face_area_percent} smaller than threshold {min_face_area_percent}, crop the input image...", flush=True)
        img = cv2.resize(cv2.imread(img_name)[:, :, ::-1], (512, 512))
        lm478 = extract_lms_mediapipe_job(img).astype(int)
        min_x = lm478[:, 0].min()
        max_x = lm478[:, 0].max()
        min_y = lm478[:, 1].min()
        max_y = lm478[:, 1].max()
        face_area = (max_x - min_x) * (max_y - min_y)
        target_total_area = face_area / min_face_area_percent
        target_hw = int(target_total_area ** 0.5)
        center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
        shrink_pixels = 2 * max(
            -(center_x - target_hw / 2),
            center_x + target_hw / 2 - 512,
            -(center_y - target_hw / 2),
            center_y + target_hw / 2 - 512
        )
        shrink_pixels = max(0, shrink_pixels)
        hw = math.floor(target_hw - shrink_pixels)
        new_min_x = int(center_x - hw / 2)
        new_max_x = int(center_x + hw / 2)
        new_min_y = int(center_y - hw / 2)
        new_max_y = int(center_y + hw / 2)
 
        img = img[new_min_y:new_max_y, new_min_x:new_max_x]
        img = cv2.resize(img, (512, 512))
        cv2.imwrite(out_name, img[:, :, ::-1])
        return out_name
 
 
class GeneFace2Infer:
    def __init__(self, audio2secc_dir, head_model_dir, torso_model_dir, device=None, inp=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # 加载模型
        self.audio2secc_model = self.load_audio2secc(audio2secc_dir)
        self.secc2video_model = self.load_secc2video(head_model_dir, torso_model_dir, inp)
        
        # ✅【纯 FP32 模式】只移动设备，不做任何 half() 转换
        self.audio2secc_model.to(device).eval()
        self.secc2video_model.to(device).eval()
        
 
        # 初始化辅助工具 (保持默认)
        self.seg_model = MediapipeSegmenter()
        self.secc_renderer = SECC_Renderer(512)
        self.face3d_helper = Face3DHelper(use_gpu=True, keypoint_mode='lm68')
        self.mp_face3d_helper = Face3DHelper(use_gpu=True, keypoint_mode='mediapipe')
        
        # 🔥 初始化静态缓存
        self.static_cache = {}

    def load_audio2secc(self, audio2secc_dir):
        config_name = f"{audio2secc_dir}/config.yaml" if not audio2secc_dir.endswith(".ckpt") else f"{os.path.dirname(audio2secc_dir)}/config.yaml"
        set_hparams(f"{config_name}", print_hparams=False)
        self.audio2secc_dir = audio2secc_dir
        self.audio2secc_hparams = copy.deepcopy(hparams)
        from modules.audio2motion.vae import VAEModel, PitchContourVAEModel
        try:
            from modules.audio2motion.in_context_audio2motion import InContextAudio2MotionModel
        except ImportError:
            InContextAudio2MotionModel = None
 
        if self.audio2secc_hparams['audio_type'] == 'hubert':
            audio_in_dim = 1024
        elif self.audio2secc_hparams['audio_type'] == 'mfcc':
            audio_in_dim = 13
 
        if 'icl' in hparams['task_cls'] and InContextAudio2MotionModel:
            self.use_icl_audio2motion = True
            model = InContextAudio2MotionModel(hparams['icl_model_type'], hparams=self.audio2secc_hparams)
        else:
            self.use_icl_audio2motion = False
            if hparams.get("use_pitch", False) is True:
                model = PitchContourVAEModel(hparams, in_out_dim=64, audio_in_dim=audio_in_dim)
            else:
                model = VAEModel(in_out_dim=64, audio_in_dim=audio_in_dim)
        load_ckpt(model, f"{audio2secc_dir}", model_name='model', strict=True)
        return model
 
    def load_secc2video(self, head_model_dir, torso_model_dir, inp):
        if inp is None:
            inp = {}
        self.head_model_dir = head_model_dir
        self.torso_model_dir = torso_model_dir
        if torso_model_dir != '':
            if torso_model_dir.endswith(".ckpt"):
                set_hparams(f"{os.path.dirname(torso_model_dir)}/config.yaml", print_hparams=False)
            else:
                set_hparams(f"{torso_model_dir}/config.yaml", print_hparams=False)
            if inp.get('head_torso_threshold', None) is not None:
                hparams['htbsr_head_threshold'] = inp['head_torso_threshold']
            self.secc2video_hparams = copy.deepcopy(hparams)
            from modules.real3d.secc_img2plane_torso import OSAvatarSECC_Img2plane_Torso
            model = OSAvatarSECC_Img2plane_Torso()
            load_ckpt(model, f"{torso_model_dir}", model_name='model', strict=True)
            if head_model_dir != '':
                print("| Warning: Assigned --torso_ckpt which also contains head, but --head_ckpt is also assigned, skipping the --head_ckpt.", flush=True)
        else:
            from modules.real3d.secc_img2plane_torso import OSAvatarSECC_Img2plane
            if head_model_dir.endswith(".ckpt"):
                set_hparams(f"{os.path.dirname(head_model_dir)}/config.yaml", print_hparams=False)
            else:
                set_hparams(f"{head_model_dir}/config.yaml", print_hparams=False)
            if inp.get('head_torso_threshold', None) is not None:
                hparams['htbsr_head_threshold'] = inp['head_torso_threshold']
            self.secc2video_hparams = copy.deepcopy(hparams)
            model = OSAvatarSECC_Img2plane()
            load_ckpt(model, f"{head_model_dir}", model_name='model', strict=True)
        return model

    def prepare_static_batch(self, src_image_path, drv_pose_path, bg_image_path, min_face_area_percent=0.2):
        """
        预处理与音频无关的部分：源图特征、驱动视频姿态、背景、分割掩码。
        结果缓存在内存中，避免重复计算。
        """
        # 生成唯一缓存键
        cache_key = hashlib.md5(f"{src_image_path}_{drv_pose_path}_{bg_image_path}".encode()).hexdigest()
        
        # 检查全局缓存
        if cache_key in self.static_cache:
            return copy.deepcopy(self.static_cache[cache_key]) # 深拷贝以防修改

        print(f"[STATIC CACHE MISS] Preprocessing static data for key: {cache_key[:8]}...", flush=True)
        
        sample = {}
        
        # 1. 处理源图像 (Src Image Processing)
        task_id = "static_prep"
        tmp_img_name = f'infer_out/tmp/cropped_src_static_{task_id}.png'
        crop_img_on_face_area_percent(src_image_path, tmp_img_name, min_face_area_percent=min_face_area_percent)
        
        image_name = tmp_img_name
        if image_name.endswith(".mp4"):
            img = read_first_frame_from_a_video(image_name)
            image_name = image_name[:-4] + '.png'
            cv2.imwrite(image_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        sample['ref_gt_img'] = load_img_to_normalized_512_bchw_tensor(image_name).cuda()
        img = load_img_to_512_hwc_array(image_name)

        segmap = self.seg_model._cal_seg_map(np.asarray(img).astype(np.uint8))
        if isinstance(segmap, tuple): segmap = segmap[0]
        sample['segmap'] = torch.tensor(segmap).float().unsqueeze(0).cuda()

        head_img = self.seg_model._seg_out_img_with_segmap(np.asarray(img), segmap, mode='head')[0]
        sample['ref_head_img'] = ((torch.tensor(head_img) - 127.5) / 127.5).float().unsqueeze(0).permute(0, 3, 1, 2).cuda()
        
        inpaint_ret = inpaint_torso_job(np.asarray(img), segmap)
        inpaint_torso_img = inpaint_ret[0] if isinstance(inpaint_ret, (tuple, list)) else inpaint_ret
        sample['ref_torso_img'] = ((torch.tensor(inpaint_torso_img) - 127.5) / 127.5).float().unsqueeze(0).permute(0, 3, 1, 2).cuda()

        if bg_image_path == '':
            bg_img = extract_background([np.asarray(img)], [segmap], 'knn')
        else:
            bg_img = cv2.imread(bg_image_path)
            if bg_img is not None:
                bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            else:
                bg_img = np.zeros((512, 512, 3), dtype=np.uint8)
        bg_img = cv2.resize(bg_img, (512, 512))
        sample['bg_img'] = ((torch.tensor(bg_img) - 127.5) / 127.5).float().unsqueeze(0).permute(0, 3, 1, 2).cuda()

        # 2. 拟合源图 3DMM (Src 3DMM Fit)
        coeff_dict = fit_3dmm_for_a_image(image_name, save=False)
        src_id = torch.tensor(coeff_dict['id']).reshape([1, 80]).cuda()
        src_exp = torch.tensor(coeff_dict['exp']).reshape([1, 64]).cuda()
        src_euler = torch.tensor(coeff_dict['euler']).reshape([1, 3]).cuda()
        src_trans = torch.tensor(coeff_dict['trans']).reshape([1, 3]).cuda()
        
        # 存储源 ID，后续动态部分会用到
        sample['src_id_base'] = src_id
        sample['src_exp_base'] = src_exp
        sample['src_euler_base'] = src_euler
        sample['src_trans_base'] = src_trans

        # 3. 提取驱动视频姿态 (Drv Pose Extraction) - 最耗时的部分
        print(f"| To extract pose from {drv_pose_path}", flush=True)
        if drv_pose_path.endswith('.mp4'):
            try:
                drv_pose_coeff_dict = fit_3dmm_for_a_video(drv_pose_path, save=False, keypoint_mode='mediapipe')
            except Exception as e:
                print(f"[WARN] fit_3dmm_for_a_video failed: {e}, fallback to static", flush=True)
                drv_pose_coeff_dict = {
                    'euler': np.repeat(np.array(coeff_dict['euler'])[None, :], 100, axis=0), # 默认长度
                    'trans': np.repeat(np.array(coeff_dict['trans'])[None, :], 100, axis=0),
                }
        else:
            drv_pose_coeff_dict = np.load(drv_pose_path, allow_pickle=True).tolist()

        eulers = convert_to_tensor(drv_pose_coeff_dict['euler']).reshape([-1, 3]).cuda()
        trans = convert_to_tensor(drv_pose_coeff_dict['trans']).reshape([-1, 3]).cuda()
        
        sample['drv_eulers'] = eulers
        sample['drv_trans'] = trans
        
        # 缓存结果
        self.static_cache[cache_key] = sample
        
        print(f"[STATIC CACHE SAVED] Key: {cache_key[:8]}", flush=True)
        return copy.deepcopy(sample)

    def prepare_dynamic_batch(self, static_sample, hubert, f0, t_x, inp):
        """
        结合静态数据和动态音频数据，生成最终推理批次。
        """
        batch = {}
        
        # 1. 复制静态数据
        batch['ref_head_img'] = static_sample['ref_head_img']
        batch['ref_torso_img'] = static_sample['ref_torso_img']
        batch['bg_img'] = static_sample['bg_img']
        batch['segmap'] = static_sample['segmap']
        
        # 2. 处理音频相关
        x_mask = torch.ones([1, t_x]).float().cuda()
        y_mask = torch.ones([1, t_x // 2]).float().cuda()
        batch.update({
            'hubert': hubert.cuda(),
            'f0': f0.cuda(),
            'x_mask': x_mask,
            'y_mask': y_mask,
            'blink': torch.zeros([1, t_x, 1]).long().cuda(),
            'audio': hubert.cuda(),
            'eye_amp': torch.ones([1, 1]).cuda(),
            'mouth_amp': torch.ones([1, 1]).cuda() * inp['mouth_amp'],
        })
        
        # 3. 构建动态 3DMM 系数
        # 源 ID 重复 T 次
        batch['id'] = static_sample['src_id_base'].repeat([t_x // 2, 1])
        
        # 源关键点
        src_kp = self.face3d_helper.reconstruct_lm2d(
            static_sample['src_id_base'], 
            static_sample['src_exp_base'], 
            static_sample['src_euler_base'], 
            static_sample['src_trans_base']
        )
        # 🔥 修复关键点维度：确保是 3D (N, K, 3)
        if src_kp.shape[-1] == 2:
            src_kp = torch.cat([src_kp, torch.zeros_like(src_kp[..., :1])], dim=-1)
        src_kp = (src_kp - 0.5) / 0.5
        batch['src_kp'] = torch.clamp(src_kp, -1, 1).repeat([t_x // 2, 1, 1])
        
        # 驱动姿态映射
        eulers = static_sample['drv_eulers']
        trans = static_sample['drv_trans']
        len_pose = len(eulers)
        index_lst = [mirror_index(i, len_pose) for i in range(t_x // 2)]
        batch['euler'] = eulers[index_lst]
        batch['trans'] = trans[index_lst]
        
        # Map to init pose logic
        if inp.get("map_to_init_pose", True):
            diff_euler = static_sample['src_euler_base'] - batch['euler'][0:1]
            batch['euler'] = batch['euler'] + diff_euler
            diff_trans = static_sample['src_trans_base'] - batch['trans'][0:1]
            batch['trans'] = batch['trans'] + diff_trans
            
        batch['trans'][:, -1] = batch['trans'][0:1, -1].repeat([batch['trans'].shape[0]])
        
        # Camera
        camera_ret = get_eg3d_convention_camera_pose_intrinsic({
            'euler': batch['euler'].cpu(),
            'trans': batch['trans'].cpu()
        })
        c2w, intrinsics = camera_ret['c2w'], camera_ret['intrinsics']
        camera = np.concatenate([c2w.reshape([-1, 16]), intrinsics.reshape([-1, 9])], axis=-1)
        camera = smooth_camera_sequence(camera, kernel_size=7)
        batch['camera'] = torch.tensor(camera).cuda().float()
        
        return batch

    def infer_once(self, inp):
        try:
            print(">>> enter infer_once", flush=True)
            self.inp = inp
            
            # 🔥 优化路径：如果可能，使用静态缓存
            # 注意：为了兼容旧代码，这里仍然调用 prepare_batch_from_inp，
            # 但建议在 main_api 中重构调用链，直接调用 prepare_static_batch + prepare_dynamic_batch
            
            samples = self.prepare_batch_from_inp(inp)
            
            seed = inp['seed'] if inp['seed'] is not None else int(time.time())
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            out_name = self.forward_system(samples, inp)
            print(">>> exit infer_once OK", flush=True)
            return out_name
        except Exception:
            err = traceback.format_exc()
            print(">>> infer_once FAILED", flush=True)
            print(err, flush=True)
            with open("inference_error.log", "a") as f:
                f.write("\n===== infer_once =====\n")
                f.write(err + "\n")
            raise
            
    
    def safe_cal_seg_map(self, img):
        try:
            result = self.seg_model._cal_seg_map(img)
            if isinstance(result, tuple):
                segmap, _, _, _ = result
            else:
                segmap = result
            return segmap
        except Exception as e:
            print(f"[ERROR] safe_cal_seg_map failed completely: {e}")
            raise e
 
    def prepare_batch_from_inp(self, inp):
        task_id = os.path.basename(inp['out_name']).replace('.mp4', '') if inp.get('out_name') else str(uuid.uuid4())[:8]
        tmp_img_name = f'infer_out/tmp/cropped_src_{task_id}.png'
        
        crop_img_on_face_area_percent(inp['src_image_name'], tmp_img_name, min_face_area_percent=inp['min_face_area_percent'])
        inp['src_image_name'] = tmp_img_name
 
        sample = {}
        if inp['drv_audio_name'][-4:] in ['.wav', '.mp3']:
            wav16k_name = save_wav16k(inp['drv_audio_name'])
            inp['drv_audio_name'] = wav16k_name
 
            if self.audio2secc_hparams['audio_type'] == 'hubert':
                hubert = self.get_hubert(wav16k_name)
            elif self.audio2secc_hparams['audio_type'] == 'mfcc':
                hubert = self.get_mfcc(wav16k_name) / 100
 
            f0 = self.get_f0(wav16k_name)
 
            if isinstance(f0, torch.Tensor):
                f0 = f0.cpu().numpy()
            if isinstance(hubert, torch.Tensor):
                hubert = hubert.cpu().numpy()
 
            if f0.shape[0] > len(hubert):
                f0 = f0[:len(hubert)]
            else:
                num_to_pad = len(hubert) - len(f0)
                f0 = np.pad(f0, pad_width=((0, num_to_pad), (0, 0)))
 
            t_x = hubert.shape[0]
            x_mask = torch.ones([1, t_x]).float()
            y_mask = torch.ones([1, t_x // 2]).float()
            sample.update({
                'hubert': torch.from_numpy(hubert).float().unsqueeze(0).cuda(),
                'f0': torch.from_numpy(f0).float().reshape([1, -1]).cuda(),
                'x_mask': x_mask.cuda(),
                'y_mask': y_mask.cuda(),
            })
            sample['blink'] = torch.zeros([1, t_x, 1]).long().cuda()
            sample['audio'] = sample['hubert']
            sample['eye_amp'] = torch.ones([1, 1]).cuda() * 1.0
            sample['mouth_amp'] = torch.ones([1, 1]).cuda() * inp['mouth_amp']
 
        elif inp['drv_audio_name'][-4:] in ['.mp4']:
            drv_motion_coeff_dict = fit_3dmm_for_a_video(inp['drv_audio_name'], save=False)
            drv_motion_coeff_dict = convert_to_tensor(drv_motion_coeff_dict)
            t_x = drv_motion_coeff_dict['exp'].shape[0] * 2
            self.drv_motion_coeff_dict = drv_motion_coeff_dict
 
        elif inp['drv_audio_name'][-4:] in ['.npy']:
            drv_motion_coeff_dict = np.load(inp['drv_audio_name'], allow_pickle=True).tolist()
            drv_motion_coeff_dict = convert_to_tensor(drv_motion_coeff_dict)
            t_x = drv_motion_coeff_dict['exp'].shape[0] * 2
            self.drv_motion_coeff_dict = drv_motion_coeff_dict
 
        image_name = inp['src_image_name']
        if image_name.endswith(".mp4"):
            img = read_first_frame_from_a_video(image_name)
            image_name = inp['src_image_name'] = image_name[:-4] + '.png'
            cv2.imwrite(image_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        sample['ref_gt_img'] = load_img_to_normalized_512_bchw_tensor(image_name).cuda()
        img = load_img_to_512_hwc_array(image_name)
 
        img = np.asarray(img)
        if img.ndim == 4:
            img = img[0]
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.ndim != 3:
            raise ValueError(f"Unexpected image shape for segmentation: {img.shape}")
 
        print("[DEBUG] seg img shape:", np.asarray(img).shape, np.asarray(img).dtype, flush=True)
        
        segmap = None
        try:
            img_for_seg = np.asarray(img)
            if img_for_seg.dtype != np.uint8:
                img_for_seg = img_for_seg.astype(np.uint8)
            if img_for_seg.ndim == 4:
                img_for_seg = img_for_seg[0]
            if img_for_seg.shape[-1] != 3:
                raise ValueError(f"unexpected image shape for seg: {img_for_seg.shape}")
        
            result = self.seg_model._cal_seg_map(img_for_seg)
            if isinstance(result, tuple):
                segmap = result[0]
            else:
                segmap = result
                
        except Exception as e:
            print(f"[WARN] seg_model._cal_seg_map failed, fallback to safe_cal_seg_map: {e}", flush=True)
            segmap = self.safe_cal_seg_map(img)
            if isinstance(segmap, tuple):
                segmap = segmap[0]
        
        if not isinstance(segmap, np.ndarray):
            segmap = np.array(segmap)
            
        sample['segmap'] = torch.tensor(segmap).float().unsqueeze(0).cuda()

        head_img = self.seg_model._seg_out_img_with_segmap(img, segmap, mode='head')[0]
        sample['ref_head_img'] = ((torch.tensor(head_img) - 127.5) / 127.5).float().unsqueeze(0).permute(0, 3, 1, 2).cuda()
 
        inpaint_ret = inpaint_torso_job(img, segmap)
        if isinstance(inpaint_ret, (tuple, list)):
            inpaint_torso_img = inpaint_ret[0]
        else:
            inpaint_torso_img = inpaint_ret
        sample['ref_torso_img'] = ((torch.tensor(inpaint_torso_img) - 127.5) / 127.5).float().unsqueeze(0).permute(0, 3, 1, 2).cuda()
 
        if inp['bg_image_name'] == '':
            bg_img = extract_background([img], [segmap], 'knn')
        else:
            bg_img = cv2.imread(inp['bg_image_name'])
            
        if bg_img is None:
            print("[WARN] Background extraction failed or image not found, using black background.", flush=True)
            bg_img = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            if len(bg_img.shape) == 3 and bg_img.shape[2] == 3:
                bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            else:
                bg_img = cv2.cvtColor(cv2.cvtColor(bg_img, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
 
        bg_img = cv2.resize(bg_img, (512, 512))
        sample['bg_img'] = ((torch.tensor(bg_img) - 127.5) / 127.5).float().unsqueeze(0).permute(0, 3, 1, 2).cuda()
 
 
        coeff_dict = fit_3dmm_for_a_image(image_name, save=False)
        assert coeff_dict is not None
 
        src_id = torch.tensor(coeff_dict['id']).reshape([1, 80]).cuda()
        src_exp = torch.tensor(coeff_dict['exp']).reshape([1, 64]).cuda()
        src_euler = torch.tensor(coeff_dict['euler']).reshape([1, 3]).cuda()
        src_trans = torch.tensor(coeff_dict['trans']).reshape([1, 3]).cuda()
        sample['id'] = src_id.repeat([t_x // 2, 1])
        
        # 🔥 修复 1: 确保 src_kp 是 3D 关键点
        src_kp = self.face3d_helper.reconstruct_lm2d(src_id, src_exp, src_euler, src_trans)
        if src_kp.shape[-1] == 2:
            src_kp = torch.cat([src_kp, torch.zeros_like(src_kp[..., :1])], dim=-1)
        src_kp = (src_kp - 0.5) / 0.5
        sample['src_kp'] = torch.clamp(src_kp, -1, 1).repeat([t_x // 2, 1, 1])
 
        print(f"| To extract pose from {inp['drv_pose_name']}", flush=True)
        if inp['drv_pose_name'] == 'static':
            sample['euler'] = torch.tensor(coeff_dict['euler']).reshape([1, 3]).cuda().repeat([t_x // 2, 1])
            sample['trans'] = torch.tensor(coeff_dict['trans']).reshape([1, 3]).cuda().repeat([t_x // 2, 1])
        else:
            if inp['drv_pose_name'].endswith('.mp4'):
                try:
                    ret = fit_3dmm_for_a_video(inp['drv_pose_name'], save=False, keypoint_mode='mediapipe')
                    if not isinstance(ret, dict):
                        raise ValueError(f"fit_3dmm_for_a_video returned invalid type: {type(ret)}")
                    drv_pose_coeff_dict = ret
                except Exception as e:
                    print(f"[WARN] fit_3dmm_for_a_video failed: {e}", flush=True)
                    print("[WARN] fallback to static pose", flush=True)
                    drv_pose_coeff_dict = {
                        'euler': np.repeat(np.array(coeff_dict['euler'])[None, :], t_x // 2, axis=0),
                        'trans': np.repeat(np.array(coeff_dict['trans'])[None, :], t_x // 2, axis=0),
                    }
            else:
                drv_pose_coeff_dict = np.load(inp['drv_pose_name'], allow_pickle=True).tolist()
 
            print(f"| Extracted pose from {inp['drv_pose_name']}", flush=True)
            eulers = convert_to_tensor(drv_pose_coeff_dict['euler']).reshape([-1, 3]).cuda()
            trans = convert_to_tensor(drv_pose_coeff_dict['trans']).reshape([-1, 3]).cuda()
            len_pose = len(eulers)
            index_lst = [mirror_index(i, len_pose) for i in range(t_x // 2)]
            sample['euler'] = eulers[index_lst]
            sample['trans'] = trans[index_lst]
 
        sample['trans'][:, -1] = sample['trans'][0:1, -1].repeat([sample['trans'].shape[0]])
 
        if inp.get("map_to_init_pose", 'True') in ['True', True]:
            diff_euler = torch.tensor(coeff_dict['euler']).reshape([1, 3]).cuda() - sample['euler'][0:1]
            sample['euler'] = sample['euler'] + diff_euler
            diff_trans = torch.tensor(coeff_dict['trans']).reshape([1, 3]).cuda() - sample['trans'][0:1]
            sample['trans'] = sample['trans'] + diff_trans
 
        camera_ret = get_eg3d_convention_camera_pose_intrinsic({
            'euler': sample['euler'].cpu(),
            'trans': sample['trans'].cpu()
        })
 
        c2w, intrinsics = camera_ret['c2w'], camera_ret['intrinsics']
        camera_smo_ksize = 7
        camera = np.concatenate([c2w.reshape([-1, 16]), intrinsics.reshape([-1, 9])], axis=-1)
        camera = smooth_camera_sequence(camera, kernel_size=camera_smo_ksize)
        camera = torch.tensor(camera).cuda().float()
        sample['camera'] = camera
 
        return sample
 
    @torch.no_grad()
    def get_hubert(self, wav16k_name):
        from data_gen.utils.process_audio.extract_hubert import get_hubert_from_16k_wav
        hubert = get_hubert_from_16k_wav(wav16k_name).detach().numpy()
        len_mel = hubert.shape[0]
        x_multiply = 8
        if len_mel % x_multiply == 0:
            num_to_pad = 0
        else:
            num_to_pad = x_multiply - len_mel % x_multiply
        hubert = np.pad(hubert, pad_width=((0, num_to_pad), (0, 0)))
        return hubert
 
    @torch.no_grad()
    def get_f0(self, wav16k_name):
        wav, sr = librosa.load(wav16k_name, sr=16000, mono=True)
        f0, voiced_flag, voiced_probs = librosa.pyin(wav, fmin=80, fmax=600, sr=sr, frame_length=2048, hop_length=320)
        if f0 is None:
            f0 = np.zeros((len(wav) // 320 + 1, 1), dtype=np.float32)
        else:
            f0 = np.nan_to_num(f0, nan=0.0).reshape([-1, 1]).astype(np.float32)
        return torch.tensor(f0)
 
    def get_mfcc(self, wav16k_name):
        from utils.audio import librosa_wav2mfcc
        hparams['fft_size'] = 1200
        hparams['win_size'] = 1200
        hparams['hop_size'] = 480
        hparams['audio_num_mel_bins'] = 80
        hparams['fmin'] = 80
        hparams['fmax'] = 12000
        hparams['audio_sample_rate'] = 24000
        mfcc = librosa_wav2mfcc(wav16k_name, fft_size=hparams['fft_size'], hop_size=hparams['hop_size'], win_length=hparams['win_size'], num_mels=hparams['audio_num_mel_bins'], fmin=hparams['fmin'], fmax=hparams['fmax'], sample_rate=hparams['audio_sample_rate'], center=True)
        mfcc = np.array(mfcc).reshape([-1, 13])
        len_mel = mfcc.shape[0]
        x_multiply = 8
        if len_mel % x_multiply == 0:
            num_to_pad = 0
        else:
            num_to_pad = x_multiply - len_mel % x_multiply
        mfcc = np.pad(mfcc, pad_width=((0, num_to_pad), (0, 0)))
        return mfcc
 
    @torch.no_grad()
    def forward_audio2secc(self, batch, inp=None):
        try:
            if inp['drv_audio_name'][-4:] in ['.wav', '.mp3']:
                ret = {}
                pred = self.audio2secc_model.forward(batch, ret=ret, train=False, temperature=inp['temperature'])
 
                if pred.shape[-1] == 144:
                    id = ret['pred'][0][:, :80]
                    exp = ret['pred'][0][:, 80:]
                else:
                    id = batch['id']
                    exp = ret['pred'][0]
                if len(id) < len(exp):
                    id = torch.cat([id, id[0].unsqueeze(0).repeat([len(exp) - len(id), 1])])
                batch['id'] = id
                batch['exp'] = exp
            else:
                drv_motion_coeff_dict = self.drv_motion_coeff_dict
                batch['exp'] = torch.FloatTensor(drv_motion_coeff_dict['exp']).cuda()
            
            batch = self.get_driving_motion(batch['id'], batch['exp'], batch['euler'], batch['trans'], batch, inp)
 
            if self.use_icl_audio2motion:
                self.audio2secc_model.empty_context()
 
            return batch
 
        except Exception:
            err = traceback.format_exc()
            print(">>> forward_audio2secc FAILED", flush=True)
            print(err, flush=True)
            with open("inference_error.log", "a") as f:
                f.write("\n===== forward_audio2secc =====\n")
                f.write(err + "\n")
            raise
 
    @torch.no_grad()
    def get_driving_motion(self, id, exp, euler, trans, batch, inp):
        device = id.device
        zero_eulers = torch.zeros([id.shape[0], 3], device=device)
        zero_trans = torch.zeros([id.shape[0], 3], device=device)
 
        with torch.no_grad():
            chunk_size = 50
            drv_secc_color_lst = []
            num_iters = len(id) // chunk_size if len(id) % chunk_size == 0 else len(id) // chunk_size + 1
            
            for i in tqdm.trange(num_iters, desc="rendering drv secc"):
                torch.cuda.empty_cache()
                
                id_chunk = id[i * chunk_size:(i + 1) * chunk_size]
                exp_chunk = exp[i * chunk_size:(i + 1) * chunk_size]
                zero_euler_chunk = zero_eulers[i * chunk_size:(i + 1) * chunk_size]
                zero_trans_chunk = zero_trans[i * chunk_size:(i + 1) * chunk_size]
 
                face_mask, drv_secc_color = self.secc_renderer(
                    id_chunk, exp_chunk, zero_euler_chunk, zero_trans_chunk
                )
                drv_secc_color_lst.append(drv_secc_color.cpu())
 
        if len(drv_secc_color_lst) == 0:
             raise RuntimeError("No frames generated")
 
        drv_secc_colors = torch.cat(drv_secc_color_lst, dim=0).cuda()
        
        _, src_secc_color = self.secc_renderer(id[0:1], exp[0:1], zero_eulers[0:1], zero_trans[0:1])
        _, cano_secc_color = self.secc_renderer(id[0:1], exp[0:1]*0, zero_eulers[0:1], zero_trans[0:1])
 
        batch['drv_secc'] = drv_secc_colors
        batch['src_secc'] = src_secc_color.cuda()
        batch['cano_secc'] = cano_secc_color.cuda()
 
        if inp['blink_mode'] == 'period':
            period = 5
            total_frames = len(batch['drv_secc'])
            for i in tqdm.trange(total_frames, desc="blinking secc"):
                if i % (25 * period) == 0:
                    blink_dur_frames = random.randint(8, 12)
                    for offset in range(blink_dur_frames):
                        j = offset + i
                        if j >= total_frames - 1: break
                        def blink_percent_fn(t, T): return -4 / T ** 2 * t ** 2 + 4 / T * t
                        blink_percent = blink_percent_fn(offset, blink_dur_frames)
                        secc = batch['drv_secc'][j]
                        out_secc = blink_eye_for_secc(secc, blink_percent).cuda()
                        batch['drv_secc'][j] = out_secc
 
        # 🔥 修复 2: 确保 drv_kp 是 3D 关键点
        drv_kp = self.face3d_helper.reconstruct_lm2d(id, exp, euler, trans)
        if drv_kp.shape[-1] == 2:
            drv_kp = torch.cat([drv_kp, torch.zeros_like(drv_kp[..., :1])], dim=-1)
        drv_kp = (drv_kp - 0.5) / 0.5
        batch['drv_kp'] = torch.clamp(drv_kp, -1, 1)
        
        return batch
 
    @torch.no_grad()
    def forward_secc2video(self, batch, inp=None):
        try:
            num_frames = len(batch['drv_secc'])
            out_fname = inp.get('out_name', '')
            if not out_fname:
                out_fname = f"infer_out/tmp/{os.path.basename(inp.get('src_image_name', 'src'))[:-4]}_{os.path.basename(inp.get('drv_pose_name', 'pose'))[:-4]}.mp4"
            
            os.makedirs(os.path.dirname(out_fname), exist_ok=True)
            temp_video_path = out_fname.replace('.mp4', '_temp.mp4')
            
            command = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-pix_fmt', 'rgb24', '-s', '512x512', '-r', '25',
                '-i', '-', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-vf', 'format=yuv420p', '-preset', 'ultrafast', '-crf', '23', temp_video_path
            ]
            pipe = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
 
            print(f"Start rendering {num_frames} frames to {out_fname}...", flush=True)
 
            for i in range(num_frames):
                cond = {
                    'cond_cano': batch['cano_secc'], 'cond_src': batch['src_secc'],
                    'cond_tgt': batch['drv_secc'][i:i+1].cuda(),
                    'ref_torso_img': batch['ref_torso_img'], 'bg_img': batch['bg_img'],
                    'segmap': batch['segmap'], 'kp_s': batch['src_kp'][i:i+1], 'kp_d': batch['drv_kp'][i:i+1]
                }
                
                gen_output = self.secc2video_model.forward(
                    img=batch['ref_head_img'], camera=batch['camera'][i:i+1], 
                    cond=cond, ret={}, cache_backbone=(i==0), use_cached_backbone=(i!=0)
                )
                
                frame_uint8 = ((gen_output['image'][0] + 1) / 2 * 255).clamp(0, 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
                pipe.stdin.write(frame_uint8.tobytes())
 
            pipe.stdin.close()
            pipe.wait()
 
            if inp.get('drv_audio_name', '').endswith(('.wav', '.mp3')):
                cmd_merge = f"ffmpeg -y -hide_banner -loglevel error -i {temp_video_path} -i {inp['drv_audio_name']} -c:v copy -c:a aac -shortest {out_fname} > /dev/null 2>&1"
                os.system(cmd_merge)
                if os.path.exists(temp_video_path): os.remove(temp_video_path)
            else:
                if temp_video_path != out_fname: os.rename(temp_video_path, out_fname)
 
            print(f"Successfully saved at {out_fname}", flush=True)
            return out_fname
 
        except Exception as e:
            print(f">>> forward_secc2video FAILED: {e}", flush=True)
            traceback.print_exc()
            raise e
 
    @torch.no_grad()
    def forward_system(self, batch, inp):
        try:
            self.forward_audio2secc(batch, inp)
            out_fname = self.forward_secc2video(batch, inp)
            return out_fname
        except Exception:
            err = traceback.format_exc()
            print(">>> forward_system FAILED", flush=True)
            print(err, flush=True)
            with open("inference_error.log", "a") as f:
                f.write("\n===== forward_system =====\n")
                f.write(err + "\n")
            raise
 
if __name__ == '__main__':
    print("Please run via app_real3dportrait.py")
