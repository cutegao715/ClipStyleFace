'''
geo and texture
complete one
'''

import os
import wandb

import torch
import numpy as np
import clip
import typing as tp

from pathlib import Path

from core.utils.text_templates import neg_templates, pos_templates
from core.utils.train_log import StreamingMeans, TimeLog, Timer
from core.utils.loggers import LoggingManager
from core.utils.class_registry import ClassRegistry
from core.utils.common import (
    mixing_noise, validate_device, compose_text_with_templates, load_clip,
    read_domain_list, read_style_images_list, determine_opt_layers,
    get_stylegan_conv_dimensions, DataParallelPassthrough, get_trainable_model_state
)

from core.loss import DirectLoss

from core.utils.image_utils import construct_paper_image_grid
from core.utils.mesh_utils import load_laplacian, JacobianSmoothness, calculate_deformation_gradient
from core.parametrizations import BaseParametrization
from core.uda_models import uda_models

import torchvision.transforms as transforms
import torchvision

trainer_registry = ClassRegistry()

from decalib.deca2 import DECA
from decalib.utils import lossfunc
from decalib.utils import util
from decalib.datasets import datasets
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.renderer import SRenderY, set_rasterizer
from pytorch_msssim import ssim

from deformnet.surfacenet_nohyper import SurfaceDeformationField
from utils_3d import batch_orth_proj, random_cam

from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes


class BaseDomainAdaptationTrainer:
    def __init__(self, config):
        # common
        self.config = config
        self.trainable = None
        self.source_generator = None
        self.deformnet = None
        self.neg_img = None

        self.current_step = 0
        self.optimizer = None
        self.loss_function = None
        self.batch_generators = None
        self.id_loss_fn = None
        self.elastic_loss_fn = None

        self.zs_for_logging = None

        self.reference_embeddings = {}

        # processed in multiple_domain trainer
        self.domain_embeddings = None
        self.desc_to_embeddings = None

        self.global_metrics = {}

    def _setup_base(self):
        self._setup_device()
        self._setup_batch_generators()
        self._setup_source_generator()
        self._setup_loss()

    def _setup_device(self):
        chosen_device = self.config.training["device"].lower()
        device = validate_device(chosen_device)
        self.device = torch.device(device)

    def _setup_template(self):
        verts, faces, _ = load_obj('10_template.obj')
        faces = faces.verts_idx[None, ...][0]
        self.template = load_laplacian(verts, faces)
        verts = (verts - verts.mean(0, keepdim=True)) / verts.std(0, keepdim=True)
        self.faces = faces.unsqueeze(0)
        mesh = Meshes(verts=verts.unsqueeze(0), faces=faces.unsqueeze(0))
        self.vertex_normals = mesh.verts_normals_padded().to(self.device)

    def _setup_source_generator(self):
        self.source_generator = uda_models[self.config.training.generator](
            **self.config.generator_args[self.config.training.generator]
        )
        # self.source_generator.patch_layers(self.config.training.patch_key)
        self.source_generator.freeze_layers()
        self.source_generator.to(self.device)

        self.albedo_generator = uda_models[self.config.training.generator](
            **self.config.albedo_args[self.config.training.generator]
        )
        self.albedo_generator.patch_layers(self.config.training.patch_key)
        self.albedo_generator.freeze_layers()
        self.albedo_generator.to(self.device)

    def _setup_loss(self):
        self.loss_function = DirectLoss(self.config.optimization_setup)
        self.id_loss_fn = lossfunc.VGGFace2Loss(pretrained_model=deca_cfg.model.fr_model_path)
        self.elastic_loss_fn = JacobianSmoothness()

    def _setup_logger(self):
        self.logger = LoggingManager(self.config)

    def _setup_batch_generators(self):
        self.batch_generators = {}

        for visual_encoder in self.config.optimization_setup.visual_encoders:
            self.batch_generators[visual_encoder] = (
                load_clip(visual_encoder, device=self.config.training.device)
            )

        self.reference_embeddings = {k: {} for k in self.batch_generators}

    @torch.no_grad()
    def _initial_logging(self):
        self.zs_for_logging = [
            mixing_noise(16, 512, 0, self.config.training.device)
            for _ in range(self.config.logging.num_grid_outputs)
        ]

        for idx, z in enumerate(self.zs_for_logging):
            images,_ = self.forward_source(z)
            self.logger.log_images(0, {f"src_domain_grids/{idx}": construct_paper_image_grid(images)})

    def _setup_optimizer(self):
        if self.config.training.patch_key == "original":
            g_reg_every = self.config.optimization_setup.g_reg_every
            lr = self.config.optimization_setup.optimizer.lr

            g_reg_ratio = g_reg_every / (g_reg_every + 1)
            betas = self.config.optimization_setup.optimizer.betas

            self.optimizer = torch.optim.Adam(
                self.trainable.parameters(),
                lr=lr * g_reg_ratio,
                betas=(betas[0] ** g_reg_ratio, betas[1] ** g_reg_ratio),
            )
        else:
            self.optimizer = torch.optim.Adam(
                [{'params': self.trainable.parameters(), 'lr':0.01},
                 {'params': self.deformnet.parameters(), 'lr':0.0005}],
                weight_decay=0.0, betas=(0.9, 0.999)
            )

    # @classmethod
    # def from_ckpt(cls, ckpt_path):
    #     m = cls(ckpt['config'])
    #     m._setup_base()
    #     return m

    def start_from_checkpoint(self):
        step = 0
        if self.config.checkpointing.start_from:
            state_dict = torch.load(self.config.checkpointing.start_from, map_location='cpu')
            step = state_dict['step']
            self.trainable.load_state_dict(state_dict['trainable'])
            self.optimizer.load_state_dict(state_dict['trainable_optimizer'])
            print('starting from step {}'.format(step))
        # TODO: python main.py --ckpt_path ./.... -> Trainer.from_ckpt()
        return step

    def get_checkpoint(self):
        state_dict = {
            "step": self.current_step,
            "trainable": self.trainable.state_dict(),
            "trainable_optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }
        return state_dict

    # TODO: refactor checkpoint
    def make_checkpoint(self):
        if not self.config.checkpointing.is_on:
            return

        ckpt = self.get_checkpoint()
        torch.save(ckpt, os.path.join(self.logger.checkpoint_dir, "checkpoint.pt"))

    def save_models(self):
        models_dict = get_trainable_model_state(
            self.config, self.trainable.state_dict()
        )

        models_dict.update(self.ckpt_info())
        torch.save(models_dict, str(
            Path(self.logger.models_dir) / f"models_{self.current_step}.pt"
        ))

    def ckpt_info(self):
        return {}

    def all_to_device(self, device):
        self.source_generator.to(device)
        self.trainable.to(device)
        self.loss_function.to(device)
        self.albedo_generator.to(device)
        self.deformnet.to(device)
        self.id_loss_fn.to(device)
        self.elastic_loss_fn.to(device)

    def train_loop(self):
        self.all_to_device(self.device)

        recovered_step = self.start_from_checkpoint()
        iter_info = StreamingMeans()

        for self.current_step in range(recovered_step, self.config.training.iter_num + 1, 1):
            for _ in range(10):
                with Timer(iter_info, "train_iter"):
                    self.train_step(iter_info)

        wandb.finish()

    @torch.no_grad()
    def encode_text(self, model, text, templates):
        text = compose_text_with_templates(text, templates=templates)
        tokens = clip.tokenize(text).to(self.config.training.device)
        text_features = model.encode_text(tokens).detach()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def clip_encode_image(self, model, image, preprocess):
        image_features = model.encode_image(preprocess(image))
        image_features /= image_features.clone().norm(dim=-1, keepdim=True)
        return image_features

    def partial_trainable_model_freeze(self):
        if not hasattr(self.config.training, 'auto_layer_iters'):
            return

        if self.config.training.auto_layer_iters == 0:
            return

        train_layers = determine_opt_layers(
            self.source_generator,
            self.trainable,
            self.batch_generators['ViT-B/32'][0],
            self.config,
            self.config.training.target_class,
            self.config.training.auto_layer_iters,
            self.config.training.auto_layer_batch,
            self.config.training.auto_layer_k,
            device=self.device,
        )

        if not isinstance(train_layers, list):
            train_layers = [train_layers]

        self.trainable.freeze_layers()
        self.trainable.unfreeze_layers(train_layers)

    def train_step(self, iter_info):
        self.trainable.train()
        self.deformnet.train()
        sample_z = mixing_noise(
            self.config.training.batch_size,
            512,
            self.config.training.mixing_noise,
            self.config.training.device,
        )

        # self.partial_trainable_model_freeze()

        batch = self.calc_batch(sample_z)
        losses = self.loss_function(batch)

        iter_info.update({f"losses/{k}": v for k, v in losses.items()})

    def forward_trainable(self, latents, *args, **kwargs):
        raise NotImplementedError()

    @torch.no_grad()
    def forward_source(self, latents, **kwargs) -> torch.Tensor:
        sampled_images, latents = self.source_generator(latents, **kwargs)
        return sampled_images.detach(), latents.detach()

    def calc_batch(self, sample_z):
        raise NotImplementedError()

    @torch.no_grad()
    def log_images(self):
        raise NotImplementedError()

    def to_multi_gpu(self):
        self.source_generator = DataParallelPassthrough(self.source_generator, device_ids=self.config.exp.device_ids)
        self.trainable = DataParallelPassthrough(self.trainable, device_ids=self.config.exp.device_ids)

    def invert_image_ii2s(self, image_info, ii2s):
        image_full_res = image_info['image_high_res_torch'].unsqueeze(0).to(self.device)
        image_resized = image_info['image_low_res_torch'].unsqueeze(0).to(self.device)

        lam = str(int(ii2s.opts.p_norm_lambda * 1000))
        name = Path(image_info['image_name']).stem + f"_{lam}.npy"
        current_latents_path = self.logger.cached_latents_local_path / name

        if current_latents_path.exists():
            latents = np.load(str(current_latents_path))
            latents = torch.from_numpy(latents).to(self.config.training.device)
        else:
            latents, = ii2s.invert_image(
                image_full_res,
                image_resized
            )

            print(f'''
            latents for {image_info['image_name']} cached in 
            {str(current_latents_path.resolve())}
            ''')

            np.save(str(current_latents_path), latents.detach().cpu().numpy())

        return latents


class SingleDomainAdaptationTrainer(BaseDomainAdaptationTrainer):
    def __init__(self, config):
        super().__init__(config)

    def _setup_trainable(self):
        if self.config.training.patch_key == 'original':
            self.trainable = uda_models[self.config.training.generator](
                **self.config.generator_args[self.config.training.generator]
            )
            trainable_layers = list(self.trainable.get_training_layers(
                phase=self.config.training.phase
            ))
            self.trainable.freeze_layers()
            self.trainable.unfreeze_layers(trainable_layers)
        else:
            self.trainable = BaseParametrization(
                self.config.training.patch_key,
                get_stylegan_conv_dimensions(self.source_generator.generator.size),
            )

        self.trainable.to(self.device)

    def forward_trainable(self, latents, **kwargs) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        if self.config.training.patch_key == "original":
            sampled_images, _ = self.trainable(
                latents, **kwargs
            )
            offsets = None
        else:
            offsets = self.trainable()
            sampled_images, _ = self.albedo_generator(
                latents, offsets=offsets, **kwargs
            )

        return sampled_images, offsets

    @torch.no_grad()
    def log_images(self):
        self.trainable.eval()
        dict_to_log = {}

        for idx, z in enumerate(self.zs_for_logging):
            sampled_images, _ = self.forward_trainable(z, truncation=self.config.logging.truncation)
            images = construct_paper_image_grid(sampled_images)
            dict_to_log.update({
                f"trg_domain_grids/{self.config.training.target_class}/{idx}": images
            })

        self.logger.log_images(self.current_step, dict_to_log)


@trainer_registry.add_to_registry("td_single")
class TextDrivenSingleDomainAdaptationTrainer(SingleDomainAdaptationTrainer):
    def __init__(self, config):
        super().__init__(config)

    def ckpt_info(self):
        return {
            'da_type': 'td',
        }

    def setup(self):
        self._setup_base()
        self._setup_deca()
        self._setup_deformnet()
        self._setup_trainable()
        self._setup_optimizer()
        self._setup_text_embeddings()
        self._setup_dataset()
        self._setup_traindata()

    def _setup_dataset(self):
        iscrop = True
        detector = 'fan'
        sample_step = 10
        self.sample_images = datasets.TestData_stylegan(self.source_generator,
                                                   batch_size=1,
                                                   iscrop=iscrop,
                                                   face_detector=detector,
                                                   sample_step=sample_step,
                                                   crop_size=1024
                                                   )
        clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                               (0.26862954, 0.26130258, 0.27577711))
        # CLIP Transform
        res = 224

        # Augmentation settings
        augment_transform = transforms.Compose([
            transforms.RandomResizedCrop(res, scale=(1.0, 1.0)),
            transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
            clip_normalizer
        ])
        self.augment_transform = augment_transform

        normaugment_transform = transforms.Compose([
            transforms.RandomResizedCrop(res, scale=(0.8, 0.8)),
            transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
            clip_normalizer
        ])
        self.normaugment_transform = normaugment_transform

    def _setup_deformnet(self):
        self.deformnet = SurfaceDeformationField(self.config.training.batch_size)
        self.deformnet = self.deformnet.to(self.device)

    def _setup_deca(self):
        self.deca = DECA(config=deca_cfg, device=self.device)
        set_rasterizer(self.config.render.rasterizer_type)
        self.render = SRenderY(self.config.render.img_size, self.config.render.dense_topology_path,
                          self.config.render.img_size, self.config.render.rasterizer_type).to(self.device)

    def _setup_text_embeddings(self):
        for visual_encoder, (model, preprocess) in self.batch_generators.items():
            self.reference_embeddings[visual_encoder][self.config.training.neg_class] = self.encode_text(
                model, self.config.training.neg_class, neg_templates
            )
            self.reference_embeddings[visual_encoder][self.config.training.source_class] = self.encode_text(
                model, self.config.training.source_class, pos_templates
            )
            self.reference_embeddings[visual_encoder][self.config.training.target_class] = self.encode_text(
                model, self.config.training.target_class, pos_templates
            )
            self.reference_embeddings[visual_encoder][self.config.training.pos_class] = self.encode_text(
                model, self.config.training.pos_class, pos_templates
            )

    @torch.no_grad()
    def _setup_traindata(self):
        sample_z = mixing_noise(
            self.config.training.batch_size,
            512,
            self.config.training.mixing_noise,
            self.config.training.device,
        )
        frozen_img, latents = self.forward_source(sample_z, truncation=0.5,
                                                  truncation_latent=self.source_generator.mean_latent)
        frozen_img = (frozen_img + 1) / 2
        self.frozen_img = frozen_img
        self.latents = latents
        data_list = []
        for i in range(frozen_img.shape[0]):
            crop_img = self.sample_images.__getitem__(frozen_img[i] * 255.)
            crop_img = crop_img['image'].to(self.device)[None, ...]
            data_list.append(crop_img)
        data_list = torch.stack(data_list, dim=0).squeeze().unsqueeze(0)
        codedict = self.deca.encode(torchvision.transforms.Resize(224)(data_list))
        codedict['image'] = data_list
        name = str(1).zfill(6)
        opdict, visdict = self.deca.decode(codedict, name)
        verts = opdict['verts']
        dense_verts = []
        for idx in range(verts.shape[0]):
            vertices = opdict['verts'][idx].cpu().numpy()
            faces = self.render.faces[0].cpu().numpy()
            texture = util.tensor2image(opdict['uv_texture_gt'][idx])
            normals = opdict['normals'][idx].cpu().numpy()
            displacement_map = opdict['displacement_map'][idx].cpu().numpy().squeeze()
            dense_vert, _, _ = util.upsample_mesh(vertices, normals, faces, displacement_map, texture,
                                                  self.deca.dense_template)
            dense_verts.append(torch.from_numpy(dense_vert).float())
        dense_verts = torch.stack(dense_verts, dim=0).to(self.device)
        trans_verts = batch_orth_proj(dense_verts, codedict['cam'])
        self.verts = dense_verts

    @torch.no_grad()
    def calc_verts(self, sample_z):
        frozen_img, latents = self.forward_source(sample_z, truncation=0.5,
                                                  truncation_latent=self.source_generator.mean_latent)
        frozen_img = (frozen_img + 1) / 2
        data_list = []
        for i in range(frozen_img.shape[0]):
            crop_img = self.sample_images.__getitem__(frozen_img[i] * 255.)
            crop_img = crop_img['image'].to(self.device)[None, ...]
            data_list.append(crop_img)
        data_list = torch.stack(data_list, dim=0).squeeze()

        codedict = self.deca.encode(torchvision.transforms.Resize(224)(data_list))
        codedict['image'] = data_list
        name = str(1).zfill(6)
        opdict, visdict = self.deca.decode(codedict, name)
        verts = opdict['verts']
        dense_verts = []
        for idx in range(verts.shape[0]):
            vertices = opdict['verts'][idx].cpu().numpy()
            faces = self.render.faces[0].cpu().numpy()
            texture = util.tensor2image(opdict['uv_texture_gt'][idx])
            normals = opdict['normals'][idx].cpu().numpy()
            displacement_map = opdict['displacement_map'][idx].cpu().numpy().squeeze()
            dense_vert, _, _ = util.upsample_mesh(vertices, normals, faces, displacement_map, texture,
                                                  self.deca.dense_template)
            dense_verts.append(torch.from_numpy(dense_vert).float())
        dense_verts = torch.stack(dense_verts, dim=0).to(self.device)
        return dense_verts, frozen_img, latents

    def calc_batch(self, sample_z):
        background = torch.ones(1, 3, 512, 512).to(self.device)
        clip_data = {
            k: {} for k in self.batch_generators
        }
        latents = self.latents
        verts = self.verts
        frozen_img = self.frozen_img
        trainable_img, offsets = self.forward_trainable([latents], input_is_latent=True)
        le, re = trainable_img[:,:,:,-512:], trainable_img[:,:,:,:512].detach()
        simm_loss = ssim(le, re)
        _trainable_img = trainable_img.repeat(8, 1, 1, 1)

        verts = (verts - verts.mean(1, keepdim=True)) / verts.std(1, keepdim=True)

        verts.requires_grad = True
        displacement = self.deformnet(verts)
        grad_style = calculate_deformation_gradient(verts, displacement)
        up_verts = verts + displacement

        cam = torch.tensor([0.5, 0.0,  0.0]).to(self.device)
        up_trans_verts = random_cam(up_verts[0], 8)
        up_trans_verts = batch_orth_proj(up_trans_verts, cam)
        up_trans_verts[:, :, 1:] = - up_trans_verts[:, :, 1:]
        _up_verts = up_verts.repeat(8, 1, 1)

        ops = self.render(_up_verts, up_trans_verts, _trainable_img, h=self.config.render.img_size,
                                   w=self.config.render.img_size, background=background)
        rendered_img = ops['images']

        if self.current_step == 30:
            self.neg_img = rendered_img.clone().detach()
        for visual_encoder_key, (model, preprocess) in self.batch_generators.items():
            trg_encoded = self.clip_encode_image(model, rendered_img, preprocess)
            src_encoded = self.clip_encode_image(model, frozen_img, preprocess)

            clip_data[visual_encoder_key].update({
                'trg_encoded': trg_encoded,
                'src_encoded': src_encoded,
                'trg_domain_emb': (
                    self.reference_embeddings[visual_encoder_key][self.config.training.target_class].unsqueeze(0)
                ),
                'src_domain_emb': (
                    self.reference_embeddings[visual_encoder_key][self.config.training.source_class].unsqueeze(0)
                )
            })
        batch = {
            'clip_data': clip_data,
            'rec_data': None,
            'offsets': offsets
        }
        losses = self.loss_function(batch)

        neg_losses = 0.0
        if self.current_step > 30:
            for visual_encoder_key, (model, preprocess) in self.batch_generators.items():
                trg_encoded = self.clip_encode_image(model, rendered_img, preprocess)
                src_encoded = self.clip_encode_image(model, self.neg_img, preprocess)

                clip_data[visual_encoder_key].update({
                    'trg_encoded': trg_encoded,
                    'src_encoded': src_encoded,
                    'trg_domain_emb': (
                        self.reference_embeddings[visual_encoder_key][self.config.training.pos_class].unsqueeze(0)
                    ),
                    'src_domain_emb': (
                        self.reference_embeddings[visual_encoder_key][self.config.training.neg_class].unsqueeze(0)
                    )
                })
            batch = {
                'clip_data': clip_data,
                'rec_data': None,
                'offsets': offsets
            }
            neg_losses += self.loss_function(batch)["total"]

        self.optimizer.zero_grad()
        for param in self.deformnet.parameters():
            param.requires_grad = False
        id_losses = self.id_loss_fn(rendered_img, frozen_img)
        tex_losses = losses["total"] + 0.1 * id_losses + 0.1 * simm_loss + 0.1 * neg_losses
        tex_losses.backward(retain_graph=True)

        for param in self.deformnet.parameters():
            param.requires_grad = True
        self.trainable.freeze_layers()
        elastic_loss = self.elastic_loss_fn(grad_style)
        geo_losses = losses["total"] + 0.0001 * elastic_loss + 0.1 * neg_losses
        geo_losses.backward(retain_graph=True)
        self.trainable.unfreeze_layers()
        self.optimizer.step()

        return {
            'clip_data': clip_data,
            'rec_data': None,
            'offsets': offsets
        }