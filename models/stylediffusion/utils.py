
from typing import Optional, Union, Tuple, List, Callable, Dict
from models.stylediffusion import global_var
from models.stylediffusion import ptp_utils_v
import torch
import torch.nn.functional as nnf
import abc
from models.stylediffusion import seq_aligner
import numpy as np
from PIL import Image
import clip
from models.stylediffusion.clip_util import VisionTransformer
clip.model.VisionTransformer = VisionTransformer
import torch.nn as nn
import torchvision.transforms as transforms
import copy

USE_INITIAL_INV=global_var.get_value("USE_INITIAL_INV")
MAX_NUM_WORDS=global_var.get_value("MAX_NUM_WORDS")
device=global_var.get_value("device")
NUM_DDIM_STEPS=global_var.get_value("NUM_DDIM_STEPS")
tokenizer=global_var.get_value("tokenizer")
LOW_RESOURCE=global_var.get_value("LOW_RESOURCE")
BLOCK_NUM=global_var.get_value("BLOCK_NUM")

class LocalBlend:

    def get_mask(self, maps, alpha, use_pool):
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=(64, 64))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1 - int(use_pool)])
        mask = mask[:1] + mask
        return mask

    def __call__(self, x_t, attention_store):
        self.counter += 1
        if self.counter > self.start_blend:

            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
            maps = torch.cat(maps, dim=1)
            mask = self.get_mask(maps, self.alpha_layers, True)
            if self.substruct_layers is not None:
                maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
                mask = mask * maps_sub
            mask = mask.float()
            x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t

    def __init__(self, prompts: List[str], words: [List[List[str]]], substruct_words=None, start_blend=0.2,
                 th=(.3, .3)):
        alpha_layers = torch.zeros(len(prompts), 1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils_v.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1

        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts), 1, 1, 1, 1, MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = ptp_utils_v.get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        self.alpha_layers = alpha_layers.to(device)
        self.start_blend = int(start_blend * NUM_DDIM_STEPS)
        self.counter = 0
        self.th = th

class EmptyControl:

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn

class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    @abc.abstractmethod
    def replace_uncond(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        else:  # self-attn of unconditional branch
            attn = self.replace_uncond(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class SpatialReplace(EmptyControl):

    def step_callback(self, x_t):
        if self.cur_step < self.stop_inject:
            b = x_t.shape[0]
            x_t = x_t[:1].expand(b, *x_t.shape[1:])
        return x_t

    def __init__(self, stop_inject: float):
        super(SpatialReplace, self).__init__()
        self.stop_inject = int((1 - stop_inject) * NUM_DDIM_STEPS)

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def replace_uncond(self, attn, is_cross: bool, place_in_unet: str):
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, tau_neg=.0):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.tau_neg = tau_neg

class AttentionControlEdit(AttentionStore, abc.ABC):

    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t

    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[2] <= 32 ** 2:
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (
                            1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce, place_in_unet)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def replace_uncond(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).replace_uncond(attn, is_cross, place_in_unet)
        if not is_cross and self.num_uncond_self_replace[0] <= self.cur_step < self.num_uncond_self_replace[1]:
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            attn[1:] = self.replace_self_attention(attn_base, attn_repalce, place_in_unet)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 uncond_self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils_v.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps,
                                                                              tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        if type(uncond_self_replace_steps) is float:
            uncond_self_replace_steps = 0, uncond_self_replace_steps
        self.num_uncond_self_replace = int(num_steps * uncond_self_replace_steps[0]), int(num_steps * uncond_self_replace_steps[1])
        self.local_blend = local_blend

class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, uncond_self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, uncond_self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)

class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, uncond_self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, uncond_self_replace_steps,
                                              local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])

class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, uncond_self_replace_steps: float, equalizer,
                 local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, uncond_self_replace_steps,
                                                local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller

def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                                                                                     Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)

    for word, val in zip(word_select, values):
        inds = ptp_utils_v.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer

def aggregate_attention(attention_store: AttentionStore, prompts: List[str], res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()

def make_controller(prompts: List[str], is_replace_controller: bool, cross_replace_steps: Dict[str, float],
                    self_replace_steps: float, uncond_self_replace_steps: float, blend_words=None, equilizer_params=None) -> AttentionControlEdit:
    if blend_words is None:
        lb = None
    else:
        lb = LocalBlend(prompts, blend_words)
    if is_replace_controller:
        controller = AttentionReplace(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps,
                                      self_replace_steps=self_replace_steps, uncond_self_replace_steps=uncond_self_replace_steps, local_blend=lb)
    else:
        controller = AttentionRefine(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps,
                                     self_replace_steps=self_replace_steps, uncond_self_replace_steps=uncond_self_replace_steps, local_blend=lb)
    if equilizer_params is not None:
        eq = get_equalizer(prompts[1], equilizer_params["words"], equilizer_params["values"])
        controller = AttentionReweight(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps,
                                       self_replace_steps=self_replace_steps, uncond_self_replace_steps=uncond_self_replace_steps,
                                       equalizer=eq, local_blend=lb, controller=controller)
    return controller

def show_cross_attention(attention_store: AttentionStore, prompts: List[str], res: int, from_where: List[str], select: int = 0, save_name='cross-attn-map'):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, prompts, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils_v.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    ptp_utils_v.view_images(np.stack(images, axis=0), save_name=save_name)

def show_hot_cross_attention(attention_store: AttentionStore, prompts: List[str], res: int, from_where: List[str], select: int = 0, save_name='cross-attn-map'):
    import cv2
    choice = 4
    colormap_dict = {
        1: cv2.COLORMAP_VIRIDIS,
        2: cv2.COLORMAP_PLASMA,
        3: cv2.COLORMAP_HOT,
        4: cv2.COLORMAP_JET,
        5: cv2.COLORMAP_INFERNO,
        6: cv2.COLORMAP_AUTUMN,
        7: cv2.COLORMAP_BONE,
        8: cv2.COLORMAP_WINTER,
        9: cv2.COLORMAP_RAINBOW,
        10: cv2.COLORMAP_OCEAN,
        11: cv2.COLORMAP_SUMMER,
        12: cv2.COLORMAP_SPRING,
        13: cv2.COLORMAP_COOL,
        14: cv2.COLORMAP_HSV,
        15: cv2.COLORMAP_PINK,
    }

    def gray_to_heatmap(gray_image, colormap):
        colored_image = cv2.applyColorMap(gray_image, colormap)
        return colored_image
    if choice not in colormap_dict:
        print("Invalid choice. Using the default colormap (viridis).")
        choice = 1
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, prompts, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        colored_image = gray_to_heatmap(image[:,:,0], colormap_dict[choice])
        cv2.imwrite(f'{save_name}-{i}.png', colored_image)

def show_self_attention_comp(attention_store: AttentionStore, prompts: List[str], res: int, from_where: List[str],
                             max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, prompts, res, from_where, False, select).numpy().reshape(
        (res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils_v.view_images(np.concatenate(images, axis=1), save_name='self-attn-map-comp')

def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h - bottom, left:w - right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image

def register_attention_control(model, trainer, controller):
    IS_TRAIN=global_var.get_value("IS_TRAIN")
    assert IS_TRAIN is not None, print("must set True or False for args.is_train.")
    assert controller is None if IS_TRAIN else (trainer and controller)
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]  # todo: ?
        else:
            to_out = self.to_out

        def forward(x, context=None, mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            # image encoded to embedding for to_v() in cross-attn of conditional branch.
            IS_TRAIN=global_var.get_value("IS_TRAIN")
            if IS_TRAIN:  # training phase
                '''
                skip when trainer.ddim_inv is True which means to store ground truth attn maps,
                these attn maps are used as supervision during the training phase
                '''
                if (not trainer.uncond and is_cross) and (not trainer.ddim_inv):
                    context = trainer.forward_embed(context)
            else:  # editing phase
                if not controller.uncond and is_cross:
                    if USE_INITIAL_INV:
                        context = trainer.forward_embed(context)
                    else:
                        i = trainer.i
                        cont = list(context.chunk(context.shape[0]))
                        for b in range(len(cont)):
                            trainer.i = trainer.I if b == 0 else i
                            cont[b] = trainer.forward_embed(cont[b])
                        context = cont[0] if len(cont) == 1 else torch.cat(cont)
                        trainer.i = i
            v = self.to_v(context)
            q = self.reshape_heads_to_batch_dim(q)
            k = self.reshape_heads_to_batch_dim(k)
            v = self.reshape_heads_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            IS_TRAIN=global_var.get_value("IS_TRAIN")
            if IS_TRAIN:  # training phase
                attn = trainer(attn, is_cross, place_in_unet)
            else:  # editing phase
                attn = controller(attn, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.reshape_batch_dim_to_heads(out)
            return to_out(out)

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0
            
    IS_TRAIN=global_var.get_value("IS_TRAIN")
    if IS_TRAIN:
        if trainer is None:
            trainer = DummyController()
    else:
        if controller is None:
            controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    IS_TRAIN=global_var.get_value("IS_TRAIN")
    if IS_TRAIN:
        trainer.num_att_layers = cross_att_count
    else:
        controller.num_att_layers = cross_att_count

def image_grid(img, grid_size):
    gw, gh = grid_size
    _N, H, W, C = img.shape
    img = img.reshape(gh, gw, H, W, C)
    img = img.transpose(0, 2, 1, 3, 4)
    img = img.reshape(gh * H, gw * W, C)
    return img

class Trainer(AttentionStore):
    def __init__(self):
        super(Trainer, self).__init__()
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        # clip image encoder
        self.clip_model, clip_preprocess = clip.load('ViT-B/16', device=self.device)
        self.clip_preprocess = clip_preprocess
        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] +  # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                             clip_preprocess.transforms[:2] +                                        # to match CLIP input scale assumptions
                                             clip_preprocess.transforms[4:])                                         # + skip convert PIL to tensor

        self.image = None
        self.embedding = []
        # image embedding
        scale = 2
        self.embedding = []
        self.convblock = nn.Sequential(nn.Conv1d(77 * scale, 77 * scale, kernel_size=1),
                                       nn.BatchNorm1d(77 * scale, affine=True),
                                       nn.LeakyReLU())
        for _ in range(NUM_DDIM_STEPS):
            self.embedding.append(nn.ModuleDict({
                'conv_start': nn.Conv1d(197, 77 * scale, kernel_size=1),  # (bs, 197, 768)->(bs, 77, 768)
                'conv_block': nn.Sequential(*[copy.deepcopy(self.convblock) for _ in range(BLOCK_NUM)]),
                'conv_end': nn.Conv1d(77 * scale, 77 * scale, kernel_size=1),  # (bs, 77, 768)->(bs, 77, 768)
            }).train().requires_grad_(False).to(device))

        self.I = None  # only for eval
        self.i = None
        self.uncond = False
        self.ddim_inv = False
        self.v_replace_steps = .5

    def load_pretrained(self, pretrained_embedding):
        for i, pre_embedding in enumerate(pretrained_embedding):
            for pre_emb, emb in zip(pre_embedding.values(), self.embedding[i].values()):
                self.copy_params_and_buffers(pre_emb, emb)

    def named_params_and_buffers(self, module):
        assert isinstance(module, torch.nn.Module)
        return list(module.named_parameters()) + list(module.named_buffers())

    def copy_params_and_buffers(self, src_vae, dst_vae, require_all=False):
        assert isinstance(src_vae, torch.nn.Module)
        assert isinstance(dst_vae, torch.nn.Module)
        vae_tensors = dict(self.named_params_and_buffers(src_vae))
        for name, tensor in self.named_params_and_buffers(dst_vae):
            assert (name in vae_tensors) or not require_all
            if name in vae_tensors and tensor.shape == vae_tensors[name].shape:
                try:
                    tensor.copy_(vae_tensors[name].detach()).requires_grad_(tensor.requires_grad)
                except Exception as e:
                    print(f'Error loading: {name} {vae_tensors[name].shape} {tensor.shape}')
                    raise e
            # else:
            #     print(f'{name}: {tensor.shape}, {vae_tensors[name].shape}')

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).to(self.device)
        return self.clip_model.encode_image(images)

    def forward_embed(self, context):
        if self.i is not None:
            img_emb = self.encode_images(self.image).to(torch.float32)
            for block in self.embedding[self.i].values():
                img_emb = block(img_emb)
        return (context * img_emb[:, :77, :] + img_emb[:, 77:, :]) if self.i is not None else context

    # def forward_embed(self, context):
    #     if self.i is not None:
    #         context = self.encode_images(self.image).to(torch.float32)
    #         for block in self.embedding[self.i].values():
    #             context = block(context)
    #     return context
