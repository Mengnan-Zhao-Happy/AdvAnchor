from omegaconf import OmegaConf
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from ldm.util import instantiate_from_config
import argparse
from convertModels import savemodelDiffusers
# Util Functions
def load_model_from_config(config, ckpt, device="cpu", verbose=False):
    """Loads a model from config and a ckpt
    if config is a path will use omegaconf to load
    """
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    pl_sd = torch.load(ckpt, map_location="cpu")
    global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model


    

@torch.no_grad()
def sample_model(model, sampler, c, h, w, ddim_steps, scale, ddim_eta, start_code=None, n_samples=1,t_start=-1,log_every_t=None,till_T=None,verbose=True):
    """Sample the model"""
    uc = None
    if scale != 1.0:
        uc = model.get_learned_conditioning(n_samples * [""])
    log_t = 100
    if log_every_t is not None:
        log_t = log_every_t
    shape = [4, h // 8, w // 8]
    samples_ddim, inters = sampler.sample(S=ddim_steps,
                                     conditioning=c,
                                     batch_size=n_samples,
                                     shape=shape,
                                     verbose=False,
                                     x_T=start_code,
                                     unconditional_guidance_scale=scale,
                                     unconditional_conditioning=uc,
                                     eta=ddim_eta,
                                     verbose_iter = verbose,
                                     t_start=t_start,
                                     log_every_t = log_t,
                                     till_T = till_T
                                    )
    if log_every_t is not None:
        return samples_ddim, inters
    return samples_ddim

def load_img(path, target_size=512):
    """Load an image, resize and output -1..1"""
    image = Image.open(path).convert("RGB")


    tform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ])
    image = tform(image)
    return 2.*image - 1.


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_loss(losses, path,word, n=100):
    v = moving_average(losses, n)
    plt.plot(v, label=f'{word}_loss')
    plt.legend(loc="upper left")
    plt.title('Average loss in trainings', fontsize=20)
    plt.xlabel('Data point', fontsize=16)
    plt.ylabel('Loss value', fontsize=16)
    plt.savefig(path)

##################### ESD Functions
def get_models(config_path, ckpt_path, devices):
    model_orig = load_model_from_config(config_path, ckpt_path, devices[1])

    model = load_model_from_config(config_path, ckpt_path, devices[0])

    return model_orig, model

def preprocess(img):
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    img = img.to(memory_format=torch.contiguous_format).float()
    return img

def train_esd(iter_break, w_iter_break, atts, att_size,ori_flag, erase_cat,erased_index, lr,lr2, config_path, ckpt_path, diffusers_config_path, devices,ddim_steps=50):
    '''
    Function to train diffusion models to erase concepts from model weights

    Parameters
    ----------
    prompt : str
        The concept to erase from diffusion model (Eg: "Van Gogh").
    train_method : str
        The parameters to train for erasure (ESD-x, ESD-u, full, selfattn).
    start_guidance : float
        Guidance to generate images for training.
    negative_guidance : float
        Guidance to erase the concepts from diffusion model.
    iterations : int
        Number of iterations to train.
    lr : float
        learning rate for fine tuning.
    config_path : str
        config path for compvis diffusion format.
    ckpt_path : str
        checkpoint path for pre-trained compvis diffusion weights.
    diffusers_config_path : str
        Config path for diffusers unet in json format.
    devices : str
        2 devices used to load the models (Eg: '0,1' will load in cuda:0 and cuda:1).
    seperator : str, optional
        If the prompt has commas can use this to seperate the prompt for individual simulataneous erasures. The default is None.
    image_size : int, optional
        Image size for generated images. The default is 512.
    ddim_steps : int, optional
        Number of diffusion time steps. The default is 50.

    Returns
    -------
    None

    '''
    # PROMPT CLEANING
    model_orig, model = get_models(config_path, ckpt_path, devices)
    if erase_cat == 'object':
        prompts = ['chain saw','church','gas pump','tench','garbage truck','english springer','golf ball','parachute','french horn']
    elif erase_cat == 'style':
        prompts = ['Cezanne', 'Van Gogh', 'Picasso', 'Jackson Pollock', 'Caravaggio', 'Keith Haring', 'Kelly McKernan', 'Tyler Edlin', 'Kilian Eng']
    elif erase_cat == 'nudity':
        prompts = ['nudity']
    elif erase_cat == 'abl_object':
        prompts = ['plane', 'church']
    else:
        print('Waiting for research ...')
        print(dsd)
    prompt = prompts[erased_index]
    path_vangogh = f'./data/{prompt}/'
    images_vangogh = os.listdir(path_vangogh)
    model.train()
    parameters = []
    for name, param in model.model.diffusion_model.named_parameters():
        # train all layers except x-attns and time_embed layers
        # if 'attn2.to_k' in name or 'attn2.to_v' in name:  
        if 'attn2' in name:
        # if 'attn2.to_k' in name or 'attn2.to_v' in name:  
            parameters.append(param)
    parameters_gt = []
    for name, param in model_orig.model.diffusion_model.named_parameters():
        # train all layers except x-attns and time_embed layers
        # if 'attn2.to_k' in name or 'attn2.to_v' in name:
        if 'attn2' in name:  
        # if 'attn2.to_k' in name or 'attn2.to_v' in name:  
            parameters_gt.append(param)
    opt = torch.optim.Adam(parameters, lr=lr)
    criteria = torch.nn.MSELoss()

    name = f'Ladv1ablation_op3-{erase_cat}-lr2_{lr2}-erased_{prompt}-iter_{iter_break}_{w_iter_break}-{ori_flag}-{att_size}-1'
    # python eval-scripts/generate-images.py --model_name='adv_anchor-nudity-lr2_0.0001-erased_nudity-iter_50_20-True-0-3' --prompts_path './data/NSFW_NudeClasses_50.csv' --save_path '/abmodels/evaluation-folder' --num_samples 1 --device 'cuda:4'
    ind_start = 1# 5
    ind_end = ind_start + len(prompt.split(' '))
    delta_prompt = torch.randn(1,ind_end - ind_start,768).to(devices[0]) * 1e-7
    for _ in range(w_iter_break):
        for i, image_name_vangogh in enumerate(images_vangogh):
            image_path_vangogh = path_vangogh + image_name_vangogh
            img_vangogh = load_img(image_path_vangogh)
            img_vangogh = preprocess(img_vangogh)
            img_vangogh = img_vangogh.to(model.device)

            encoder_posterior_vangogh = model.encode_first_stage(img_vangogh)
            z_vangogh = model.get_first_stage_encoding(encoder_posterior_vangogh).detach()

            delta_prompt = Variable(delta_prompt.detach(), requires_grad=True)
            opt2 = torch.optim.Adam([delta_prompt], lr=lr2)
            if ori_flag:
                cu_len = 1
            else:
                cu_len = 0
            emb_att_set = []
            for cu_number in range(att_size):
                emb_att_set.append(model_orig.get_learned_conditioning([f"{atts[cu_number]}'s painting"]))

            emb_nor = model_orig.get_learned_conditioning([f"{prompt} painting"])
            emb_blank = model_orig.get_learned_conditioning(['painting'])

            t_enc = torch.randint(ddim_steps, (1,), device=devices[0])
            og_num = round((int(t_enc)/ddim_steps)*1000)
            og_num_lim = round((int(t_enc+1)/ddim_steps)*1000)
            t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])
            z_n = model.q_sample(z_vangogh, t_enc_ddpm)
            
            emb_noise = model_orig.get_learned_conditioning([f"{prompt}'s painting"], delta_prompt.to(devices[1]), ind_start, ind_end)
            e_blank = model_orig.apply_model(z_n.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_blank.to(devices[1]))[0]
            e_noise_op = model_orig.apply_model(z_n.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_noise.to(devices[1]))[0]
            e_noise_blank = e_noise_op.to(devices[1])
            with torch.no_grad():
                e_nor_blank_set = []
                if cu_len == 1:
                    e_nor = model_orig.apply_model(z_n.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_nor.to(devices[1]))[0]
                    e_nor_blank = e_nor.to(devices[1])
                    e_nor_blank_set.append(e_nor_blank)
                for cu_number in range(att_size):
                    e_nor = model_orig.apply_model(z_n.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_att_set[cu_number].to(devices[1]))[0]
                    e_nor_blank = e_nor.to(devices[1])
                    e_nor_blank_set.append(e_nor_blank)
            loss_sum = 0
            coeff = torch.tensor([1., 0.24539877, 0.03190184, 0.02699387, 0.05644172, 0.06257669, 0.25889571, 0.2404908 , 0.07730061])
            for cu_number in range(cu_len + att_size):      
                loss_sum = loss_sum + coeff[cu_number] * torch.mean(e_noise_blank * e_nor_blank_set[cu_number])
            print(_,loss_sum)
            loss = loss_sum / (cu_len + 1)
            if loss > 0:
                loss.backward(retain_graph=True)
                opt2.step()
            if i == iter_break:
                break
    for i, image_name_vangogh in enumerate(images_vangogh):
        image_path_vangogh = path_vangogh + image_name_vangogh
        img_vangogh = load_img(image_path_vangogh)
        img_vangogh = preprocess(img_vangogh)
        img_vangogh = img_vangogh.to(model.device)

        encoder_posterior_vangogh = model.encode_first_stage(img_vangogh)
        z_vangogh = model.get_first_stage_encoding(encoder_posterior_vangogh).detach()

        if ori_flag:
            cu_len = 1
        else:
            cu_len = 0
        emb_att_set = []
        for cu_number in range(att_size):
            emb_att_set.append(model_orig.get_learned_conditioning([f"{atts[cu_number]}'s painting"]))
            

        emb_nor = model_orig.get_learned_conditioning([f"{prompt} painting"])
        emb_blank = model_orig.get_learned_conditioning(['painting'])

        t_enc = torch.randint(ddim_steps, (1,), device=devices[0])
        og_num = round((int(t_enc)/ddim_steps)*1000)
        og_num_lim = round((int(t_enc+1)/ddim_steps)*1000)
        t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])
        z_n = model.q_sample(z_vangogh, t_enc_ddpm)
        
        loss_sum = 0
        for j, params in enumerate(parameters):
            loss_j = torch.sum(torch.abs(params.to(devices[0]) - parameters_gt[j].to(devices[0])))
            loss_sum += loss_j
        loss_reg = loss_sum / len(parameters)
        emb_noise = model_orig.get_learned_conditioning([f"{prompt}'s painting"], delta_prompt.detach().to(devices[1]), ind_start, ind_end)
        with torch.no_grad():
            e_blank = model_orig.apply_model(z_n.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_blank.to(devices[1]))[0]
            e_noise = model_orig.apply_model(z_n.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_noise.to(devices[1]))[0]
        e_blank_op = model.apply_model(z_n.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_blank.to(devices[0]))[0]
        e_nor_op = model.apply_model(z_n.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_nor.to(devices[0]))[0]
        e_noise.requires_grad = False
        e_blank.requires_grad = False
        loss1 = criteria(e_blank_op.to(devices[0]), e_blank.to(devices[0]))
        loss = criteria(e_nor_op.to(devices[0]), e_noise.to(devices[0])) + 10*loss1
        print("loss:", loss_reg, loss)
        loss.backward()
        opt.step()
        if i == iter_break:
            break
    model.eval()

    save_model(model, name, None, save_compvis=True, save_diffusers=True, compvis_config_file=config_path, diffusers_config_file=diffusers_config_path)

def save_model(model, name, num, compvis_config_file=None, diffusers_config_file=None, device='cpu', save_compvis=True, save_diffusers=True):
    # SAVE MODEL

#     PATH = f'{FOLDER}/{model_type}-word_{word_print}-method_{train_method}-sg_{start_guidance}-ng_{neg_guidance}-iter_{i+1}-lr_{lr}-startmodel_{start_model}-numacc_{numacc}.pt'

    # folder_path = f'models/{name}'
    folder_path = f'/abmodels/{name}'
    os.makedirs(folder_path, exist_ok=True)
    if num is not None:
        path = f'{folder_path}/{name}-epoch_{num}.pt'
    else:
        path = f'{folder_path}/{name}.pt'
    if save_compvis:
        torch.save(model.state_dict(), path)

    if save_diffusers:
        print('Saving Model in Diffusers Format')
        savemodelDiffusers(name, compvis_config_file, diffusers_config_file, device=device )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'TrainESD',
                    description = 'Finetuning stable diffusion model to erase concepts using ESD method')
    parser.add_argument('--lr', help='learning rate used to train', type=float, required=False, default=1e-5)
    parser.add_argument('--lr2', help='learning rate used to learn adv emb', type=float, required=False, default=1e-6)
    parser.add_argument('--config_path', help='config path for stable diffusion v1-4 inference', type=str, required=False, default='configs/stable-diffusion/v1-inference.yaml')
    parser.add_argument('--ckpt_path', help='ckpt path for stable diffusion v1-4', type=str, required=False, default='models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt')
    parser.add_argument('--diffusers_config_path', help='diffusers unet config json path', type=str, required=False, default='diffusers_unet_config.json')
    parser.add_argument('--devices', help='cuda devices to train on', type=str, required=False, default='0,0')
    parser.add_argument('--erase_cat', help='category for erasure', type=str, required=False, default='object') # style
    parser.add_argument('--erased_index', help='index of the forgotten concept', type=int, required=True)
    parser.add_argument('--iter_break', help='iter_break used to break train', type=int, required=False, default=50)
    parser.add_argument('--w_iter_break', help='iter_break used to break word building ', type=int, required=False, default=20)
    parser.add_argument('--att_size', help='the number of used atts', type=int, required=True, default=0)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=50)
    parser.add_argument('--ori_flag', action='store_true', help='whether using c_u during adv anchoring')
    args = parser.parse_args()

    devices = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]
    atts = ['FEMALE BREAST EXPOSED', 'FEMALE GENITALIA EXPOSED', 'MALE GENITALIA EXPOSED', 'MALE BREAST EXPOSED', 'BUTTOCKS EXPOSED', 'ARMPITS EXPOSED', 'BELLY EXPOSED', 'FEET EXPOSED']
    
    # atts = ["Wings","Fuselage","Engines","Landing Gear", "Tail Assembly"]
    # church_atts = ["Spire and Dome", "Architectural Windows", "Bell Tower", "Altar and Furnishings", "Murals and Stained Glass Windows"]
    # vangogh_atts = ["Bold and Expressive Brushstrokes","Vibrant and Contrasting Colors","motional and Psychological Depth","Impasto Technique","Innovative Use of Perspective and Composition"]
    # picassoatts = ["Cubism", "Geometric Shapes", "Distorted Proportions", "Bold Colors and Contrasts", "Inventive Texture and Technique"]
    train_esd(args.iter_break, args.w_iter_break, atts,args.att_size,args.ori_flag, args.erase_cat, args.erased_index,args.lr, args.lr2, args.config_path, args.ckpt_path, args.diffusers_config_path, devices, args.ddim_steps)

