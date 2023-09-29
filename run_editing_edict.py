from models.edict.edict_functions import *
import argparse
from utils.utils import txt_draw,load_512
import json

def mask_decode(encoded_mask,image_shape=[512,512]):
    length=image_shape[0]*image_shape[1]
    mask_array=np.zeros((length,))
    
    for i in range(0,len(encoded_mask),2):
        splice_len=min(encoded_mask[i+1],length-encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i]+j]=1
            
    mask_array=mask_array.reshape(image_shape[0], image_shape[1])
    # to avoid annotation errors in boundary
    mask_array[0,:]=1
    mask_array[-1,:]=1
    mask_array[:,0]=1
    mask_array[:,-1]=1
            
    return mask_array

def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def edit_image_edict_p2p(
    image_path,
    prompt_src,
    prompt_tar,
    use_p2p):
    
    im = load_im_into_format_from_path(image_path)
    latents = coupled_stablediffusion(prompt_src,
                                reverse=True,
                                    init_image=im,
                                    run_baseline=False,
                                )
    recon = coupled_stablediffusion(prompt_src,
                                reverse=False,
                                    fixed_starting_latent=latents,
                                    run_baseline=False,
                                )
    recon = recon[0]
    edit=EDICT_editing(image_path,
              prompt_src,
              prompt_tar,
              use_p2p=use_p2p)[0]
    
    image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
    
    image_gt = load_512(image_path)
    
    out_image=Image.fromarray(np.concatenate((image_instruct,image_gt,np.array(recon),np.array(edit),),axis=1))

    return out_image

image_save_paths={
    "edict+direct_forward":"edict+direct_forward",
    "edict+p2p":"edict+p2p",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun_exist_images', action= "store_true") # rerun existing images
    parser.add_argument('--data_path', type=str, default="data") # the editing category that needed to run
    parser.add_argument('--output_path', type=str, default="output") # the editing category that needed to run
    parser.add_argument('--edit_category_list', nargs = '+', type=str, default=["0","1","2","3","4","5","6","7","8","9"]) # the editing category that needed to run
    parser.add_argument('--edit_method_list', nargs = '+', type=str, default=["edict+p2p"]) # the editing methods that needed to run
    args = parser.parse_args()
    
    rerun_exist_images=args.rerun_exist_images
    data_path=args.data_path
    output_path=args.output_path
    edit_category_list=args.edit_category_list
    edit_method_list=args.edit_method_list
    
    
    with open(f"{data_path}/mapping_file.json", "r") as f:
        editing_instruction = json.load(f)
    
    for key, item in editing_instruction.items():
        
        if item["editing_type_id"] not in edit_category_list:
            continue
        
        original_prompt = item["original_prompt"].replace("[", "").replace("]", "")
        editing_prompt = item["editing_prompt"].replace("[", "").replace("]", "")
        image_path = os.path.join(f"{data_path}/annotation_images", item["image_path"])
        editing_instruction = item["editing_instruction"]
        blended_word = item["blended_word"].split(" ") if item["blended_word"] != "" else []
        mask = Image.fromarray(np.uint8(mask_decode(item["mask"])[:,:,np.newaxis].repeat(3,2))).convert("L")

        for edit_method in edit_method_list:
            if edit_method=="edict+direct_forward":
                use_p2p=False
            elif edit_method=="edict+p2p":
                use_p2p=True
            present_image_save_path=image_path.replace(data_path, os.path.join(output_path,image_save_paths[edit_method]))
            if ((not os.path.exists(present_image_save_path)) or rerun_exist_images):
                print(f"editing image [{image_path}] with [{edit_method}]")
                setup_seed()
                torch.cuda.empty_cache()
                edited_image = edit_image_edict_p2p(
                        image_path=image_path,
                        prompt_src=original_prompt,
                        prompt_tar=editing_prompt,
                        use_p2p=use_p2p) 
                    
                if not os.path.exists(os.path.dirname(present_image_save_path)):
                    os.makedirs(os.path.dirname(present_image_save_path))
                edited_image.save(present_image_save_path)
                
                print(f"finish")
                
            else:
                print(f"skip image [{image_path}] with [{edit_method}]")
        
        