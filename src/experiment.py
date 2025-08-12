import torch
import json
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
from HookedVLM import HookedLVLM
import json
import argparse
import os
from tqdm import tqdm
from InputEmbed import InputsEmbeds
from sklearn.linear_model import LinearRegression
import numpy as np
from utils import correct_annotations_for_crop, find_overlapping_patches, get_object_patch_indices, replace_image_regions_with_patches
import torch.nn.functional as F
from CocoDataset import COCOImageDataset
from pycocotools.coco import COCO 
print("Starting script...")


def create_logit_lens(hidden_states, norm, lm_head, tokenizer, image, model_name, image_filename, prompt, save_folder = ".", image_size=336, patch_size=14, misc_text=""):
    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt)
    
    # Find the image token and replace it with image tokens
    img_token_id = 32000  # The token ID for <img>
    img_token_count = (image_size // patch_size) ** 2  # 576 for 336x336 image with 14x14 patches
    
    token_labels = []
    for token_id in input_ids:
        if token_id == img_token_id:
            # One indexed because the HTML logic wants it that way
            token_labels.extend([f"<IMG{(i+1):03d}>" for i in range(img_token_count)])
        else:
            token_labels.append(tokenizer.decode([token_id]))
    
    # Exclude the input embedding layer if it's included
    num_layers = len(hidden_states)
    sequence_length = hidden_states[0].size(1)
    
    all_top_tokens = []
    out_path = os.path.join(save_folder, "logit_lens_output.json")
    
    with open(out_path, "w") as f:
        for layer in range(num_layers):
            layer_hidden_states = hidden_states[layer]
            
            # Apply norm and lm_head
            normalized = norm(layer_hidden_states)
            logits = lm_head(normalized)
            
            # Get probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Get top 5 tokens and their probabilities for each position
            top_5_values, top_5_indices = torch.topk(probs, k=10, dim=-1)
            
            layer_top_tokens = []
            
            for pos in range(sequence_length):
                top_5_tokens = [tokenizer.decode(idx.item()) for idx in top_5_indices[0, pos]]
                top_5_probs = [f"{prob.item():.4f}" for prob in top_5_values[0, pos]]
                
                entry = {
                    "layer": layer,
                    "position": pos,
                    "token_label": token_labels[pos] if pos < len(token_labels) else f"pos_{pos}",
                    "top_tokens": top_5_tokens,
                    "probabilities": top_5_probs
                }
                f.write(json.dumps(entry) + "\n")

                layer_top_tokens.append(list(zip(top_5_tokens, top_5_probs)))
            
            all_top_tokens.append(layer_top_tokens)

    
    return all_top_tokens

def is_image_file(filename):
    valid_extensions = ('.jpeg', '.jpg', '.png', '.JPEG', '.JPG', '.PNG')
    return filename.lower().endswith(valid_extensions)

def process_images(image_folder, save_folder, device, quantize_type, num_images):
    # Import Model
    ann_file="/home/jingyi/others/VLM_Thesis/data/annotations/instances_train2017.json"
    coco = COCO(ann_file)  
    model = HookedLVLM(device=device, quantize=True, quantize_type=quantize_type)
    print("success")
    # Load components needed for logit lens
    norm = model.model.language_model.model.norm
    lm_head = model.model.language_model.lm_head
    tokenizer = model.processor.tokenizer
    model_name = model.model.config._name_or_path.split("/")[-1]

    # Load images
    image_files = [f for f in os.listdir(image_folder) if is_image_file(f)]
    if num_images:
        image_files = image_files[:num_images]
    
    image_paths = [os.path.join(image_folder, f) for f in image_files]
    images = {}
    for image_path in image_paths:
        try:
            image = Image.open(image_path)
            print(image_path)
            images[image_path] = image
        except IOError:
            print(f"Could not open image file: {image_path}")

    # Run forward pass
    for image_path, image in tqdm(images.items()):
        filename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(filename)[0]  # "000000000061"
        
# Convert to int (drops leading zeros automatically)
        img_id = int(name_without_ext)       
        print(img_id)
        ann_id = coco.getAnnIds(imgIds=img_id)
        ann = coco.loadAnns(ann_id)
        text_question = "Describe the image."
        prompt = f"USER: <image>\n{text_question} ASSISTANT:"
        activations = model.get_text_model_in(image, prompt)
        inputs_embeds = InputsEmbeds(model.processor.tokenizer, activations, prompt)
        img_embeds, text_embeds, start_end_indices = inputs_embeds.get_img_and_text_embed()
        img_embeds, text_embeds, start_end_indices = inputs_embeds.get_img_and_text_embed()

        patch_ablation_indices_0 = get_object_patch_indices(image, ann, start_end_indices, buffer=0)
        patch_ablation_indices_1 = get_object_patch_indices(image, ann, start_end_indices, buffer=1)
        patch_ablation_indices_2 = get_object_patch_indices(image, ann, start_end_indices, buffer=2)
     
        
        token_embed = model.model.get_input_embeddings().weight.detach().cpu().numpy()  # [vocab_size, D]
        print(img_embeds.shape)
        token_embed_tensor = torch.tensor(token_embed, dtype=torch.float32, device=device).T  # [4096, vocab_size]

# Precompute pseudoinverse (slow once, fast later)
        XtX_inv_Xt = torch.pinverse(token_embed_tensor)  # [vocab_size, 4096]
        
        # For each image patch, perform linear regression
        for i, img_embed in enumerate(img_embeds):
            if i >= 1:  # Only process first patch for now
                break
                
            image_emb_cpu = img_embed.detach().cpu().numpy()  # Shape: [576, 4096]
            # For each patch, perform linear regression (only first 5 patches)
            #for patch_idx in range(image_emb_cpu.shape[0]):
            for patch_idx in patch_ablation_indices_0:
                patch_embedding = torch.tensor(image_emb_cpu[patch_idx], dtype=torch.float32, device=device)  # [4096]

    # Linear regression: solve for weights [vocab_size]
                weights = XtX_inv_Xt @ patch_embedding  # [vocab_size]

    # Top tokens by absolute weight
                topk = torch.topk(weights.abs(), k=10)
                top_indices = topk.indices.tolist()
                top_weights = weights[top_indices].tolist()
                top_tokens = tokenizer.convert_ids_to_tokens(top_indices)

                regression_results = {
        "image_id": img_id,
        "patch_idx": int(patch_idx),
        "tokens_and_weights": [
            {"token": token, "weight": float(weight)}
            for token, weight in zip(top_tokens, top_weights)
        ]
    }

    # Write to file
                regression_file = os.path.join(save_folder, "patch_regression_results.jsonl")
                with open(regression_file, "a") as f:
                    f.write(json.dumps(regression_results) + "\n")

                print(f"Patch {patch_idx}:")
                for token, weight in zip(top_tokens, top_weights):
                    print(f"  {token:15} : {weight:.4f}")
                print()




        #hidden_states = model.forward(image, prompt, output_hidden_states=True).hidden_states
        #create_logit_lens(hidden_states, norm, lm_head, tokenizer, image, model_name, image_path, prompt, save_folder)

def main():
    parser = argparse.ArgumentParser(description="Process images using HookedLVLM model")
    parser.add_argument("--image_folder", required=True, help="Path to the folder containing images")
    parser.add_argument("--save_folder", required=True, help="Path to save the results")
    parser.add_argument("--device", default="cuda:0", help="Device to run the model on")
    parser.add_argument("--quantize_type", default="fp16", help="Quantization type")
    parser.add_argument("--num_images", type=int, help="Number of images to process (optional)")

    args = parser.parse_args()
    print(args.image_folder)
    process_images(args.image_folder, args.save_folder, args.device, args.quantize_type, args.num_images)

if __name__ == "__main__":
    main()