import json
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

def analyze_regression_results_per_image(file_path):
    """Analyze the patch regression results per image"""
    
    # Load all results
    results = []
    with open(file_path, 'r') as f:
        for line in f:
            results.append(json.loads(line.strip()))
    
    print(f"Total patches analyzed: {len(results)}")
    
    # Group by image_id
    images_data = defaultdict(list)
    for result in results:
        if 'image_id' in result:
            image_id = result['image_id']
            images_data[image_id].append(result)
        else:
            # Handle the case where image_id is missing (first entry)
            continue
    
    print(f"Number of images: {len(images_data)}")
    
    # Analyze each image separately
    for image_id, image_results in images_data.items():
        print(f"\n{'='*50}")
        print(f"IMAGE {image_id}")
        print(f"{'='*50}")
        print(f"Number of patches: {len(image_results)}")
        
        # Extract all tokens and their weights for this image
        all_tokens = []
        token_weights = defaultdict(list)
        
        for result in image_results:
            patch_idx = result['patch_idx']
            for token_data in result['tokens_and_weights']:
                token = token_data['token']
                weight = token_data['weight']
                all_tokens.append(token)
                token_weights[token].append(weight)
        
        # Most frequent tokens for this image
        token_counts = Counter(all_tokens)
        print(f"\nTop 10 most frequent tokens for image {image_id}:")
        for token, count in token_counts.most_common(10):
            print(f"  {token}: {count} occurrences")
        
        # Tokens with highest average absolute weights for this image
        avg_abs_weights = {}
        for token, weights in token_weights.items():
            avg_abs_weights[token] = np.mean(np.abs(weights))
        
        print(f"\nTop 10 tokens by average absolute weight for image {image_id}:")
        for token, avg_weight in sorted(avg_abs_weights.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {token}: {avg_weight:.4f}")
        
        # Analyze weight distributions for this image
        all_weights = [w for result in image_results for w in [t['weight'] for t in result['tokens_and_weights']]]
        print(f"\nWeight statistics for image {image_id}:")
        print(f"  Mean: {np.mean(all_weights):.4f}")
        print(f"  Std: {np.std(all_weights):.4f}")
        print(f"  Min: {np.min(all_weights):.4f}")
        print(f"  Max: {np.max(all_weights):.4f}")
        
        # Analyze patches with extreme weights for this image
        max_weight_patches = []
        for result in image_results:
            max_weight = max(abs(t['weight']) for t in result['tokens_and_weights'])
            if max_weight > 0.8:  # High weight threshold
                max_weight_patches.append((result['patch_idx'], max_weight))
        
        print(f"\nPatches with extreme weights (>0.8) for image {image_id}: {len(max_weight_patches)}")
        for patch_idx, max_weight in sorted(max_weight_patches, key=lambda x: x[1], reverse=True)[:5]:
            print(f"  Patch {patch_idx}: max weight {max_weight:.4f}")
        
        # Analyze positive vs negative weights for this image
        positive_weights = [w for w in all_weights if w > 0]
        negative_weights = [w for w in all_weights if w < 0]
        
        print(f"\nWeight sign distribution for image {image_id}:")
        print(f"  Positive weights: {len(positive_weights)} ({len(positive_weights)/len(all_weights)*100:.1f}%)")
        print(f"  Negative weights: {len(negative_weights)} ({len(negative_weights)/len(all_weights)*100:.1f}%)")
        print(f"  Zero weights: {len(all_weights) - len(positive_weights) - len(negative_weights)}")
    
    return results, images_data

if __name__ == "__main__":
    results, images_data = analyze_regression_results_per_image("results/patch_regression_results.jsonl") 