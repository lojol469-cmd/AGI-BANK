import sys
import json
import time
import requests
import os
import shutil
import argparse

COMFYUI_URL = 'http://127.0.0.1:8188'
WORKFLOW_PATH = r'D:\kibali-IA\outils\comfy_workflow_turbo.json'
COMFY_OUTPUT_DIR = r'D:\ComfyUI_windows_portable\ComfyUI_windows_portable\ComfyUI\output'
KIBALI_ASSETS_DIR = r'D:\kibali-IA\assets'

def generate_scene(prompt, scene_index):
    if not os.path.exists(WORKFLOW_PATH):
        return {'success': False, 'error': 'Workflow JSON introuvable (comfy_workflow_turbo.json).'}

    try:
        with open(WORKFLOW_PATH, 'r', encoding='utf-8') as f:
            workflow = json.load(f)

        prompt_injected = False
        for node_id, node in workflow.items():
            if node.get('class_type') == 'CLIPTextEncode':
                if not prompt_injected:
                    node['inputs']['text'] = prompt
                    prompt_injected = True
                    break 
        
        if not prompt_injected:
            return {'success': False, 'error': 'Impossible de trouver le noeud CLIPTextEncode.'}

        r = requests.post(f'{COMFYUI_URL}/prompt', json={'prompt': workflow}, timeout=10)
        if r.status_code != 200:
            return {'success': False, 'error': f'Erreur ComfyUI: {r.text}'}
        
        prompt_id = r.json()['prompt_id']
        
        for _ in range(120):
            time.sleep(2)
            hist = requests.get(f'{COMFYUI_URL}/history/{prompt_id}', timeout=5).json()
            if prompt_id in hist:
                outputs = hist[prompt_id].get('outputs', {})
                for node_out in outputs.values():
                    if 'images' in node_out:
                        for img in node_out['images']:
                            filename = img['filename']
                            source_path = os.path.join(COMFY_OUTPUT_DIR, filename)
                            
                            os.makedirs(KIBALI_ASSETS_DIR, exist_ok=True)
                            ext = os.path.splitext(filename)[1]
                            target_name = f'scene_{scene_index:02d}{ext}'
                            target_path = os.path.join(KIBALI_ASSETS_DIR, target_name)
                            
                            if os.path.exists(source_path):
                                shutil.copy(source_path, target_path)
                                return {'success': True, 'path': target_path}
                            else:
                                return {'success': False, 'error': f'Fichier source {source_path} introuvable.'}
        
        return {'success': False, 'error': 'Timeout de generation.'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--storyboard', type=str, required=True)
    args = parser.parse_args()

    try:
        prompts = json.loads(args.storyboard)
        if not isinstance(prompts, list):
            prompts = [prompts]
    except json.JSONDecodeError:
        prompts = [args.storyboard]

    results = []
    for i, prompt in enumerate(prompts):
        print(f'[{i+1}/{len(prompts)}] Generation : {prompt}...', file=sys.stderr)
        res = generate_scene(prompt, i+1)
        results.append(res)

    print(json.dumps({'success': True, 'assets': results}))
