python - <<'PY'
import json, os
model_path = "/data/taosen/code/MCITlib/checkpoint/UCIT/LLaVA-1.5/HiDe/Task1_llava_lora"
model_base = "/data/taosen/pretrain/llava-v1.5-7b"
for p in [model_path, model_base]:
    cfg = os.path.join(p, "config.json")
    print("==", cfg)
    with open(cfg) as f:
        j = json.load(f)
    for k in ["mm_vision_tower","mm_use_im_patch_token","mm_use_im_start_end","mm_vision_select_layer","image_aspect_ratio"]:
        print(k, j.get(k))
    print("tokenizer_model_max_length", j.get("tokenizer_model_max_length"))
    print()
PY