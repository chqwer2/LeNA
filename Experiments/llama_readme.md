
pip install "datasets<4.0.0" "huggingface_hub<0.24"

pip install transformers -U`

# 1) LoRA training command (your script already supports this)

# huggyllama/llama-7b | microsoft/phi-2  | meta-llama/Llama-3.2-1B  | meta-llama/Llama-3.2-3B
 
#  google/boolq | piqa | allenai/social_i_qa | Rowan/hellaswag | allenai/winogrande (winogrande_xl) allenai/ai2_arc (ARC-Easy) allenai/openbookqa (main)

dataset=google/boolq
dataset=piqa

                   # no config
ds = load_dataset("social_i_qa")                # no config
ds = load_dataset("hellaswag")                  # no config
ds = load_dataset("winogrande", "winogrande_xl")
ds = load_dataset("ai2_arc", "ARC-Easy")
ds = load_dataset("openbookqa", "main")



conda activate lora
cd ~/hao/repo/FLoRA/Experiments
model=meta-llama/Llama-3.2-1B     # Test

model=meta-llama/Llama-2-7b-hf
model=meta-llama/Llama-2-13b-hf
model=huggyllama/llama-7b


model=meta-llama/Meta-Llama-3-8B    # Test
dataset=google/boolq



```bash
CUDA_VISIBLE_DEVICES=0  python Llama_Adaptation.py \
  --base_model $model \
  --data_path $dataset \
  --output_dir runs/lora \
  --batch_size 1 \
  --num_epochs 3 \
  --learning_rate 3e-4 \
  --cutoff_len 512 \
  --eval_step 10 \
  --save_step 100 \
  --device auto \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --methods lora
```


# 2) DoRA
CUDA_VISIBLE_DEVICES=0 python Llama_Adaptation.py  \
  --base_model $model \
  --data_path $dataset \
  --output_dir runs/dora \
  --batch_size 1 \
  --num_epochs 3 \
  --learning_rate 3e-4 \
  --cutoff_len 512 \
  --eval_step 10 \
  --save_step 100 \
  --device auto \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --methods dora


-----

# 3) FLoRA training commands
Below are your **rewritten commands** where **all activation-specific knobs are passed via** `--flora_activation_kwargs_json '...json...'` (instead of `--flora_fourier_terms`, `--flora_spline_knots`, `--flora_poly_degree`, etc.). I also kept your GPU selection via `CUDA_VISIBLE_DEVICES=...`.

---

## 3.3 FLoRA + Fourier

### (a) channel, gate 

```bash
strength=soft
strength=hard
CUDA_VISIBLE_DEVICES=0 python Llama_Adaptation.py \
  --base_model "$model" \
  --data_path "$dataset" \
  --output_dir runs/flora_fourier_channel \
  --batch_size 1 \
  --num_epochs 3 \
  --learning_rate 3e-4 \
  --cutoff_len 512 \
  --eval_step 10 \
  --save_step 100 \
  --device auto \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --methods flora \
  --flora_activations fourier \
  --flora_flex_mode "spatial" \
  --flora_gate_type none \
  --flora_gate_position after_b \
  --flora_gate_mode none \
  --gate_strength ${strength} \
  --flora_activation_kwargs_json '{"n_terms":5,"init_scale":0.01, "use_gate": "hard"}'   >> output.txt



```




## 3.2 FLoRA + Spline (channel), gate OFF

```bash
CUDA_VISIBLE_DEVICES=0 python Llama_Adaptation.py \
  --base_model "$model" \
  --data_path "$dataset" \
  --output_dir runs/flora_spline_channel \
  --batch_size 1 \
  --num_epochs 3 \
  --learning_rate 3e-4 \
  --cutoff_len 512 \
  --eval_step 10 \
  --save_step 100 \
  --device auto \
  --lora_r 8 --lora_alpha 16  --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --methods flora \
  --flora_activations spline \
  --flora_flex_mode channel \
  --flora_gate_type none \
  --flora_gate_position after_b \
  --flora_activation_kwargs_json '{"n_knots":16,"x_min":-3.0,"x_max":3.0,"init":"identity", "use_gate": "hard"}'   >> output.txt


```



---

## 3.5 FLoRA + Polynomial (channel), gate OFF

```bash
CUDA_VISIBLE_DEVICES=0 python Llama_Adaptation.py \
  --base_model "$model" \
  --data_path "$dataset" \
  --output_dir runs/flora_poly_channel \
    --batch_size 1 \
  --num_epochs 3 \
  --learning_rate 3e-4 \
  --cutoff_len 512 \
  --eval_step 10 \
  --save_step 100 \
  --device auto \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --methods flora \
  --flora_activations polynomial \
  --flora_flex_mode channel \
  --flora_gate_type none \
  --flora_gate_position after_b \
  --flora_activation_kwargs_json '{"degree":3,"init":"identity", "use_gate": "hard"}'   >> output.txt
   
  
```

---

# 4) FLoRA + gate examples (activation can be on or off)

Below I keep Fourier as your example, and still route Fourier params through JSON.

## 4.1 Gate after A (sigmoid)

```bash
CUDA_VISIBLE_DEVICES=1 python Llama_Adaptation.py \
  --base_model "$model" \
  --data_path "$dataset" \
  --output_dir runs/flora_fourier_gate_after_a \
  --batch_size 1 --num_epochs 3 --learning_rate 3e-4 --cutoff_len 512 \
  --eval_step 10 --save_step 100 --device auto \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --methods flora \
  --flora_activations fourier --flora_flex_mode channel \
  --flora_gate_type sigmoid --flora_gate_position after_a \
  --flora_activation_kwargs_json '{"n_terms":4,"init_scale":0.01}'
```

## 4.2 Gate after B (sigmoid)

```bash
CUDA_VISIBLE_DEVICES=1 python Llama_Adaptation.py \
  --base_model "$model" \
  --data_path "$dataset" \
  --output_dir runs/flora_fourier_gate_after_b \
  --batch_size 1 --num_epochs 3 --learning_rate 3e-4 --cutoff_len 512 \
  --eval_step 10 --save_step 100 --device auto \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --methods flora \
  --flora_activations fourier --flora_flex_mode channel \
  --flora_gate_type sigmoid --flora_gate_position after_b \
  --flora_activation_kwargs_json '{"n_terms":4,"init_scale":0.01}'
```

## 4.3 Gate both (sigmoid)

```bash
CUDA_VISIBLE_DEVICES=1 python Llama_Adaptation.py \
  --base_model "$model" \
  --data_path "$dataset" \
  --output_dir runs/flora_fourier_gate_both \
  --batch_size 1 --num_epochs 3 --learning_rate 3e-4 --cutoff_len 512 \
  --eval_step 10 --save_step 100 --device auto \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --methods flora \
  --flora_activations fourier --flora_flex_mode channel \
  --flora_gate_type sigmoid --flora_gate_position both \
  --flora_activation_kwargs_json '{"n_terms":4,"init_scale":0.01}'
```

---

### Notes (so your JSON matches your activation code)

* **fourier** supports: `n_terms`, `init_scale`, plus whatever else your `FlexFourier.__init__` takes.
* **spline** supports: `n_knots`, `x_min`, `x_max`, `init`, etc.
* **polynomial** supports: `degree`, `init`, `init_scale`, etc.
* If you switch to `flora_flex_mode spatial` or `voxel`, you’ll also want JSON like `{"max_h":512,"max_w":1,...}` so variable sequence lengths can slice safely.

If you paste your exact `FlexFourier/FlexSpline/FlexPolynomial` `__init__` signatures (you already pasted most of it), I can align the JSON keys 1:1 with what your implementation actually accepts.

```

---

## Notes on `tok/s`

`tok/s = seq_len / mean_forward_time` is just a **rough throughput estimate for a forward pass**, not true generation speed. It can look “wrong” if your seq_len is padded, if the model uses caching, or if timing includes overhead. For training, you usually care about **samples/sec** or **tokens/sec over actual batch tokens** (Trainer logs can do that).

