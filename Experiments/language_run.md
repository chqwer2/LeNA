

# 1) LoRA training command (your script already supports this)


dataset="google/boolq" 


dataset="google/boolq \
         piqa \
         allenai/social_i_qa \
         Rowan/hellaswag \
         allenai/winogrande:winogrande_xl \
         allenai/ai2_arc:ARC-Easy \
         allenai/ai2_arc:ARC-Challenge \
         allenai/openbookqa" 



conda activate lora
cd ~/hao/repo/FLoRA/Experiments


model=meta-llama/Llama-3.2-1B     # Test

model=meta-llama/Llama-2-7b-hf
model=meta-llama/Llama-2-13b-hf
model=huggyllama/llama-7b


model=meta-llama/Meta-Llama-3-8B    # Test




```bash
CUDA_VISIBLE_DEVICES=0  python Llama_Adaptation.py \
  --base_model $model \
  --dataset $dataset \
  --output_dir runs/lora \
  --batch_size 1 \
  --num_epochs 3 \
  --learning_rate 3e-4 \
  --cutoff_len 512 \
  --eval_step 10 \
  --save_step 100 \
  --device auto \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,up_proj,down_proj \
  --methods lora
```




[//]: # (o_proj,gate_proj,)

# 2) DoRA
CUDA_VISIBLE_DEVICES=0 python Llama_Adaptation.py  \
  --base_model $model \
  --dataset $dataset \
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
Below are your **rewritten commands** where **all activation-specific knobs are passed via** `--lena_activation_kwargs_json '...json...'` (instead of `--lena_fourier_terms`, `--lena_spline_knots`, `--lena_poly_degree`, etc.). I also kept your GPU selection via `CUDA_VISIBLE_DEVICES=...`.

---

## 3.3 FLoRA + Fourier

### (a) channel, gate 

```bash
strength=soft
strength=hard
CUDA_VISIBLE_DEVICES=0 python Llama_Adaptation.py \
  --base_model "$model" \
  --dataset "$dataset" \
  --output_dir runs/lena_fourier_channel \
  --batch_size 1 \
  --num_epochs 3 \
  --learning_rate 3e-4 \
  --cutoff_len 512 \
  --eval_step 10 \
  --save_step 100 \
  --device auto \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --methods lena \
  --lena_activations fourier \
  --lena_flex_mode "spatial" \
  --lena_gate_type none \
  --lena_gate_position after_b \
  --lena_gate_mode voxel \
  --gate_strength ${strength} \
  --lena_activation_kwargs_json '{"n_terms":5,"init_scale":0.01, "use_gate": "hard"}'   >> output.txt

```




## 3.2 FLoRA + Spline (channel), gate OFF

```bash
CUDA_VISIBLE_DEVICES=0 python Llama_Adaptation.py \
  --base_model "$model" \
  --dataset "$dataset" \
  --output_dir runs/lena_spline_channel \
  --batch_size 1 \
  --num_epochs 3 \
  --learning_rate 3e-4 \
  --cutoff_len 512 \
  --eval_step 10 \
  --save_step 100 \
  --device auto \
  --lora_r 32 --lora_alpha 64  --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,up_proj,down_proj \
  --methods lena \
  --lena_activations spline \
  --lena_flex_mode voxel \
  --lena_gate_type none \
  --lena_gate_position after_b \
  --lena_activation_kwargs_json '{"n_knots":16,"x_min":-3.0,"x_max":3.0,"init":"identity", "use_gate": "hard"}'  

```

[//]: # (o_proj,gate_proj,)

[//]: # ( >> output.txt)


---

## 3.5 FLoRA + Polynomial (channel), gate OFF

```bash
CUDA_VISIBLE_DEVICES=0 python Llama_Adaptation.py \
  --base_model "$model" \
  --dataset "$dataset" \
  --output_dir runs/lena_poly_channel \
    --batch_size 1 \
  --num_epochs 3 \
  --learning_rate 3e-4 \
  --cutoff_len 512 \
  --eval_step 10 \
  --save_step 100 \
  --device auto \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --methods lena \
  --lena_activations polynomial \
  --lena_flex_mode channel \
  --lena_gate_type none \
  --lena_gate_position after_b \
  --lena_activation_kwargs_json '{"degree":3,"init":"identity", "use_gate": "hard"}'   >> output.txt
   
  
```

---

# 4) FLoRA + gate examples (activation can be on or off)

Below I keep Fourier as your example, and still route Fourier params through JSON.

## 4.1 Gate after A (sigmoid)

```bash
CUDA_VISIBLE_DEVICES=1 python Llama_Adaptation.py \
  --base_model "$model" \
  --dataset "$dataset" \
  --output_dir runs/lena_fourier_gate_after_a \
  --batch_size 1 --num_epochs 3 --learning_rate 3e-4 --cutoff_len 512 \
  --eval_step 10 --save_step 100 --device auto \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --methods lena \
  --lena_activations fourier --lena_flex_mode channel \
  --lena_gate_type sigmoid --lena_gate_position after_a \
  --lena_activation_kwargs_json '{"n_terms":4,"init_scale":0.01}'
```

## 4.2 Gate after B (sigmoid)

```bash
CUDA_VISIBLE_DEVICES=1 python Llama_Adaptation.py \
  --base_model "$model" \
  --dataset "$dataset" \
  --output_dir runs/lena_fourier_gate_after_b \
  --batch_size 1 --num_epochs 3 --learning_rate 3e-4 --cutoff_len 512 \
  --eval_step 10 --save_step 100 --device auto \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --methods lena \
  --lena_activations fourier --lena_flex_mode channel \
  --lena_gate_type sigmoid --lena_gate_position after_b \
  --lena_activation_kwargs_json '{"n_terms":4,"init_scale":0.01}'
```

## 4.3 Gate both (sigmoid)

```bash
CUDA_VISIBLE_DEVICES=1 python Llama_Adaptation.py \
  --base_model "$model" \
  --dataset "$dataset" \
  --output_dir runs/lena_fourier_gate_both \
  --batch_size 1 --num_epochs 3 --learning_rate 3e-4 --cutoff_len 512 \
  --eval_step 10 --save_step 100 --device auto \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --methods lena \
  --lena_activations fourier --lena_flex_mode channel \
  --lena_gate_type sigmoid --lena_gate_position both \
  --lena_activation_kwargs_json '{"n_terms":4,"init_scale":0.01}'
```

---

### Notes (so your JSON matches your activation code)

* **fourier** supports: `n_terms`, `init_scale`, plus whatever else your `FlexFourier.__init__` takes.
* **spline** supports: `n_knots`, `x_min`, `x_max`, `init`, etc.
* **polynomial** supports: `degree`, `init`, `init_scale`, etc.
* If you switch to `lena_flex_mode spatial` or `voxel`, you’ll also want JSON like `{"max_h":512,"max_w":1,...}` so variable sequence lengths can slice safely.

If you paste your exact `FlexFourier/FlexSpline/FlexPolynomial` `__init__` signatures (you already pasted most of it), I can align the JSON keys 1:1 with what your implementation actually accepts.

```

---

## Notes on `tok/s`

`tok/s = seq_len / mean_forward_time` is just a **rough throughput estimate for a forward pass**, not true generation speed. It can look “wrong” if your seq_len is padded, if the model uses caching, or if timing includes overhead. For training, you usually care about **samples/sec** or **tokens/sec over actual batch tokens** (Trainer logs can do that).

