

# 1) LoRA training command (your script already supports this)

# huggyllama/llama-7b | microsoft/phi-2  | meta-llama/Llama-3.2-1B  | meta-llama/Llama-3.2-3B
 
#  google/boolq | ybisk/piqa | allenai/social_i_qa | Rowan/hellaswag | allenai/winogrande (winogrande_xl) allenai/ai2_arc (ARC-Easy) allenai/openbookqa (main)


model=meta-llama/Llama-3.2-1B
dataset=google/boolq


```bash
CUDA_VISIBLE_DEVICES=0  python Llama_Dora.py \
  --base_model $model \
  --data_path $dataset \
  --output_dir runs/lora \
  --batch_size 1 \
  --num_epochs 1 \
  --learning_rate 3e-4 \
  --cutoff_len 512 \
  --eval_step 10 \
  --save_step 100 \
  --device auto \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

**Note:** LoRA is the default when you do **not** pass `--use_dora`.

---



# 2) DoRA training command (your script already supports this)

```bash
CUDA_VISIBLE_DEVICES=1 python Llama_Dora.py \
  --base_model $model \
  --data_path $dataset \
  --output_dir runs/dora \
  --batch_size 1 \
  --num_epochs 1 \
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
```

---

# 3) FLoRA training commands (requires script supports `--adapter flora` + FloraConfig)

## 3.0 Pure LoRA-equivalent FLoRA (activation OFF + gate OFF)

This should behave like standard LoRA (but via your Flora wrappers).

```bash
python Llama_Dora.py \
  --adapter flora \
  --base_model $model \
  --data_path $dataset \
  --output_dir runs/flora_pure_lora \
  --batch_size 1 \
  --num_epochs 1 \
  --learning_rate 3e-4 \
  --cutoff_len 512 \
  --eval_step 10 \
  --save_step 100 \
  --device auto \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --flora_activation identity \
  --flora_flex_mode channel \
  --flora_gate_type none
```

## 3.1 FLoRA + ReLU activation (channel mode), gate OFF

```bash
python Llama_Dora.py \
  --adapter flora \
  --base_model $model \
  --data_path $dataset \
  --output_dir runs/flora_relu_channel \
  --batch_size 1 \
  --num_epochs 1 \
  --learning_rate 3e-4 \
  --cutoff_len 512 \
  --eval_step 10 \
  --save_step 100 \
  --device auto \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --flora_activation relu \
  --flora_flex_mode channel \
  --flora_gate_type none
```

## 3.2 FLoRA + GELU (channel), gate OFF

```bash
python Llama_Dora.py \
  --adapter flora \
  --base_model $model \
  --data_path $dataset \
  --output_dir runs/flora_gelu_channel \
  --batch_size 1 \
  --num_epochs 1 \
  --learning_rate 3e-4 \
  --cutoff_len 512 \
  --eval_step 10 \
  --save_step 100 \
  --device auto \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --flora_activation gelu \
  --flora_flex_mode channel \
  --flora_gate_type none
```

## 3.3 FLoRA + Fourier (channel), gate OFF

```bash
python Llama_Dora.py \
  --adapter flora \
  --base_model $model \
  --data_path $dataset \
  --output_dir runs/flora_fourier_channel \
  --batch_size 1 \
  --num_epochs 1 \
  --learning_rate 3e-4 \
  --cutoff_len 512 \
  --eval_step 10 \
  --save_step 100 \
  --device auto \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --flora_activation fourier \
  --flora_flex_mode channel \
  --flora_gate_type none \
  --flora_fourier_terms 4
```

## 3.4 FLoRA + Spline (channel), gate OFF

```bash
python Llama_Dora.py \
  --adapter flora \
  --base_model $model \
  --data_path $dataset \
  --output_dir runs/flora_spline_channel \
  --batch_size 1 \
  --num_epochs 1 \
  --learning_rate 3e-4 \
  --cutoff_len 512 \
  --eval_step 10 \
  --save_step 100 \
  --device auto \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --flora_activation spline \
  --flora_flex_mode channel \
  --flora_gate_type none \
  --flora_spline_knots 16
```

## 3.5 FLoRA + Polynomial (channel), gate OFF

```bash
python Llama_Dora.py \
  --adapter flora \
  --base_model $model \
  --data_path $dataset \
  --output_dir runs/flora_poly_channel \
  --batch_size 1 \
  --num_epochs 1 \
  --learning_rate 3e-4 \
  --cutoff_len 512 \
  --eval_step 10 \
  --save_step 100 \
  --device auto \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --flora_activation polynomial \
  --flora_flex_mode channel \
  --flora_gate_type none \
  --flora_poly_degree 3
```

---

# 4) FLoRA + gate examples (activation can be on or off)

## 4.1 Gate after A (sigmoid)

```bash
python Llama_Dora.py \
  --adapter flora \
  --base_model $model \
  --data_path $dataset \
  --output_dir runs/flora_fourier_gate_after_a \
  --batch_size 1 --num_epochs 1 --learning_rate 3e-4 --cutoff_len 512 \
  --eval_step 10 --save_step 100 --device auto \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --flora_activation fourier --flora_flex_mode channel \
  --flora_gate_type sigmoid --flora_gate_position after_a
```

## 4.2 Gate after B (sigmoid)

```bash
python Llama_Dora.py \
  --adapter flora \
  --base_model $model \
  --data_path $dataset \
  --output_dir runs/flora_fourier_gate_after_b \
  --batch_size 1 --num_epochs 1 --learning_rate 3e-4 --cutoff_len 512 \
  --eval_step 10 --save_step 100 --device auto \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --flora_activation fourier --flora_flex_mode channel \
  --flora_gate_type sigmoid --flora_gate_position after_b
```

## 4.3 Gate both (sigmoid)

```bash
python Llama_Dora.py \
  --adapter flora \
  --base_model $model \
  --data_path $dataset \
  --output_dir runs/flora_fourier_gate_both \
  --batch_size 1 --num_epochs 1 --learning_rate 3e-4 --cutoff_len 512 \
  --eval_step 10 --save_step 100 --device auto \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --flora_activation fourier --flora_flex_mode channel \
  --flora_gate_type sigmoid --flora_gate_position both
```

---

## Notes on `tok/s`

`tok/s = seq_len / mean_forward_time` is just a **rough throughput estimate for a forward pass**, not true generation speed. It can look “wrong” if your seq_len is padded, if the model uses caching, or if timing includes overhead. For training, you usually care about **samples/sec** or **tokens/sec over actual batch tokens** (Trainer logs can do that).

