"""End-to-end on a tiny random Llama: training step with left padding; cached beam-search decoding equals
uncached decoding (prefix statistics carried and reordered correctly). Run with an env that has transformers."""
import torch
from transformers import LlamaConfig, LlamaForCausalLM, GenerationConfig
from lena_v7 import inject_lena, install_lena_hooks, lena_stream, adapter_state_dict, trainable_count

torch.manual_seed(0)


def build(**kw):
    cfg = LlamaConfig(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                      pad_token_id=0, bos_token_id=1, eos_token_id=2)
    model = LlamaForCausalLM(cfg).double()
    for p in model.parameters():
        p.requires_grad_(False)
    names = inject_lena(model, ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"], r=4, alpha=8,
                        causal=True, **kw)
    assert len(names) == 10
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith(".B"):
                p.normal_(0, 0.2)
            if n.endswith(".rho"):
                p.add_(0.5 * torch.randn_like(p))
            if n.endswith(".A1") or n.endswith(".A2"):
                p.mul_(10)                        # spread codes over the grid so fits are active
    install_lena_hooks(model)
    return model


def test_train_step_left_padding():
    model = build()
    model.train()
    ids = torch.randint(3, 97, (3, 20)); mask = torch.ones_like(ids)
    ids[0, :5] = 0; mask[0, :5] = 0
    labels = ids.masked_fill(mask == 0, -100)
    out = model(input_ids=ids, attention_mask=mask, labels=labels)
    out.loss.backward()
    assert torch.isfinite(out.loss)
    g = [p.grad for n, p in model.named_parameters() if n.endswith(".A2")]
    assert all(x is not None and x.abs().sum() > 0 for x in g)
    # padding invariance: row 0 with pads equals the unpadded sequence
    model.eval()
    with torch.no_grad():
        pos = (mask[:1].cumsum(1) - 1).clamp_min(0)            # same RoPE positions as the unpadded run
        a = model(input_ids=ids[:1], attention_mask=mask[:1], position_ids=pos).logits[0, 5:]
        b = model(input_ids=ids[:1, 5:], attention_mask=mask[:1, 5:]).logits[0]
    assert torch.allclose(a, b, atol=1e-8), (a - b).abs().max()
    print("trainable", trainable_count(model), "adapter tensors", len(adapter_state_dict(model)))


def test_cached_beam_search_equals_uncached():
    model = build()
    model.eval()
    s_on = 0
    for L in [m for m in model.modules() if hasattr(m, "log_stats")]:
        L.log_stats = True
    ids = torch.randint(3, 97, (2, 12)); mask = torch.ones_like(ids)
    ids[1, :4] = 0; mask[1, :4] = 0                              # left padding as in DoRA's evaluation
    gc = GenerationConfig(num_beams=4, do_sample=False, max_new_tokens=10, pad_token_id=0, eos_token_id=2)
    with torch.no_grad():
        with lena_stream(model):
            cached = model.generate(input_ids=ids, attention_mask=mask, generation_config=gc, use_cache=True)
        uncached = model.generate(input_ids=ids, attention_mask=mask, generation_config=gc, use_cache=False)
    s = [m.last["s"] for m in model.modules() if getattr(m, "last", None) is not None]
    frac = torch.cat([x.flatten() for x in s]).gt(0).float().mean().item()
    print("cached", cached.tolist(), "\nuncached", uncached.tolist(), "\nfraction s>0", frac)
    assert torch.equal(cached, uncached)
    assert frac > 0                                              # the fit is actually used


if __name__ == "__main__":
    test_train_step_left_padding(); test_cached_beam_search_equals_uncached(); print("ok")
