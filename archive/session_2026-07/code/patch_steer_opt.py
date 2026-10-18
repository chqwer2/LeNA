import ast
F = "Llama_Adaptation.py"
s = open(F).read()

# Fix 1: add "steer" to the trainable attr tuple in _mark_trainable
old = 'for attr in ("A", "B", "act", "gate", "gate_after_a", "gate_after_b", "lena_A", "lena_B", "lena_act"):'
new = 'for attr in ("A", "B", "act", "gate", "steer", "gate_after_a", "gate_after_b", "lena_A", "lena_B", "lena_act"):'
assert old in s, "mark_trainable anchor missing"
if new not in s:
    s = s.replace(old, new, 1)
    print("Fix1: steer added to trainable list")
else:
    print("Fix1: already present")

# Fix 2: give steer its own no-decay group in build_grouped_optimizer
anchor = "        if is_act(n):"
add = ('        if ".steer." in n:\n'
       '            groups["adapter_nodecay"].append(p)   # steer: base_lr, no weight-decay\n'
       '        elif is_act(n):')
if '".steer." in n' not in s:
    assert anchor in s, "grouping anchor missing"
    s = s.replace(anchor, add, 1)
    print("Fix2: steer no-decay group added")
else:
    print("Fix2: already present")

open(F, "w").write(s)
ast.parse(open(F).read())
print("syntax OK")
