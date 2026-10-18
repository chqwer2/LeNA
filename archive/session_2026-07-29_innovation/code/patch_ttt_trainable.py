import ast
F = "Llama_Adaptation.py"
s = open(F).read()

# 1) mark_trainable attr tuple: add "ttt" so TTTHead params stay trainable
old_tuple = 'for attr in ("A", "B", "act", "gate", "steer", "outgate", "gate_after_a", "gate_after_b", "lena_A", "lena_B", "lena_act"):'
new_tuple = 'for attr in ("A", "B", "act", "gate", "steer", "outgate", "ttt", "gate_after_a", "gate_after_b", "lena_A", "lena_B", "lena_act"):'
if new_tuple not in s:
    assert old_tuple in s, "attr tuple anchor missing"
    s = s.replace(old_tuple, new_tuple, 1)
    print("ttt added to mark_trainable tuple")
else:
    print("ttt already in tuple")

# 2) grouped optimizer: add an elif branch for .ttt. (nodecay, base_lr) -- mirrors steer, keeps
#    the if/elif chain intact (each param -> exactly one group).
anchor = '''        if ".steer." in n:
            groups["adapter_nodecay"].append(p)   # steer: base_lr, no weight-decay
        elif is_act(n):'''
new = '''        if ".steer." in n:
            groups["adapter_nodecay"].append(p)   # steer: base_lr, no weight-decay
        elif ".ttt." in n:
            groups["adapter_nodecay"].append(p)   # ttt: base_lr, no weight-decay
        elif is_act(n):'''
if '".ttt." in n' not in s:
    assert anchor in s, "steer/elif optim-group anchor missing"
    s = s.replace(anchor, new, 1)
    print("ttt optim nodecay elif added")
else:
    print("ttt optim group already present")

open(F, "w").write(s)
ast.parse(open(F).read())
print("syntax OK")
