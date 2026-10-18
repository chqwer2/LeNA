import ast
F="Llama_Adaptation.py"; s=open(F).read()
old='for attr in ("A", "B", "act", "gate", "steer", "gate_after_a", "gate_after_b", "lena_A", "lena_B", "lena_act"):'
new='for attr in ("A", "B", "act", "gate", "steer", "outgate", "gate_after_a", "gate_after_b", "lena_A", "lena_B", "lena_act"):'
if '"outgate"' not in s:
    assert old in s, "trainable attr anchor missing"; s=s.replace(old,new,1); open(F,"w").write(s); print("outgate added to trainable")
else: print("already present")
ast.parse(open(F).read()); print("syntax OK")
