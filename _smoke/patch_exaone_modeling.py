"""캐시의 모든 modeling_exaone.py를 transformers 5.x 호환으로 멱등 패치.
create_causal_mask(input_embeds=…, cache_position=…) → (inputs_embeds=…), cache_position 제거."""
import glob
import re

pat = glob.glob("/Users/swxvno/.cache/huggingface/modules/transformers_modules/*/*/*/modeling_exaone.py")
for p in pat:
    s = open(p, encoding="utf-8").read()
    if "create_causal_mask(" not in s:
        continue

    def fix(m):
        blk = m.group(0)
        blk = blk.replace("input_embeds=inputs_embeds", "inputs_embeds=inputs_embeds")
        blk = re.sub(r"\n\s*cache_position=cache_position,", "", blk)
        return blk

    s2 = re.sub(r"create_causal_mask\((.*?)\)", fix, s, flags=re.DOTALL, count=1)
    if s2 != s:
        open(p + ".orig.bak", "w", encoding="utf-8").write(s)
        open(p, "w", encoding="utf-8").write(s2)
        print("패치:", p)
    else:
        print("이미 패치됨/불필요:", p)
print("완료")
