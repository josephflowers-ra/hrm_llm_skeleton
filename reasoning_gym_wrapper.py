
import random
from torch.utils.data import Dataset

try:
    import reasoning_gym as rg
except Exception:
    rg=None

class ToyAddDataset(Dataset):
    def __init__(self,n=1000,max_n=999):
        self.data=[]
        for _ in range(n):
            a=random.randint(0,max_n); b=random.randint(0,max_n)
            self.data.append((f"Add the numbers:\nA: {a}\nB: {b}\nAnswer:", str(a+b)))
    def __len__(self): return len(self.data)
    def __getitem__(self,i):
        p,t=self.data[i]
        return {"prompt":p,"target":t,"verify": lambda out: out.strip()==t}

def build_reasoning_dataset(name="toy_add",split="train",n=2000):
    if rg is None or name=="toy_add": return ToyAddDataset(n=n)
    if name=="maze":
        env=rg.environments.Maze(); return rg.datasets.TextGenerationDataset(env=env,split=split,n=n)
    if name=="arithmetic":
        env=rg.environments.Arithmetic(); return rg.datasets.TextGenerationDataset(env=env,split=split,n=n)
    return ToyAddDataset(n=n)
