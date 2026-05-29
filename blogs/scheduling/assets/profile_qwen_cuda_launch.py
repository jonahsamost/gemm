'''
initial run:
nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi --stats=true python run.py

display stats:
nsys stats --force-export=true -r cuda_api_sum report1.nsys-rep

~2500 kernel launches with ~3us overhead per launch equats to about 7.3ms overhead
'''
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

seq_len = 2048
model_name = "Qwen/Qwen3.5-0.8B"
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer("hello there!", return_tensors="pt").to("cuda")
# input_ids = torch.randint(0, 50_000, (1, seq_len), device='cuda')

model = torch.compile(model)

# warmup
with torch.no_grad():
    out = model(**inputs)
    torch.cuda.synchronize()

torch.cuda.cudart().cudaProfilerStart()
with torch.no_grad():
    out = model(**inputs)
    torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
