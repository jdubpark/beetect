import torch

stats = torch.cuda.memory_stats(device='cuda')
# print(stats)

summary = torch.cuda.memory_summary(device='cuda')
print(summary)

torch.cuda.empty_cache()
