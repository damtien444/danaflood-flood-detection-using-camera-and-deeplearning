import os

import torch
from config import model, trace_folder, ENCODER

dummy_input = torch.rand(3, 3, 512, 512)
model.eval()
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save(trace_folder+os.sep+ENCODER+".pt")


