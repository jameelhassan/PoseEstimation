import torch
import torch.backends.cudnn
from config import INPLANES, NUM_FEATS, MODEL_TAG, GHOST

if GHOST:
    from stacked_hourglass.ghostnet import hg1, hg2, hg8
    from stacked_hourglass.predictor import HumanPosePredictor
else:
    from stacked_hourglass import hg1, hg2, hg8

from pthflops import count_ops

if torch.cuda.is_available():
    device = torch.device('cuda', torch.cuda.current_device())
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

print("""---------------- [ERROR] CANT RUN HERE!!!!!!! -----------------
GFLOP test wont work in main branch due to perceptual loss. Switch to branch hg1-3 and run it""")

arch = "hg2"
if arch == 'hg1':
    model = hg1(pretrained=False)
elif arch == 'hg2':
    model = hg2(pretrained=False)
# elif arch == 'hg3':
#     model = hg3(pretrained=False)
elif arch == 'hg8':
    model = hg8(pretrained=False)
else:
    raise Exception('unrecognised model architecture')
model = model.to(device)

# Create a network and a corresponding input
inp = torch.rand(1,3,256,256).to(device)

# Count the number of FLOPs
count_ops(model, inp)
