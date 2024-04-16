import torch
from torch import nn
from argparse import ArgumentParser
class MetaModel(nn.Module):
    
    
    def __init__(self, window_len, num_channels, num_classes) -> None:
        super(MetaModel, self).__init__()
        self.window_len = window_len
        self.num_channels = num_channels
        self.num_classes = num_classes
        
        self.mlp = nn.Linear(in_features=window_len*num_channels, out_features=num_classes)
    def forward(self, x):
        
        transformed_x = x.view(x.shape[0], self.window_len*self.num_channels)
        
        out = self.mlp(transformed_x)
        
        return out

model = MetaModel(window_len=4, num_channels=7, num_classes=10)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
loss_fn = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10)
parser = ArgumentParser()
args = parser.parse_args()

args.volunteer_split = "1,2|3|4"
args.dataset_name = "opportunity"
args.sliding_window_length = 64
args.sliding_window_step = 32
from utils import get_loader_by_volunteers
loaders = get_loader_by_volunteers(args)
print(loaders)

