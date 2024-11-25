import torch
from torch import nn
from argparse import ArgumentParser
import torch.autograd as autograd
import torch.nn.functional as F


def linear(inputs, weight, bias, meta_step_size=0.001, meta_loss=None):
    if meta_loss is not None:
        grad_weight = autograd.grad(meta_loss, weight, create_graph=True)[0]
        grad_bias = autograd.grad(meta_loss, bias, create_graph=True)[0]
        bias_adapt = bias - grad_bias * meta_step_size
        grad_adapt = weight - grad_weight * meta_step_size
        return F.linear(inputs, grad_adapt, bias_adapt)
    else:
        return F.linear(inputs, weight, bias)

class MetaModel(nn.Module):
    
    
    def __init__(self, window_len, num_channels, num_classes) -> None:
        super(MetaModel, self).__init__()
        self.window_len = window_len
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.meta_step_size = 0.001
        self.mlp = nn.Linear(in_features=window_len*num_channels, out_features=num_classes)
    def forward(self, x, meta_loss=None):
        
        transformed_x = x.view(x.shape[0], self.window_len*self.num_channels)
        out = linear(transformed_x, self.mlp.weight, self.mlp.bias, meta_step_size=self.meta_step_size, meta_loss=meta_loss)
        
        return out
        

model = MetaModel(window_len=64, num_channels=113, num_classes=18)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
loss_fn = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10)
parser = ArgumentParser()
args = parser.parse_args()
torch.manual_seed(3407)
args.volunteer_split = "1|2|3|4"
args.dataset_name = "opportunity"
args.sliding_window_length = 64
args.sliding_window_step = 32
args.num_volunteers = 4
args.num_classes = 18
args.num_channels = 113
args.batch_size = 128
from utils import get_loader_by_volunteers, weights_init, MetaSampler
from torch.utils.data import DataLoader
loaders = get_loader_by_volunteers(args)
train_loader = loaders[0]
ds = loaders[0].dataset
train_sampler = MetaSampler(ds, args.num_classes)
train_loader = DataLoader(ds, batch_size=args.batch_size,drop_last=True, sampler=train_sampler)
valid_loader = loaders[1:3]
test_loader = loaders[3]
epcohs = 5
meta_loops = 5
criterion = nn.CrossEntropyLoss()
meta_step_size= 0.001
weights_init(model)
for epoch in range(epcohs):
    model.train()
    model.cuda()
    best_acc_val = -1
    
    for meta_loop in range(meta_loops):
        
        index_val = 0 # 随机选择一个域中的数据
        n = len(train_loader)
        meta_train_loss, meta_val_loss = 0.0, 0.0
        
        for data in train_loader:
            
            x, y, _ = data
            x, y = x.float().cuda(), y.long().cuda() # +Variable
            y = y.squeeze(1)
            predict_y = model(x)
            
            train_loss = criterion(predict_y, y)
            meta_train_loss += train_loss
      
        meta_train_loss = meta_train_loss / n
        
        for i, data in enumerate(valid_loader[index_val]):
            x, y, _ = data
            x, y = x.float().cuda(), y.long().cuda()
            y = y.squeeze(1)
            predict_y = model(x, meta_loss=meta_train_loss)
            val_loss = criterion(predict_y, y)
            meta_val_loss += val_loss
            if i == 10:
                break
            
        # meta_val_loss = meta_val_loss / len(valid_loader[index_val])
        total_loss = meta_train_loss + meta_val_loss * 0.1
        
        optimizer.zero_grad()
        
        total_loss.backward()
        
        optimizer.step()
        
        print()
            

