import torch
import torch.nn as nn


class NoNorm(nn.Module):
    def __init__(self):
        super(NoNorm, self).__init__()

    def forward(self, x):
        return x


class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(BatchNorm, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        shape = (1, self.num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        
        self.register_buffer('running_mean',torch.zeros(shape))
        self.register_buffer('running_var',torch.zeros(shape))

        self.running_mean = self.running_mean.to(device)
        self.running_var = self.running_var.to(device)

    def forward(self, x):
        if self.training:
            n = x.numel() / x.size(1)
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * (n/(n-1)) * var

        else:
            mean = self.running_mean
            var = self.running_var

        x_normalized = self.gamma*(x - mean) / torch.sqrt(var + self.eps) + self.beta



        return x_normalized



class InstanceNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(InstanceNorm, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features



    def forward(self, x):
        N, C, H, W = x.shape

        assert C == self.num_features

        dimensions = (2, 3)
        mean = x.mean(dim=dimensions, keepdim=True)
        var = x.var(dim=dimensions, unbiased=False, keepdim=True)

        x_normalized = (x - mean) / torch.sqrt(var + self.eps)



        return x_normalized


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.num_features = num_features



    def forward(self, x):
        N, C, H, W = x.shape
        assert C == self.num_features


        dimensions = (1, 2, 3)
        mean = x.mean(dim=dimensions, keepdim=True)
        var = x.var(dim=dimensions, unbiased=False, keepdim=True)



        x_normalized = (x - mean) / torch.sqrt(var + torch.tensor(self.eps))


        return x_normalized


class GroupNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, group=4):
        super(GroupNorm, self).__init__()
        self.eps = eps
        self.num_features = num_features
        self.group = group

    def forward(self, x):


        N, C, H, W = x.shape

        x = x.view(N, self.group, int(C / self.group), H, W)
        dimensions = (2, 3, 4)
        
        mean = x.mean(dim=dimensions, keepdim=True)
        var = x.var(dim=dimensions, unbiased=False, keepdim=True)

        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        x_normalized = x_normalized.view(N, C, H, W)


        return x_normalized

class BatchInstanceNorm(nn.Module):
    def __init__(self, num_features, momentum = 0.1, eps=1e-5, rho=0.5, affine=True):
        super(BatchInstanceNorm, self).__init__()
        self.momentum = momentum
        self.running_mean = 0
        self.running_var = 0
        self.eps = torch.tensor(eps)
        self.num_features = num_features

        self.rho = rho
        shape = (1, self.num_features, 1, 1)


        self._param_init()


    def _param_init(self):
        nn.init.zeros_(self.beta)
        nn.init.ones_(self.gamma)

    
    def forward(self, x):
        if self.training:            
                
            n = x.numel() / x.size(1)
            dimensions = (0,2,3)
            var_bn = x.var(dim=dimensions, keepdim=True, unbiased=False)
            mean_bn = x.mean(dim=dimensions, keepdim=True)

            with torch.no_grad():
                
                self.running_mean = self.momentum * mean_bn + (1 - self.momentum) * self.running_mean
                self.running_var = self.momentum * (n/(n-1)) * var_bn + (1 - self.momentum) * self.running_var

        else:
            mean_bn = self.running_mean
            var_bn = self.running_var
        dn = torch.sqrt(var_bn + self.eps)
        x_bn = (x - mean_bn)/ dn
        dimensions = (2,3)
        mean_in = x.mean(dim=dimensions, keepdim=True)
        var_in = x.var(dim=dimensions, keepdim=True)
        dn = torch.sqrt(var_in + self.eps)
        x_in = (x - mean_in)/ dn

        x = self.rho * x_bn + (1-self.rho) * x_in



        return x




