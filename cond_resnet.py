'''pretty much taken from the torchvision resnet model'''

import torch.nn as nn
import torch.nn.functional as F

def build_entry_flow():
    net = nn.Sequential(nn.Conv1d(1, 3, 1),
                        nn.ReLU())
    return net


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )
        

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18_1D_connection(nn.Module):
    def __init__(self, channels, levels):
        super(ResNet18_1D, self).__init__()
        self.ch = channels # eval(args['model']['cond_net_channels'])
        self.res_lev = levels # len(eval(args['model']['inn_coupling_blocks']))
        self.in_planes = self.ch # in_planes correspond to channels
        
        # input layer and preprocessing
        self.conv1 = nn.Conv1d(1, self.ch[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(self.ch[0])
        self.relu = nn.LeakyReLU()
        self.preprocess_level = nn.Sequential(self.conv1, self.bn1, self.relu)
       
        # connection layers
        self.connection = [nn.Sequential(nn.Conv1d(i, 1, kernel_size=5, stride=4, padding=1, bias=False),
                                         nn.BatchNorm1d(1),
                                         self.relu
                                        )
                           for i in self.ch]

        self.layers = {}
        in_planes = self.ch[0]
        
        # hidden layers build:
        for i in range(self.res_lev):
            if i == 0:
                self.layers['layer'+ str(i)] = self._make_layer(BasicBlock, in_planes, self.ch[i],
                                                                num_blocks=2, stride=1)
            else:
                self.layers['layer' + str(i)] = self._make_layer(BasicBlock, in_planes, self.ch[i],
                                                                 num_blocks=2, stride=1) # stride 2 only here?
            in_planes = self.ch[i]

        # make resolution layers automatically depending on default/conf.ini
        
        self.connections = nn.ModuleList(self.connection)
        self.resolution_levels = nn.ModuleList([self.layers['layer'+str(i)] for i in range(self.res_lev)])
        # print('resnet18 resolution: ', self.resolution_levels)

    def _make_layer(self, block, in_planes, planes, num_blocks, stride):
        layers = [block(in_planes, planes, stride)]
        for i in range(num_blocks - 1):
            layers.append(block(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        '''Pass data through the conditional network. 
        Returns a list with the conditions c for each layer for the INN
        c has shape [batch_size, data_size]
        '''
        internal = [x]
        condition_for_each_layer = []
        for i, layer in enumerate(self.resolution_levels):
            if i == 0:
                out = self.preprocess_level(internal[-1])
                out = layer(out) # layer returns x which should be passed to the next layer, c is the condition for the respective INN layer
                internal.append(out)
                c = self.connection[i](out)
                condition_for_each_layer.append(c.squeeze())
            else:
                out = layer(internal[-1])
                internal.append(out)
                c = self.connection[i](out)
                condition_for_each_layer.append(c.squeeze())
        return condition_for_each_layer

    
    
class ResNet18_1D(nn.Module):
    def __init__(self, channels, levels):
        super(ResNet18_1D, self).__init__()
        self.ch = channels # eval(args['model']['cond_net_channels'])
        self.res_lev = levels # len(eval(args['model']['inn_coupling_blocks']))
        self.in_planes = self.ch # in_planes correspond to channels
        
        # input layer and preprocessing
        self.conv1 = nn.Conv1d(1, self.ch[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(self.ch[0])
        self.relu = nn.LeakyReLU()
        self.preprocess_level = nn.Sequential(self.conv1, self.bn1, self.relu) 
        self.layers = {}
        in_planes = self.ch[0]
        
        # hidden layers build:
        for i in range(self.res_lev):
            if i == 0:
                self.layers['layer'+ str(i)] = self._make_layer(BasicBlock, in_planes, self.ch[i],
                                                                num_blocks=2, stride=1)
            else:
                self.layers['layer' + str(i)] = self._make_layer(BasicBlock, in_planes, self.ch[i],
                                                                 num_blocks=2, stride=2)
            in_planes = self.ch[i]

        # make resolution layers automatically depending on default/conf.ini
        self.conv1x1 = nn.Conv1d(self.ch[-1], 1, kernel_size=1, stride=1, padding=1, bias=False)
        self.output_level = nn.Sequential(self.conv1x1, self.relu)
        self.resolution_levels = nn.ModuleList([self.layers['layer'+str(i)] for i in range(self.res_lev)])
#         self.resolution_levels.append(self.output_level)
        
    def _make_layer(self, block, in_planes, planes, num_blocks, stride):
        layers = [block(in_planes, planes, stride)]
        for i in range(num_blocks - 1):
            layers.append(block(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        outputs = [x]
        for i, layer in enumerate(self.resolution_levels):
            if i == 0:
                x = self.preprocess_level(x)
                outputs.append(layer(x))
            elif i == len(self.resolution_levels)-1:
                out = layer(outputs[-1])
                outputs.append(self.output_level(out))
            else:
                outputs.append(layer(outputs[-1]))
        return outputs[-1] ###
    
    
class ConvNet_1D_dense_output(nn.Module):
    def __init__(self, channels, levels):
        super().__init__()
        self.ch = channels # eval(args['model']['cond_net_channels'])
        self.res_lev = levels # len(eval(args['model']['inn_coupling_blocks']))
        
        # input layer and preprocessing
        self.conv1 = nn.Conv1d(1, self.ch[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(self.ch[0])
        self.relu = nn.LeakyReLU()
        self.preprocess_level = nn.Sequential(self.conv1, self.bn1, self.relu) 
        self.layers = {}
        in_planes = self.ch[0]
        
        # hidden layers build:
        for i in range(self.res_lev):
            if i == 0:
                self.layers['layer'+ str(i)] = self._make_layer(BasicBlock, in_planes, self.ch[i],
                                                                num_blocks=2, stride=1)
            else:
                self.layers['layer' + str(i)] = self._make_layer(BasicBlock, in_planes, self.ch[i],
                                                                 num_blocks=2, stride=2)
            in_planes = self.ch[i]

        # make resolution layers automatically depending on default/conf.ini
        self.conv1x1 = nn.Conv1d(self.ch[-1], 1, kernel_size=1, stride=1, padding=1, bias=False)
        self.out_dense = nn.Linear(52, 15)
        self.output_level = nn.Sequential(self.conv1x1, self.relu, self.out_dense, self.relu)
        self.resolution_levels = nn.ModuleList([self.layers['layer'+str(i)] for i in range(self.res_lev)])
#         self.resolution_levels.append(self.output_level)
        
    def _make_layer(self, block, in_planes, planes, num_blocks, stride):
        layers = [block(in_planes, planes, stride)]
        for i in range(num_blocks - 1):
            layers.append(block(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        outputs = [x]
        for i, layer in enumerate(self.resolution_levels):
            if i == 0:
                x = self.preprocess_level(x)
                outputs.append(layer(x))
            elif i == len(self.resolution_levels)-1:
                out = layer(outputs[-1])
                outputs.append(self.output_level(out))
            else:
                outputs.append(layer(outputs[-1]))
        return outputs[-1] ###
    
    
class ResNet_microcavity(nn.Module):
    def __init__(self, channels, levels):
        super(ResNet_microcavity, self).__init__()
        self.ch = channels # eval(args['model']['cond_net_channels'])
        self.res_lev = levels # len(eval(args['model']['inn_coupling_blocks']))
        self.in_planes = self.ch # in_planes correspond to channels
        
        # input layer and preprocessing
        self.conv1 = nn.Conv1d(1, self.ch[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(self.ch[0])
        self.relu = nn.LeakyReLU()
        self.preprocess_level = nn.Sequential(self.conv1, self.bn1, self.relu) 
        self.layers = {}
        in_planes = self.ch[0]
        
        # hidden layers build:
        for i in range(self.res_lev):
            if i == 0:
                self.layers['layer'+ str(i)] = self._make_layer(BasicBlock, in_planes, self.ch[i],
                                                                num_blocks=2, stride=2)
            else:
                self.layers['layer' + str(i)] = self._make_layer(BasicBlock, in_planes, self.ch[i],
                                                                 num_blocks=2, stride=2)
            in_planes = self.ch[i]

        # make resolution layers automatically depending on default/conf.ini
        self.conv1x1 = nn.Conv1d(self.ch[-1], 1, kernel_size=1, stride=1, padding=1, bias=False)
        self.out_dense = nn.Linear(52, 15)
        self.output_level = nn.Sequential(self.conv1x1, self.relu, self.out_dense, self.relu)
        self.resolution_levels = nn.ModuleList([self.layers['layer'+str(i)] for i in range(self.res_lev)])
#         self.resolution_levels.append(self.output_level)
        
    def _make_layer(self, block, in_planes, planes, num_blocks, stride):
        layers = [block(in_planes, planes, stride)]
        for i in range(num_blocks - 1):
            layers.append(block(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        outputs = [x]
        for i, layer in enumerate(self.resolution_levels):
            if i == 0:
                x = self.preprocess_level(x)
                outputs.append(layer(x))
            elif i == len(self.resolution_levels)-1:
                
                out = layer(outputs[-1])
                outputs.append(self.output_level(out))
            else:
                outputs.append(layer(outputs[-1]))
        return outputs[-1] ###