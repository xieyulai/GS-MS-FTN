import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummaryX import summary


class GeneratorFCOS(nn.Module):

    def __init__(self, d_model_c, voc_size):
        super(GeneratorFCOS, self).__init__()
        self.linear = nn.Linear(d_model_c, voc_size)      # 300-->10172/1024-->10172
        print('Using vanilla Generator')

    def forward(self, x):
        '''
        Inputs:
            x: (B, Sc, Dc)
        Outputs:
            (B, seq_len, voc_size)
        '''
        x = self.linear(x)
        return F.log_softmax(x, dim=-1)


if __name__ == '__main__':
    generator = GeneratorFCOS(
                    d_model_c=300,
                    voc_size=10172,
                )
    c = torch.randn(2,30,300)

    c_score = generator(c)

    print('每个词向量的得分:', c_score.shape)

    print(summary(generator, c))
