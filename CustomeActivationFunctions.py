import torch
from torch import nn


# create a class wrapper from PyTorch nn.Module, so
# the function now can be easily used in models
class HalfSizeSigmoid(nn.Module):
    '''
    Applies the Half Size Sigmoid (HSSi) function element-wise:
        HSSi(x) = 0.5 * sigmoid(x)
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    References:
        -  Related paper:
        original
    Examples:
        >>> m = halfSizeSigmoid()
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__() # init the base class

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return self.halfSizeSigmoid(input) # simply apply already implemented SiLU

    # simply define a silu function
    def halfSizeSigmoid(self, input):
        '''
        Applies the Half Size Sigmoid (HSSi) function element-wise:
            HSSi(x) = 0.5 * sigmoid(x)
        '''
        return torch.tensor(0.5) * torch.sigmoid(input)  # use torch.sigmoid to make sure that we created the most efficient implemetation based on builtin PyTorch functions


# create a class wrapper from PyTorch nn.Module, so
# the function now can be easily used in models
class QuarterSizeSigmoid(nn.Module):
    '''
    Applies the Quarter Size Sigmoid (QSSi) function element-wise:
        QSSi(x) = 0.25 * sigmoid(x)
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    References:
        -  Related paper:
        original
    Examples:
        >>> m = quarterSizeSigmoid()
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__() # init the base class

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return self.quarterSizeSigmoid(input) # simply apply already implemented SiLU

    # simply define a silu function
    def quarterSizeSigmoid(self, input):
        '''
        Applies the Quarter Size Sigmoid (QSSi) function element-wise:
            QSSi(x) = 0.25 * sigmoid(x)
        '''
        # return torch.tensor(0.0625) * torch.sigmoid(torch.tensor(1.0)*input)  # use torch.sigmoid to make sure that we created the most efficient implemetation based on builtin PyTorch functions
        # return torch.tensor(0.0625) * torch.sigmoid(torch.tensor(2.0)*input)  # use torch.sigmoid to make sure that we created the most efficient implemetation based on builtin PyTorch functions
        # return torch.tensor(0.03125) * torch.sigmoid(torch.tensor(2.0)*input)  # use torch.sigmoid to make sure that we created the most efficient implemetation based on builtin PyTorch functions
        return torch.tensor(0.0625) * torch.sigmoid(torch.tensor(3.0)*input)  # use torch.sigmoid to make sure that we created the most efficient implemetation based on builtin PyTorch functions


class SoftQuarterSizeSigmoid(nn.Module):
    '''
    Applies the Quarter Size Sigmoid (QSSi) function element-wise:
        QSSi(x) = 0.25 * sigmoid(x)
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    References:
        -  Related paper:
        original
    Examples:
        >>> m = softQuarterSizeSigmoid()
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__() # init the base class

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return self.softQuarterSizeSigmoid(input) # simply apply already implemented SiLU

    # simply define a silu function
    def softQuarterSizeSigmoid(self, input):
        '''
        Applies the Quarter Size Sigmoid (QSSi) function element-wise:
            QSSi(x) = 0.25 * sigmoid(x)
        '''
        return torch.tensor(0.25) * torch.sigmoid(input*0.5)  # use torch.sigmoid to make sure that we created the most efficient implemetation based on builtin PyTorch functions
