import sys
import torch
import inspect
import functools
import numpy as np
import pandas as pd
import torch.nn as nn
from enum import Enum
import torch.nn.functional as F
from numpy import array, ndarray
from torch import as_tensor,Tensor
from collections.abc import Iterable, Generator
from torch.nn.init import kaiming_uniform_,normal_
from torch.nn.utils import weight_norm, spectral_norm

NormType = Enum('NormType', 'Batch BatchZero Weight Spectral Instance InstanceZero')

class Defaults:
    def __init__(self):
        self.activation = nn.ReLU
defaults = Defaults()

def delegates(to=None, keep=False, but=None):
    "Decorator: replace `**kwargs` in signature with params from `to`"
    if but is None: but = []
    def _f(f):
        if to is None: to_f,from_f = f.__base__.__init__,f.__init__
        else:          to_f,from_f = to.__init__ if isinstance(to,type) else to,f
        from_f = getattr(from_f,'__func__',from_f)
        to_f = getattr(to_f,'__func__',to_f)
        if hasattr(from_f,'__delwrap__'): return f
        sig = inspect.signature(from_f)
        sigd = dict(sig.parameters)
        k = sigd.pop('kwargs')
        s2 = {k:v for k,v in inspect.signature(to_f).parameters.items()
              if v.default != inspect.Parameter.empty and k not in sigd and k not in but}
        sigd.update(s2)
        if keep: sigd['kwargs'] = k
        else: from_f.__delwrap__ = to_f
        from_f.__signature__ = sig.replace(parameters=sigd.values())
        return f
    return _f

def _mk_param(n,d=None): return inspect.Parameter(n, inspect.Parameter.KEYWORD_ONLY, default=d)

def use_kwargs_dict(keep=False, **kwargs):
    "Decorator: replace `**kwargs` in signature with `names` params"
    def _f(f):
        sig = inspect.signature(f)
        sigd = dict(sig.parameters)
        k = sigd.pop('kwargs')
        s2 = {n:_mk_param(n,d) for n,d in kwargs.items() if n not in sigd}
        sigd.update(s2)
        if keep: sigd['kwargs'] = k
        f.__signature__ = sig.replace(parameters=sigd.values())
        return f
    return _f

def _conv_func(ndim=2, transpose=False):
    "Return the proper conv `ndim` function, potentially `transposed`."
    assert 1 <= ndim <=3
    return getattr(nn, f'Conv{"Transpose" if transpose else ""}{ndim}d')

def init_linear(m, act_func=None, init='auto', bias_std=0.01):
    if getattr(m,'bias',None) is not None and bias_std is not None:
        if bias_std != 0: normal_(m.bias, 0, bias_std)
        else: m.bias.data.zero_()
    if init=='auto':
        if act_func in (F.relu_,F.leaky_relu_): init = kaiming_uniform_
        else: init = getattr(act_func.__class__, '__default_init__', None)
        if init is None: init = getattr(act_func, '__default_init__', None)
    if init is not None: init(m.weight)

@delegates(nn.BatchNorm2d)
def BatchNorm(nf, ndim=2, norm_type=NormType.Batch, **kwargs):
    "BatchNorm layer with `nf` features and `ndim` initialized depending on `norm_type`."
    return _get_norm('BatchNorm', nf, ndim, zero=norm_type==NormType.BatchZero, **kwargs)

@delegates(nn.InstanceNorm2d)
def InstanceNorm(nf, ndim=2, norm_type=NormType.Instance, affine=True, **kwargs):
    "InstanceNorm layer with `nf` features and `ndim` initialized depending on `norm_type`."
    return _get_norm('InstanceNorm', nf, ndim, zero=norm_type==NormType.InstanceZero, affine=affine, **kwargs)

def _get_norm(prefix, nf, ndim=2, zero=False, **kwargs):
    "Norm layer with `nf` features and `ndim` initialized depending on `norm_type`."
    assert 1 <= ndim <= 3
    bn = getattr(nn, f"{prefix}{ndim}d")(nf, **kwargs)
    if bn.affine:
        bn.bias.data.fill_(1e-3)
        bn.weight.data.fill_(0. if zero else 1.)
    return bn

class PrePostInitMeta(type):
    "A metaclass that calls optional `__pre_init__` and `__post_init__` methods"
    def __new__(cls, name, bases, dct):
        x = super().__new__(cls, name, bases, dct)
        old_init = x.__init__
        def _pass(self): pass
        @functools.wraps(old_init)
        def _init(self,*args,**kwargs):
            self.__pre_init__()
            old_init(self, *args,**kwargs)
            self.__post_init__()
        x.__init__ = _init
        if not hasattr(x,'__pre_init__'):  x.__pre_init__  = _pass
        if not hasattr(x,'__post_init__'): x.__post_init__ = _pass
        return x

class Module(nn.Module, metaclass=PrePostInitMeta):
    "Same as `nn.Module`, but no need for subclasses to call `super().__init__`"
    def __pre_init__(self, *args, **kwargs): super().__init__()
    def __init__(self): pass

class ConvLayer(nn.Sequential):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and `norm_type` layers."
    @delegates(nn.Conv2d)
    def __init__(self, ni, nf, ks=3, stride=1, padding=None, bias=None, ndim=2, norm_type=NormType.Batch, bn_1st=True,
                 act_cls=defaults.activation, transpose=False, init='auto', xtra=None, bias_std=0.01, **kwargs):
        if padding is None: padding = ((ks-1)//2 if not transpose else 0)
        bn = norm_type in (NormType.Batch, NormType.BatchZero)
        inn = norm_type in (NormType.Instance, NormType.InstanceZero)
        if bias is None: bias = not (bn or inn)
        conv_func = _conv_func(ndim, transpose=transpose)
        conv = conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding, **kwargs)
        act = None if act_cls is None else act_cls()
        init_linear(conv, act, init=init, bias_std=bias_std)
        if   norm_type==NormType.Weight:   conv = weight_norm(conv)
        elif norm_type==NormType.Spectral: conv = spectral_norm(conv)
        layers = [conv]
        act_bn = []
        if act is not None: act_bn.append(act)
        if bn: act_bn.append(BatchNorm(nf, norm_type=norm_type, ndim=ndim))
        if inn: act_bn.append(InstanceNorm(nf, norm_type=norm_type, ndim=ndim))
        if bn_1st: act_bn.reverse()
        layers += act_bn
        if xtra: layers.append(xtra)
        super().__init__(*layers)

def _array2tensor(x):
    if x.dtype==np.uint16: x = x.astype(np.float32)
    # windows default numpy int dytpe is int32, while torch tensor default int dtype is int64
    # https://github.com/numpy/numpy/issues/9464
    if sys.platform == "win32":
        if x.dtype==np.int: x = x.astype(np.int64)
    return torch.from_numpy(x)

def is_iter(o):
    "Test whether `o` can be used in a `for` loop"
    #Rank 0 tensors in PyTorch are not really iterable
    return isinstance(o, (Iterable,Generator)) and getattr(o,'ndim',1)

@use_kwargs_dict(dtype=None, device=None, requires_grad=False, pin_memory=False)
def tensor(x, *rest, **kwargs):
    "Like `torch.as_tensor`, but handle lists too, and can pass multiple vector elements directly."
    if len(rest): x = (x,)+rest
    # There was a Pytorch bug in dataloader using num_workers>0. Haven't confirmed if fixed
    # if isinstance(x, (tuple,list)) and len(x)==0: return tensor(0)
    res = (x if isinstance(x, Tensor)
           else torch.tensor(x, **kwargs) if isinstance(x, (tuple,list))
           else _array2tensor(x) if isinstance(x, ndarray)
           else as_tensor(x.values, **kwargs) if isinstance(x, (pd.Series, pd.DataFrame))
           else as_tensor(x, **kwargs) if hasattr(x, '__array__') or is_iter(x)
           else _array2tensor(array(x), **kwargs))
    if res.dtype is torch.float64: return res.float()
    return res

class PooledSelfAttention2d(Module):
    "Pooled self attention layer for 2d."
    def __init__(self, n_channels):
        self.n_channels = n_channels
        self.theta,self.phi,self.g = [self._conv(n_channels, c) for c in (n_channels//8,n_channels//8,n_channels//2)]
        self.o   = self._conv(n_channels//2, n_channels)
        self.gamma = nn.Parameter(tensor([0.]))

    def _conv(self,n_in,n_out):
        return ConvLayer(n_in, n_out, ks=1, norm_type=NormType.Spectral, act_cls=None, bias=False)

    def forward(self, x):
        n_ftrs = x.shape[2]*x.shape[3]
        f = self.theta(x).view(-1, self.n_channels//8, n_ftrs)
        g = F.max_pool2d(self.phi(x),   [2,2]).view(-1, self.n_channels//8, n_ftrs//4)
        h = F.max_pool2d(self.g(x),     [2,2]).view(-1, self.n_channels//2, n_ftrs//4)
        
        beta = F.softmax(torch.bmm(f.transpose(1, 2), g), -1)
        o = self.o(torch.bmm(h, beta.transpose(1,2)).view(-1, self.n_channels//2, x.shape[2], x.shape[3]))
        return self.gamma * o + x

class AdaptiveConcatPool(nn.Module):
    def __init__(self, pool=1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(pool)
        self.max_pool = nn.AdaptiveMaxPool2d(pool)
    def forward(self, x):
        x = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)
        return x


from timm import create_model
import re
def _is_pool_type(l): return re.search(r'Pool[123]d$', l.__class__.__name__)
def has_pool_type(m):
    "Return `True` if `m` is a pooling layer or has one in its children"
    if _is_pool_type(m): return True
    for l in m.children():
        if has_pool_type(l): return True
    return False
def create_timm_body(arch:str, pretrained=True, cut=None):
    model = create_model(arch, pretrained=pretrained)
    if cut is None:
        ll = list(enumerate(model.children()))
        cut = next(i for i,o in reversed(ll) if has_pool_type(o))
    if isinstance(cut, int): return nn.Sequential(*list(model.children())[:cut])
    elif callable(cut): return cut(model)
    else: raise NamedError("cut must be either integer or function")

def create_effnet(pretrained, name):
    return create_timm_body(name, pretrained=pretrained)

def icnr_init(x, scale=2, init=nn.init.kaiming_normal_):
    "ICNR init of `x`, with `scale` and `init` function"
    ni,nf,h,w = x.shape
    ni2 = int(ni/(scale**2))
    k = init(x.new_zeros([ni2,nf,h,w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    return k.contiguous().view([nf,ni,h,w]).transpose(0, 1)

class PixelShuffle_ICNR(nn.Sequential):
    "Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`."
    def __init__(self, ni, nf=None, scale=2, blur=False, norm_type=NormType.Weight, act_cls=defaults.activation):
        super().__init__()
        nf = ifnone(nf, ni)
        layers = [ConvLayer(ni, nf*(scale**2), ks=1, norm_type=norm_type, act_cls=act_cls, bias_std=0),
                  nn.PixelShuffle(scale)]
        if norm_type == NormType.Weight:
            layers[0][0].weight_v.data.copy_(icnr_init(layers[0][0].weight_v.data))
            layers[0][0].weight_g.data.copy_(((layers[0][0].weight_v.data**2).sum(dim=[1,2,3])**0.5)[:,None,None,None])
        else:
            layers[0][0].weight.data.copy_(icnr_init(layers[0][0].weight.data))
        if blur: layers += [nn.ReplicationPad2d((1,0,1,0)), nn.AvgPool2d(2, stride=1)]
        super().__init__(*layers)

ifnone = lambda x,y: y if x is None else x