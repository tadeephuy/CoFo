import torch
import numpy as np

def extract_ampl_phase(fft_im):
    # fft_im: size should be bx3xhxwx2
    fft_amp = fft_im[:,:,:,:,0]**2 + fft_im[:,:,:,:,1]**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2( fft_im[:,:,:,:,1], fft_im[:,:,:,:,0] )
    return fft_amp, fft_pha

def low_freq_mutate( amp_src, amp_trg, L=0.1 ):
    _, _, h, w = amp_src.size()
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)     # get b
    amp_src[:,:,0:b,0:b]     = amp_trg[:,:,0:b,0:b]      # top left
    amp_src[:,:,0:b,w-b:w]   = amp_trg[:,:,0:b,w-b:w]    # top right
    amp_src[:,:,h-b:h,0:b]   = amp_trg[:,:,h-b:h,0:b]    # bottom left
    amp_src[:,:,h-b:h,w-b:w] = amp_trg[:,:,h-b:h,w-b:w]  # bottom right
    return amp_src

@torch.no_grad()
def create_fda_batch(fr, to, beta):
    # fr->to batch với FDA theo beta (style:fr, content:to)
    fft_src = torch.fft.fft2(to.clone()) 
    fft_trg = torch.fft.fft2(fr.clone())
    amp_src, pha_src = torch.abs(fft_src), torch.angle(fft_src)
    amp_trg, pha_trg = torch.abs(fft_trg), torch.angle(fft_trg)
    amp_src_ = low_freq_mutate( amp_src.clone(), amp_trg.clone(), L=beta )
    fft_src_ = amp_src_ * torch.exp( 1j * pha_src )
    src_in_trg = torch.fft.ifft2( fft_src_, dim=(-2, -1))
    src_in_trg = torch.real(src_in_trg)
    return src_in_trg