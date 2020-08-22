import torch 
import torch.nn as nn 
import torch.nn.functional as F
IMAGE_SCALE = 2.0/255


def get_kernel(size, nsig, mode='gaussian', device='cuda:0'):
    if mode == 'gaussian':
        # since we have to normlize all the numbers 
        # there is no need to calculate the const number like \pi and \sigma.
        vec = torch.linspace(-nsig, nsig, steps=size).to(device)
        vec = torch.exp(-vec*vec/2)
        res = vec.view(-1, 1) @ vec.view(1, -1) 
        res = res / torch.sum(res)
    elif mode == 'linear':
        # originally, res[i][j] = (1-|i|/(k+1)) * (1-|j|/(k+1))
        # since we have to normalize it
        # calculate res[i][j] = (k+1-|i|)*(k+1-|j|)
        vec = (size+1)/2 - torch.abs(torch.arange(-(size+1)/2, (size+1)/2+1, step=1)).to(device)
        res = vec.view(-1, 1) @ vec.view(1, -1) 
        res = res / torch.sum(res)
    else:
        raise ValueError("no such mode in get_kernel.")
    return res


class NoOpAttacker():
    
    def attack(self, image, label, model):
        return image, -torch.ones_like(label)


class PGDAttacker():
    def __init__(self, num_iter, epsilon, step_size, kernel_size=15, prob_start_from_clean=0.0, translation=False, device='cuda:0'):
        step_size = max(step_size, epsilon / num_iter)
        self.num_iter = num_iter
        self.epsilon = epsilon * IMAGE_SCALE
        self.step_size = step_size*IMAGE_SCALE
        self.prob_start_from_clean = prob_start_from_clean
        self.device=device
        self.translation = translation
        if translation:
            # this is equivalent to deepth wise convolution
            # details can be found in the docs of Conv2d.
            # "When groups == in_channels and out_channels == K * in_channels, where K is a positive integer, this operation is also termed in literature as depthwise convolution."
            self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=kernel_size, stride=(kernel_size-1)//2, bias=False, groups=3).to(self.device)
            self.gkernel = get_kernel(kernel_size, nsig=3, device=self.device).to(self.device)
            self.conv.weight = self.gkernel
    
    def _create_random_target(self, label):
        label_offset = torch.randint_like(label, low=0, high=1000)
        return (label + label_offset) % 1000

    def attack(self, image_clean, label, model, original=False):
        if original:
            target_label = label
        else:
            target_label = self._create_random_target(label)
        lower_bound = torch.clamp(image_clean - self.epsilon, min=-1., max=1.)
        upper_bound = torch.clamp(image_clean + self.epsilon, min=-1., max=1.)

        ori_images = image_clean.clone().detach()

        init_start = torch.empty_like(image_clean).uniform_(-self.epsilon, self.epsilon)
        
        start_from_noise_index = (torch.randn([])>self.prob_start_from_clean).float() 
        start_adv = image_clean + start_from_noise_index * init_start

        adv = start_adv
        for i in range(self.num_iter):
            adv.requires_grad = True
            logits = model(adv)
            losses = F.cross_entropy(logits, target_label)
            g = torch.autograd.grad(losses, adv, 
                                    retain_graph=False, create_graph=False)[0]
            if self.translation:
                g = self.conv(g)
            if original:
                adv = adv + torch.sign(g)*self.step_size
            else:
                adv = adv - torch.sign(g) * self.step_size
            adv = torch.where(adv > lower_bound, adv, lower_bound)
            adv = torch.where(adv < upper_bound, adv, upper_bound).detach()
        
        return adv, target_label