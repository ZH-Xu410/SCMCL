import datetime
import os
import time

import numpy as np
import torch
from munch import Munch
from torch.backends import cudnn

from manipulator.checkpoint.checkpoint import CheckpointIO
from manipulator.data.scmcl_dataset import InputFetcher, get_train_loader
from manipulator.models.model import create_model
from manipulator.options.train_options import TrainOptions
from manipulator.util import util
from manipulator.util.logger import StarganV2Logger
from similarity.models import ExprEncoder, ScaleLayer
from similarity.cross_attention import CrossAttentionSeq

def save_checkpoint(epoch):
    for ckptio in ckptios:
        ckptio.save(epoch)


def load_checkpoint(epoch):
    for ckptio in ckptios:
        ckptio.load(epoch)


def reset_grad():
    for optim in optims.values():
        optim.zero_grad()


def get_lr(net, opt):
    if net in ['generator', 'style_encoder', 'discriminator']:
        return opt.lr
    # elif net == 'discriminator':
    #     return opt.lr * 0.8
    elif net == 'mapping_network':
        return opt.f_lr


def compute_d_loss(nets, opt, x_real, y_org, y_trg, x_ref, x_tgt=None, tgt_mask=None):
    out = nets.discriminator(x_real, y_org)
    loss_real = adv_loss(out, 1)
    if x_tgt is not None and tgt_mask.sum() > 0:
        x_tgt_ = x_tgt[tgt_mask > 0]
        y_trg_ = y_trg[tgt_mask > 0]
        out_= nets.discriminator(x_tgt_, y_trg_)
        loss_real = (loss_real + adv_loss(out_, 1)) * 0.5
    
    # with fake images
    with torch.no_grad():
        s_trg = nets.style_encoder(x_ref)
        x_fake = nets.generator(x_real, s_trg)
        if x_tgt is not None and tgt_mask.sum() > 0:
            s_trg_ = s_trg[tgt_mask > 0]
            x_fake_ = nets.generator(x_tgt_, s_trg_)
            x_fake = torch.cat([x_fake, x_fake_], dim=0)
            y_trg = torch.cat([y_trg, y_trg_], dim=0)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake

    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item())


def compute_g_loss(nets, opt, x_real, y_org, y_trg, x_ref, x_tgt=None, tgt_mask=None, dists=None, inter_tgt=None, inter_dists=None):
    # adversarial loss
    s_trg = nets.style_encoder(x_ref)

    x_fake = nets.generator(x_real, s_trg)
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out, 1)

    # the first param corresponds to the jaw opening (similar to lip distance)
    dist_real = x_real[:, :, 0]
    dist_fake = x_fake[:, :, 0]

    # mouth loss (Pearson Correlation Coefficient)
    v_real = dist_real - torch.mean(dist_real, dim=1, keepdim=True)
    v_fake = dist_fake - torch.mean(dist_fake, dim=1, keepdim=True)
    loss_mouth_f = 1 - torch.mean(torch.mean(v_real * v_fake, dim=1) * torch.rsqrt(
        torch.mean(v_real ** 2, dim=1)+1e-7) * torch.rsqrt(torch.mean(v_fake ** 2, dim=1)+1e-7))

    # style reconstruction loss
    s_pred = nets.style_encoder(x_fake)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg)) * opt.lambda_sty

    # cycle-consistency loss
    s_org = nets.style_encoder(x_real)
    x_rec = nets.generator(x_fake, s_org)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real)) * opt.lambda_cyc

    # mouth loss backward
    dist_rec = x_rec[:, :, 0]

    v_rec = dist_rec - torch.mean(dist_rec, dim=1, keepdim=True)
    loss_mouth_b = 1 - torch.mean(torch.mean(v_fake * v_rec, dim=1) * torch.rsqrt(
        torch.mean(v_fake ** 2, dim=1)+1e-7) * torch.rsqrt(torch.mean(v_rec ** 2, dim=1)+1e-7))

    loss_mouth = (loss_mouth_f + loss_mouth_b) * opt.lambda_mouth

    if x_tgt is not None and tgt_mask.sum() > 0:
        m = tgt_mask > 0
        x_tgt_ = x_tgt[m]
        s_tgt = nets.style_encoder(x_tgt_)
        loss_sty_ = torch.mean(torch.abs(s_pred[m] - s_tgt) + torch.abs(s_trg[m] - s_tgt)) / 2 * opt.lambda_sty
        loss_sty = (loss_sty + loss_sty_) * 0.5

        x_tgt_rec = nets.generator(x_tgt_, s_tgt)
        loss_cyc_ = torch.mean(torch.abs(x_tgt_rec - x_tgt_)) * opt.lambda_cyc
        loss_cyc = (loss_cyc + loss_cyc_) * 0.5

        e_real = x_tgt[:, :, 1:]
        e_fake = x_fake[:, :, 1:]
        loss_paired = 1 - (tgt_mask.view(-1, 1) * (e_real * e_fake).mean(-1) * torch.rsqrt(
            e_real.pow(2).mean(-1)+1e-7) * torch.rsqrt(e_fake.pow(2).mean(-1)+1e-7))
        
        if dists is None:
            sims = torch.ones((loss_paired.shape[0], 1), device=loss_paired.device)
        else:
            sims = (opt.dist_thresh / (1 + (dists + inter_dists) / 2)).unsqueeze(1)
        
        loss_paired = (sims * loss_paired).sum(0).mean() / tgt_mask.sum()
        
        if dists is not None:
            loss_sim = []
            tau = 0.1
            for i in range(x_real.shape[0]):
                if tgt_mask[i] == 0:
                    continue
                if (y_org[i] == 0) ^ (y_trg[i] == 0):
                    c = x_real.shape[-1]
                    vec_real_1 = nets.expr_encoder.encode(x_real[i, -1].view(1, 1, c))
                    vec_real_2 = nets.expr_encoder.encode(x_real[i, -2].view(1, 1, c))
                    vec_fake_1 = nets.expr_encoder.encode(x_fake[i, -1].view(1, 1, c))
                    vec_fake_2 = nets.expr_encoder.encode(x_fake[i, -2].view(1, 1, c))
                    act_real = nets.expr_encoder.proj(nets.expr_attn(vec_real_1, vec_real_2)).detach()
                    fused_vec_fake = nets.expr_attn(vec_fake_1, vec_fake_2)
                    act_fake = nets.expr_encoder.proj(fused_vec_fake)
                    expr_dist = torch.dist(act_fake, act_real, p=2)
                    loss_sim.append(torch.abs(nets.scale_layer(expr_dist) - dists[i]))
                    
                elif y_org[i] != y_trg[i]:
                    c = x_real.shape[-1]
                    vec_real_1 = nets.expr_encoder.encode(inter_tgt[i, -1].view(1, 1, c))
                    vec_real_2 = nets.expr_encoder.encode(inter_tgt[i, -2].view(1, 1, c))
                    vec_fake_1 = nets.expr_encoder.encode(x_fake[i, -1].view(1, 1, c))
                    vec_fake_2 = nets.expr_encoder.encode(x_fake[i, -2].view(1, 1, c))
                    act_real = nets.expr_encoder.proj(nets.expr_attn(vec_real_1, vec_real_2)).detach()
                    fused_vec_fake = nets.expr_attn(vec_fake_1, vec_fake_2)
                    act_fake = nets.expr_encoder.proj(fused_vec_fake)
                    expr_dist = torch.dist(act_fake, act_real, p=2)
                    loss_sim.append(torch.abs(nets.scale_layer(expr_dist) - inter_dists[i]))
            
            loss_sim = torch.stack(loss_sim).mean() if len(loss_sim) else x_real.new_zeros(1)
        else:
            loss_sim = x_real.new_zeros(1)
    else:
        loss_paired = x_real.new_zeros(1)
        loss_sim = x_real.new_zeros(1)

    loss = loss_adv + loss_sty + loss_cyc + loss_mouth + \
           loss_paired * opt.lambda_paired + loss_sim * opt.lambda_sim
    return loss, Munch(adv=loss_adv.item(),
                       sty=loss_sty.item(),
                       cyc=loss_cyc.item(),
                       mouth=loss_mouth.item(),
                       paired=loss_paired.item(),
                       sim=loss_sim.item())


def adv_loss(logits, target):
    """Implements LSGAN loss"""
    assert target in [1, 0]
    return torch.mean((logits - target)**2)


if __name__ == '__main__':

    opt = TrainOptions().parse()
    device = f'cuda:{opt.gpu_ids[0]}' if len(opt.gpu_ids) else 'cpu'

    cudnn.enabled = True
    cudnn.benchmark = True
    torch.manual_seed(777)
    np.random.seed(777)

    # initialize train dataset
    loader_src = get_train_loader(opt, which='source', prefix='MEAD')
    loader_ref = get_train_loader(opt, which='reference', prefix='MEAD')

    # initialize models
    nets = create_model(opt)

    # print network params and initialize them
    for name, module in nets.items():
        util.print_network(module, name)
        print('Initializing %s...' % name)
        module.apply(util.he_init)

    # set optimizers
    optims = Munch()
    for net in nets.keys():
        optims[net] = torch.optim.Adam(params=nets[net].parameters(
        ), lr=get_lr(net, opt), betas=[opt.beta1, opt.beta2])

    ckptios = [CheckpointIO(os.path.join(opt.checkpoints_dir, '{:02d}_nets.pth'), opt, len(opt.gpu_ids) > 0, **nets),
               CheckpointIO(os.path.join(opt.checkpoints_dir, '{:02d}_optims.pth'), opt, False, **optims)]

    nets.expr_encoder = ExprEncoder().to(device).eval()
    nets.scale_layer = ScaleLayer().to(device).eval()
    nets.expr_attn = CrossAttentionSeq(nets.expr_encoder.feat_dim, nets.expr_encoder.dim, 1, heads=4).to(device).eval()
    ckpt = torch.load(opt.encoder_ckpt, map_location=device)['model']
    nets.expr_encoder.load_state_dict(ckpt['expr_encoder'])
    nets.scale_layer.load_state_dict(ckpt['scale_layer'])
    nets.expr_attn.load_state_dict(ckpt['expr_attn'])

    # create logger
    logger = StarganV2Logger(opt.checkpoints_dir)

    # Training loop
    if opt.finetune:
        # load nets if finetuning
        ckptios[0].load(opt.finetune_epoch)
        for ckptio in ckptios:
            ckptio.fname_template = ckptio.fname_template.replace(
                '.pth', '_finetuned.pth')
    else:
        # resume training if necessary
        if opt.resume_epoch > 0:
            load_checkpoint(opt.resume_epoch)

    loss_log = os.path.join(opt.checkpoints_dir, 'loss_log.txt')
    logfile = open(loss_log, "a")

    fetcher = InputFetcher(loader_src, loader_ref, pseudo=True)

    print('Start training...')
    start_time = time.time()
    for epoch in range(0 if opt.finetune else opt.resume_epoch, opt.niter):
        for model in nets:
            nets[model].train()

        for i in range(len(loader_src)):

            # fetch sequences and labels
            inputs = next(fetcher)
            x_real, y_org = inputs.x_src, inputs.y_src
            x_ref, y_trg = inputs.x_ref, inputs.y_ref
            x_tgt, tgt_mask = inputs.x_tgt, inputs.tgt_mask
            inter_tgt = inputs.inter_tgt
            dists, inter_dists = inputs.dists, inputs.inter_dists

            x_real = x_real.to(device)
            y_org = y_org.to(device)
            x_ref = x_ref.to(device)
            y_trg = y_trg.to(device)
            x_tgt = x_tgt.to(device)
            tgt_mask = tgt_mask.to(device)
            dists = dists.to(device)
            inter_tgt = inter_tgt.to(device)
            inter_dists = inter_dists.to(device)

            d_loss, d_losses_ref = compute_d_loss(
                nets, opt, x_real, y_org, y_trg, x_ref, x_tgt=x_tgt, tgt_mask=tgt_mask)
            reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            g_loss, g_losses_ref = compute_g_loss(
                nets, opt, x_real, y_org, y_trg, x_ref, x_tgt=x_tgt, tgt_mask=tgt_mask, dists=dists, inter_tgt=inter_tgt, inter_dists=inter_dists)
            reset_grad()
            g_loss.backward()
            optims.generator.step()
            optims.style_encoder.step()

            iteration = i + epoch*len(loader_src)
            # print out log info
            if (i+1) % opt.print_freq == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Epoch [%i/%i], Iteration [%i/%i], " % (
                    elapsed, epoch+1, opt.niter, i+1, len(loader_src))
                all_losses = dict()
                for loss, prefix in zip([d_losses_ref, g_losses_ref],
                                        ['D/', 'G/']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                        logger.log_training(
                            "train/"+prefix+key, value, iteration)
                log += ' '.join(['%s: [%.4f]' % (key, value)
                                for key, value in all_losses.items()])

                print(log)
                logfile.write(log)
                logfile.write('\n')
                logfile.flush()
        
        # save model checkpoints
        if (epoch+1) % opt.save_epoch_freq == 0:
            save_checkpoint(epoch=epoch+1)

        # Decay learning rates.
        if (epoch+1) > (opt.niter - opt.niter_decay) and epoch != opt.niter - 1:
            log = 'Decayed learning rate'
            for net in nets.keys():
                if net not in optims:
                    continue
                lr_new = optims[net].param_groups[0]['lr'] - \
                    (get_lr(net, opt) / float(opt.niter_decay))
                for param_group in optims[net].param_groups:
                    param_group['lr'] = lr_new
                log += f', {net} {lr_new:g}'

            print(log)
            logfile.write(log)
            logfile.write('\n')
            logfile.flush()

    logfile.close()
