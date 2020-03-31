import torch
import torch.nn as nn
from itertools import permutations
EPS = 1e-8


class ConvTasCriterion(nn.Module):
    def __init__(self):
        super(ConvTasCriterion, self).__init__()

    def forward(self, outputs, sources):
        assert outputs.size() == sources.size()
        B, C, T = sources.size()

        # Zero-mean norm
        mean_target = torch.sum(sources, dim=2, keepdim=True) / T
        mean_outputs = torch.sum(outputs, dim=2, keepdim=True) / T
        zero_mean_target = sources - mean_target
        zero_mean_estimate = outputs - mean_outputs

        # Step 2. SI-SNR with PIT
        # reshape to use broadcast
        s_target = torch.unsqueeze(zero_mean_target, dim=1)  # [B, 1, C, T]
        s_estimate = torch.unsqueeze(zero_mean_estimate, dim=2)  # [B, C, 1, T]
        # s_target = <s', s>s / ||s||^2
        pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)  # [B, C, C, 1]
        s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + EPS  # [B, 1, C, 1]
        pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, C, T]
        # e_noise = s' - s_target
        e_noise = s_estimate - pair_wise_proj  # [B, C, C, T]
        # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
        pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + EPS)
        pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C, C]

        # Get max_snr of each utterance
        # permutations, [C!, C]
        perms = sources.new_tensor(list(permutations(range(C))), dtype=torch.long)
        # one-hot, [C!, C, C]
        index = torch.unsqueeze(perms, 2)
        perms_one_hot = sources.new_zeros((*perms.size(), C)).scatter_(2, index, 1)
        # [B, C!] <- [B, C, C] einsum [C!, C, C], SI-SNR sum of each permutation
        snr_set = torch.einsum('bij,pij->bp', [pair_wise_si_snr, perms_one_hot])
        max_snr_idx = torch.argmax(snr_set, dim=1)  # [B]
        # max_snr = torch.gather(snr_set, 1, max_snr_idx.view(-1, 1))  # [B, 1]
        max_snr, _ = torch.max(snr_set, dim=1, keepdim=True)
        max_snr /= C
        loss = 0 - torch.mean(max_snr)
        return loss

