```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio
from pesq import pesq
from pystoi import stoi
from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE
import torchaudio.functional as F

#——————————计算SI-SNR，用于和模型估计的SI-SNR对照——————————
def si_snr(estimate, reference, epsilon=1e-8):
    estimate = estimate - estimate.mean()#去直流分量
    reference = reference - reference.mean()
    reference_pow = reference.pow(2).mean(axis=1, keepdim=True)#计算信号能量  .mean(axis=1, keepdim=True)：沿时间轴（假设 axis=1 是时间维度）求平均，保留原来的维度结构
    mix_pow = (estimate * reference).mean(axis=1, keepdim=True)#信号与参考信号的互相关
    scale = mix_pow / (reference_pow + epsilon) #优缩放因子 α
    reference = scale * reference
    error = estimate - reference

    reference_pow = reference.pow(2)
    error_pow = error.pow(2)

    reference_pow = reference_pow.mean(axis=1)
    error_pow = error_pow.mean(axis=1)

    si_snr = 10* torch.log10(reference_pow) -10 * torch.log10(error_pow)
    return si_snr.item()

# #——————————画波形/声谱图———————————
# def plot(waveform, title, sample_rate=16000):
#     wav_numpy = waveform.numpy()
#     sample_size = waveform.shape[1]
#     time_axis = torch.arange(0, sample_size) / sample_rate
#
#     figure, axes = plt.subplots(2,1)
#     axes[0].plot(time_axis, wav_numpy[0], linewidth=1)
#     axes[0].grid(True)
#     axes[1].specgram(wav_numpy[0], Fs=sample_rate)
#     figure.suptitle(title)

WAVEFORM_SPEECH, SAMPLE_RATE_SPEECH = torchaudio.load(r"D:\PythonProject\QoS\dataset\Speaker1_C_0.wav")
WAVEFORM_NOISE, SAMPLE_RATE_NOISE = torchaudio.load("Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo.wav")# 噪声

#单通道
print(WAVEFORM_SPEECH.shape)
print(WAVEFORM_NOISE.shape)
WAVEFORM_SPEECH = WAVEFORM_SPEECH.mean(dim=0,keepdim=True)
WAVEFORM_NOISE =WAVEFORM_NOISE[0:1, :]#取一通道
# WAVEFORM_DISTORTED = WAVEFORM_DISTORTED.mean(dim=0,keepdim=True)

#SQUIM仅支持16kHz，重采样到16k
if SAMPLE_RATE_SPEECH != 16000:
    WAVEFORM_SPEECH = F.resample(WAVEFORM_SPEECH, SAMPLE_RATE_SPEECH, 16000)
if SAMPLE_RATE_NOISE != 16000:
    WAVEFORM_NOISE = F.resample(WAVEFORM_NOISE, SAMPLE_RATE_NOISE, 16000)

#对齐长度（裁剪到相同的帧数）
min_len= min(WAVEFORM_SPEECH.shape[1], WAVEFORM_NOISE.shape[1])
WAVEFORM_SPEECH = WAVEFORM_SPEECH[:, :min_len]
WAVEFORM_NOISE = WAVEFORM_NOISE[:, :min_len]

#————————合成失真语音————————
snr_dbs = torch.tensor([-5])
WAVEFORM_DISTORTED = F.add_noise(WAVEFORM_SPEECH, WAVEFORM_NOISE, snr_dbs)

# #————————可视化————————
# plot(WAVEFORM_SPEECH, "Clean Speech")
# plot(WAVEFORM_NOISE, "Noise")
# # plot(WAVEFORM_DISTORTED[0:1], f"Distorted Speech with {snr_dbs[0]}dB SNR")
# # plot(WAVEFORM_DISTORTED[1:2], f"Distorted Speech with {snr_dbs[1]}dB SNR")
# plot(WAVEFORM_DISTORTED, "Distorted Speech")
# plt.show()

#————————客观评估指标————————
objective_model = SQUIM_OBJECTIVE.get_model()

stoi_hyp, pesq_hyp, si_sdr_hyp = objective_model(WAVEFORM_DISTORTED)
print("Estimated metrics for distorted speech\n")
print(f"STOI: {stoi_hyp[0]}")
print(f"PESQ: {pesq_hyp[0]}")
print(f"SI-SDR: {si_sdr_hyp[0]}\n")

pesq_ref = pesq(16000, WAVEFORM_SPEECH[0].numpy(), WAVEFORM_DISTORTED[0].numpy(), mode="wb")
stoi_ref = stoi(WAVEFORM_SPEECH[0].numpy(), WAVEFORM_DISTORTED[0].numpy(), 16000, extended=False)
si_sdr_ref = si_snr(WAVEFORM_DISTORTED, WAVEFORM_SPEECH)
print("Reference metrics for distorted speech\n")
print(f"STOI: {stoi_ref}")
print(f"PESQ: {pesq_ref}")
print(f"SI-SDR: {si_sdr_ref}")

#———————————————MOS+NMR————————————————
subjective_model = SQUIM_SUBJECTIVE.get_model()

#NMR语音
# NMR_SPEECH = torchaudio.load("1688-142285-0007.wav")
WAVEFORM_NMR, SAMPLE_RATE_NMR = torchaudio.load(r"D:\PythonProject\QoS\AISHELL-2-sample\data\wav\C0936\IC0936W0131.wav")#AISHELL-2中文语音数据集
if SAMPLE_RATE_NMR != 16000:
    WAVEFORM_NMR = F.resample(WAVEFORM_NMR, SAMPLE_RATE_NMR, 16000)
# plot(WAVEFORM_NMR, "NMR")
# plt.show()

#估计MOS
mos = []
mos.append(subjective_model(WAVEFORM_SPEECH, WAVEFORM_NMR)[0].item())
mos.append(subjective_model(WAVEFORM_DISTORTED, WAVEFORM_NMR)[0].item())
print(f"Estimated MOS for clean speech is MOS: {mos[0]}")
print(f"Estimated MOS for distorted speech is MOS: {mos[1]}")

