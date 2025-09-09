```python

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio
from pesq import pesq
from pystoi import stoi
from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE
from torchaudio.utils import download_asset
import torchaudio.functional as F
#播放音频
from scipy.io import wavfile
import sounddevice as sd

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

#——————————画波形/声谱图———————————
def plot(waveform, title, sample_rate=16000):
    wav_numpy = waveform.numpy()
    sample_size = waveform.shape[1]
    time_axis = torch.arange(0, sample_size) / sample_rate

    figure, axes = plt.subplots(2,1)
    axes[0].plot(time_axis, wav_numpy[0], linewidth=1)
    axes[0].grid(True)
    axes[1].specgram(wav_numpy[0], Fs=sample_rate)
    figure.suptitle(title)

# SAMPLE_SPEECH = torchaudio.load("Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")
# SAMPLE_NOISE = torchaudio.load("Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo.wav")

WAVEFORM_SPEECH, SAMPLE_RATE_SPEECH = torchaudio.load("Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")
WAVEFORM_NOISE, SAMPLE_RATE_NOISE = torchaudio.load("Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo.wav")
WAVEFORM_NOISE =WAVEFORM_NOISE[0:1, :]#取一通道

#SQUIM仅支持16kHz，重采样到16k
if SAMPLE_RATE_SPEECH != 16000:
    WAVEFORM_SPEECH = F.resample(WAVEFORM_SPEECH, SAMPLE_RATE_SPEECH, 16000)
if SAMPLE_RATE_NOISE != 16000:
    WAVEFORM_NOISE = F.resample(WAVEFORM_NOISE, SAMPLE_RATE_NOISE, 16000)

#对齐长度（裁剪到相同的帧数）
if WAVEFORM_SPEECH.shape[1] < WAVEFORM_NOISE.shape[1]:
    WAVEFORM_NOISE = WAVEFORM_NOISE[:, : WAVEFORM_SPEECH.shape[1]]
else:
    WAVEFORM_SPEECH = WAVEFORM_SPEECH[:, : WAVEFORM_NOISE.shape[1]]

#————————合成两段不同SNR的失真语音————————
snr_dbs = torch.tensor([20,-5])
WAVEFORM_DISTORTED = F.add_noise(WAVEFORM_SPEECH, WAVEFORM_NOISE, snr_dbs)

#————————可视化————————

plot(WAVEFORM_SPEECH, "Clean Speech")
plot(WAVEFORM_NOISE, "Noise")
plot(WAVEFORM_DISTORTED[0:1], f"Distorted Speech with {snr_dbs[0]}dB SNR")
plot(WAVEFORM_DISTORTED[1:2], f"Distorted Speech with {snr_dbs[1]}dB SNR")
plt.show()

#————————客观评估指标————————
objective_model = SQUIM_OBJECTIVE.get_model()

# 20 dB 样本的预测与参考对照
stoi_hyp, pesq_hyp, si_sdr_hyp = objective_model(WAVEFORM_DISTORTED[0:1, :])
print(f"Estimated metrics for distorted speech at {snr_dbs[0]}dB are\n")
print(f"STOI: {stoi_hyp[0]}")
print(f"PESQ: {pesq_hyp[0]}")
print(f"SI-SDR: {si_sdr_hyp[0]}\n")

pesq_ref = pesq(16000, WAVEFORM_SPEECH[0].numpy(), WAVEFORM_DISTORTED[0].numpy(), mode="wb")
stoi_ref = stoi(WAVEFORM_SPEECH[0].numpy(), WAVEFORM_DISTORTED[0].numpy(), 16000, extended=False)
si_sdr_ref = si_snr(WAVEFORM_DISTORTED[0:1], WAVEFORM_SPEECH)
print(f"Reference metrics for distorted speech at {snr_dbs[0]}dB are\n")
print(f"STOI: {stoi_ref}")
print(f"PESQ: {pesq_ref}")
print(f"SI-SDR: {si_sdr_ref}")

# -5 dB 样本的预测与参考对照
stoi_hyp, pesq_hyp, si_sdr_hyp = objective_model(WAVEFORM_DISTORTED[1:2, :])
print(f"Estimated metrics for distorted speech at {snr_dbs[1]}dB are\n")
print(f"STOI: {stoi_hyp[0]}")
print(f"PESQ: {pesq_hyp[0]}")
print(f"SI-SDR: {si_sdr_hyp[0]}\n")

pesq_ref = pesq(16000, WAVEFORM_SPEECH[0].numpy(), WAVEFORM_DISTORTED[0].numpy(), mode="wb")
stoi_ref = stoi(WAVEFORM_SPEECH[0].numpy(), WAVEFORM_DISTORTED[1].numpy(), 16000, extended=False)
si_sdr_ref = si_snr(WAVEFORM_DISTORTED[1:2], WAVEFORM_SPEECH)
print(f"Reference metrics for distorted speech at {snr_dbs[1]}dB are\n")
print(f"STOI: {stoi_ref}")
print(f"PESQ: {pesq_ref}")
print(f"SI-SDR: {si_sdr_ref}")

#————————————————主管指标评估 MOS+NMR————————————————
subjective_model = SQUIM_SUBJECTIVE.get_model()

#NMR语音
# NMR_SPEECH = torchaudio.load("1688-142285-0007.wav")
WAVEFORM_NMR, SAMPLE_RATE_NMR = torchaudio.load("1688-142285-0007.wav")
if SAMPLE_RATE_NMR != 16000:
    WAVEFORM_NMR = F.resample(WAVEFORM_NMR, SAMPLE_RATE_NMR, 16000)

#对两段SNR失真语音估计MOS
mos = subjective_model(WAVEFORM_DISTORTED[0:1, :], WAVEFORM_NMR)
print(f"Estimated MOS for distorted speech at {snr_dbs[0]}dB is MOS: {mos[0]}")

mos = subjective_model(WAVEFORM_DISTORTED[1:2, :], WAVEFORM_NMR)
print(f"Estimated MOS for distorted speech at {snr_dbs[1]}dB is MOS: {mos[0]}")




