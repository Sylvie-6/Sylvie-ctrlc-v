```python
import os, glob, csv
import torch
import torchaudio
import torchaudio.functional as F
from pesq import pesq
from pystoi import stoi
from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE

# --------------------- 你可以只改这里的路径和参数 ---------------------
CLEAN_DIR = r"D:\PythonProject\QoS\dataset\train_wavs_root"         # 训练集干净语音根目录（递归）
NOISE_PATH = r"Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo.wav"         # 仅一条噪声
NMR_PATH   = r"D:\PythonProject\QoS\AISHELL-2-sample\data\wav\C0936\IC0936W0131.wav"  # 仅一条NMR
OUT_CSV    = r"D:\PythonProject\QoS\labels_full.csv"               # 输出CSV
SNR_LIST   = [-5]                                                  # 想多打几档SNR就加进去，比如 [-5, 0, 5, 20]
SR = 16000
# -------------------------------------------------------------------

device = "cpu"  # 按你的要求：Windows 上先用 CPU，后续到 Linux 再上 GPU

def to_mono_16k(wav, sr):
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    elif wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if sr != SR:
        wav = F.resample(wav, sr, SR)
    return wav

def crop_to_min_len(a, b):
    """对齐到相同最短长度（与你现有逻辑一致，不用固定窗）"""
    L = min(a.size(-1), b.size(-1))
    return a[..., :L], b[..., :L]

def match_len_like(target_len, x):
    """把 x 调整为与 target_len 一样长：长则裁，短则重复（仅为 MOS 的 NMR 对齐，不用固定窗口）"""
    T = x.size(-1)
    if T == target_len:
        return x
    if T > target_len:
        return x[..., :target_len]
    repeat = (target_len + T - 1) // T
    return x.repeat(1, repeat)[..., :target_len]

def extract_mos(out):
    if isinstance(out, torch.Tensor):
        return float(out.squeeze().item())
    if hasattr(out, "mos"):
        return float(out.mos.squeeze().item())
    return float(out.squeeze().item())

# 加载唯一的噪声 & NMR
noise_wav, sr_n = torchaudio.load(NOISE_PATH)
noise_wav = to_mono_16k(noise_wav, sr_n).to(device)

nmr_wav, sr_r = torchaudio.load(NMR_PATH)
nmr_wav = to_mono_16k(nmr_wav, sr_r).to(device)

# 老师模型
objective_model = SQUIM_OBJECTIVE.get_model().to(device).eval()
subjective_model = SQUIM_SUBJECTIVE.get_model().to(device).eval()

# 遍历训练集所有 wav
wav_paths = sorted(glob.glob(os.path.join(CLEAN_DIR, "**", "*.wav"), recursive=True))
print(f"共 {len(wav_paths)} 条干净语音，将与唯一噪声 {os.path.basename(NOISE_PATH)} 组合，SNR档位：{SNR_LIST}")

# 写CSV
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "path_clean", "snr_db",
        "stoi_ref", "pesq_ref", "si_snr_ref",
        "stoi_hyp", "pesq_hyp", "si_sdr_hyp",
        "mos_clean", "mos_dist"
    ])

    for i, cpath in enumerate(wav_paths, 1):
        try:
            clean_wav, sr_c = torchaudio.load(cpath)
            clean_wav = to_mono_16k(clean_wav, sr_c).to(device)

            # 参考指标先对齐 clean/noise 到同长
            clean_ref, noise_ref = crop_to_min_len(clean_wav, noise_wav)

            for snr_db in SNR_LIST:
                # 合成失真（与你现有写法一致）
                distorted = F.add_noise(clean_ref, noise_ref, torch.tensor([float(snr_db)], dtype=torch.float32))[0:1, :]

                # --- 参考指标（与干净对比）---
                # 1) PESQ (16k→wb)
                try:
                    pesq_ref = pesq(SR, clean_ref[0].cpu().numpy(), distorted[0].cpu().numpy(), mode="wb")
                except Exception:
                    pesq_ref = float("nan")
                # 2) STOI
                try:
                    stoi_ref = stoi(clean_ref[0].cpu().numpy(), distorted[0].cpu().numpy(), SR, extended=False)
                except Exception:
                    stoi_ref = float("nan")
                # 3) SI-SNR（你的实现）
                try:
                    si_snr_ref = si_snr(distorted.cpu(), clean_ref.cpu())
                except Exception:
                    si_snr_ref = float("nan")

                # --- 老师模型（非侵入式预测，针对失真音频）---
                with torch.inference_mode():
                    stoi_hyp, pesq_hyp, si_sdr_hyp = objective_model(distorted)
                stoi_hyp = float(stoi_hyp.squeeze().item())
                pesq_hyp = float(pesq_hyp.squeeze().item())
                si_sdr_hyp = float(si_sdr_hyp.squeeze().item())

                # --- 主观 MOS（NORESQA）：不固定窗口，只把 NMR 调到与当前测试同长 ---
                with torch.inference_mode():
                    # 干净 MOS
                    nmr_c = match_len_like(clean_ref.size(-1), nmr_wav)
                    mos_clean = extract_mos(subjective_model(clean_ref, nmr_c))
                    # 失真 MOS
                    nmr_d = match_len_like(distorted.size(-1), nmr_wav)
                    mos_dist = extract_mos(subjective_model(distorted, nmr_d))

                writer.writerow([
                    cpath, snr_db,
                    f"{stoi_ref:.6f}", f"{pesq_ref:.6f}", f"{si_snr_ref:.6f}",
                    f"{stoi_hyp:.6f}", f"{pesq_hyp:.6f}", f"{si_sdr_hyp:.6f}",
                    f"{mos_clean:.6f}", f"{mos_dist:.6f}"
                ])

        except Exception as e:
            print(f"[{i}/{len(wav_paths)}] 跳过 {cpath}，因错误：{e}")

        if i % 20 == 0:
            print(f"进度 [{i}/{len(wav_paths)}]")

print("批量打标签完成 ->", OUT_CSV)

