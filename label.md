```python
import os, glob, csv
import torch
import torchaudio
import torchaudio.functional as F
from pesq import pesq
from pystoi import stoi
from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE

# --------------------- 可配置路径 ---------------------
CLEAN_DIR = r"D:\PythonProject\QoS\dataset\train_wavs_root"   # 干净语音
NOISE_PATH = r"Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo.wav"   # 唯一噪声
NMR_DIR   = r"D:\PythonProject\QoS\clean_pool"                # 多条干净 NMR（建议4~8条）
OUT_CSV   = r"D:\PythonProject\QoS\labels_full.csv"           # 输出CSV
SNR_LIST  = [-5]                                              # 需要的SNR
SR = 16000
# -------------------------------------------------------

device = "cpu"  # Windows上先用CPU，后续Linux再GPU

def to_mono_16k(wav, sr):
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    elif wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if sr != SR:
        wav = F.resample(wav, sr, SR)
    return wav

def crop_to_min_len(a, b):
    """对齐到相同最短长度"""
    L = min(a.size(-1), b.size(-1))
    return a[..., :L], b[..., :L]

def match_len_like(target_len, x):
    """把x调整为与target_len一样长"""
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

# 加载唯一噪声
noise_wav, sr_n = torchaudio.load(NOISE_PATH)
noise_wav = to_mono_16k(noise_wav, sr_n).to(device)

# 加载多条 NMR
nmr_paths = sorted(glob.glob(os.path.join(NMR_DIR, "*.wav")))
if len(nmr_paths) < 1:
    raise RuntimeError("NMR_DIR 为空，请放入多条干净语音")
nmr_list = []
for p in nmr_paths:
    try:
        w, sr = torchaudio.load(p)
        w = to_mono_16k(w, sr).to(device)
        nmr_list.append(w)
    except Exception as e:
        print("跳过NMR:", p, "错误:", e)
print(f"NMR 准备完成，共 {len(nmr_list)} 条")

# 老师模型
objective_model = SQUIM_OBJECTIVE.get_model().to(device).eval()
subjective_model = SQUIM_SUBJECTIVE.get_model().to(device).eval()

# 遍历训练集
wav_paths = sorted(glob.glob(os.path.join(CLEAN_DIR, "**", "*.wav"), recursive=True))
print(f"共 {len(wav_paths)} 条干净语音")

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

            # clean/noise 对齐
            clean_ref, noise_ref = crop_to_min_len(clean_wav, noise_wav)

            for snr_db in SNR_LIST:
                distorted = F.add_noise(clean_ref, noise_ref, torch.tensor([float(snr_db)], dtype=torch.float32))[0:1, :]

                # --- 参考指标 ---
                try:
                    pesq_ref = pesq(SR, clean_ref[0].cpu().numpy(), distorted[0].cpu().numpy(), mode="wb")
                except Exception:
                    pesq_ref = float("nan")
                try:
                    stoi_ref = stoi(clean_ref[0].cpu().numpy(), distorted[0].cpu().numpy(), SR, extended=False)
                except Exception:
                    stoi_ref = float("nan")
                try:
                    si_snr_ref = si_snr(distorted.cpu(), clean_ref.cpu())
                except Exception:
                    si_snr_ref = float("nan")

                # --- 老师客观预测 ---
                with torch.inference_mode():
                    stoi_hyp, pesq_hyp, si_sdr_hyp = objective_model(distorted)
                stoi_hyp = float(stoi_hyp.squeeze().item())
                pesq_hyp = float(pesq_hyp.squeeze().item())
                si_sdr_hyp = float(si_sdr_hyp.squeeze().item())

                # --- 主观 MOS（多NMR平均）---
                with torch.inference_mode():
                    # 干净
                    scores_c = []
                    for nmr in nmr_list:
                        nmr_c = match_len_like(clean_ref.size(-1), nmr)
                        scores_c.append(extract_mos(subjective_model(clean_ref, nmr_c)))
                    mos_clean = sum(scores_c) / len(scores_c)

                    # 失真
                    scores_d = []
                    for nmr in nmr_list:
                        nmr_d = match_len_like(distorted.size(-1), nmr)
                        scores_d.append(extract_mos(subjective_model(distorted, nmr_d)))
                    mos_dist = sum(scores_d) / len(scores_d)

                writer.writerow([
                    cpath, snr_db,
                    f"{stoi_ref:.6f}", f"{pesq_ref:.6f}", f"{si_snr_ref:.6f}",
                    f"{stoi_hyp:.6f}", f"{pesq_hyp:.6f}", f"{si_sdr_hyp:.6f}",
                    f"{mos_clean:.6f}", f"{mos_dist:.6f}"
                ])
        except Exception as e:
            print(f"[{i}/{len(wav_paths)}] 跳过 {cpath} 错误：{e}")

        if i % 20 == 0:
            print(f"进度 [{i}/{len(wav_paths)}]")

print("批量打标签完成 ->", OUT_CSV)
