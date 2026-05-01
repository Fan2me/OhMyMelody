import numpy as np
import time
import scipy
import scipy.signal
import math
import json


CFP_LAST_PROFILE = {}
FREQ_LOG_FREQ_CACHE = {}
QUEF_LOG_FREQ_CACHE = {}


def _ms_since(start_time):
    return (time.perf_counter() - start_time) * 1000.0


def _set_profile_value(profile, key, value):
    if profile is None:
        return
    profile[key] = float(value)


def _build_freq_mapping_cache_key(f, fr, fc, tc, num_per_oct):
    freq_arr = np.asarray(f, dtype=np.float32)
    return (
        int(freq_arr.shape[0]),
        float(fr),
        float(fc),
        float(tc),
        int(num_per_oct),
        float(freq_arr[0]) if freq_arr.size else 0.0,
        float(freq_arr[-1]) if freq_arr.size else 0.0,
    )


def _build_quef_mapping_cache_key(q, fs, fc, tc, num_per_oct):
    quef_arr = np.asarray(q, dtype=np.float32)
    return (
        int(quef_arr.shape[0]),
        int(fs),
        float(fc),
        float(tc),
        int(num_per_oct),
        float(quef_arr[0]) if quef_arr.size else 0.0,
        float(quef_arr[-1]) if quef_arr.size else 0.0,
    )


def _apply_sparse_banded_mapping(spec, source):
    out = np.zeros((spec["rows"], source.shape[1]), dtype=np.float32)
    for band in spec["bands"]:
        row_idx = band["row"]
        start = band["start"]
        weights = band["weights"]
        end = start + weights.shape[0]
        segment = source[start:end, :]
        out[row_idx, :] = np.sum(segment * weights[:, np.newaxis], axis=0, dtype=np.float32)
    return out


def _build_freq_mapping_spec(f, fr, fc, tc, NumPerOct):
    StartFreq = fc
    StopFreq = 1 / tc
    Nest = int(np.ceil(np.log2(StopFreq / StartFreq)) * NumPerOct)
    central_freq = []

    for i in range(0, Nest):
        CenFreq = StartFreq * pow(2, float(i) / NumPerOct)
        if CenFreq < StopFreq:
            central_freq.append(CenFreq)
        else:
            break

    Nest = len(central_freq)
    bands = []
    for i in range(1, Nest - 1):
        l = int(round(central_freq[i - 1] / fr))
        r = int(round(central_freq[i + 1] / fr) + 1)
        if l >= r - 1:
            bands.append(
                {
                    "row": int(i),
                    "start": int(l),
                    "weights": np.array([1.0], dtype=np.float32),
                }
            )
            continue

        weight_start = max(0, l)
        weight_end = min(len(f), r)
        weights = np.zeros(max(0, weight_end - weight_start), dtype=np.float32)
        for j in range(weight_start, weight_end):
            weight = 0.0
            if f[j] > central_freq[i - 1] and f[j] < central_freq[i]:
                weight = (f[j] - central_freq[i - 1]) / (
                    central_freq[i] - central_freq[i - 1]
                )
            elif f[j] > central_freq[i] and f[j] < central_freq[i + 1]:
                weight = (central_freq[i + 1] - f[j]) / (
                    central_freq[i + 1] - central_freq[i]
                )
            weights[j - weight_start] = weight
        bands.append({"row": int(i), "start": weight_start, "weights": weights})

    return {
        "rows": max(0, Nest - 1),
        "bands": bands,
        "central_freq": tuple(central_freq),
    }


def _build_quef_mapping_spec(q, fs, fc, tc, NumPerOct):
    StartFreq = fc
    StopFreq = 1 / tc
    Nest = int(np.ceil(np.log2(StopFreq / StartFreq)) * NumPerOct)
    central_freq = []

    for i in range(0, Nest):
        CenFreq = StartFreq * pow(2, float(i) / NumPerOct)
        if CenFreq < StopFreq:
            central_freq.append(CenFreq)
        else:
            break

    f = 1 / q
    Nest = len(central_freq)
    bands = []
    for i in range(1, Nest - 1):
        start = int(round(fs / central_freq[i + 1]))
        end = int(round(fs / central_freq[i - 1]) + 1)
        weight_start = max(0, start)
        weight_end = min(len(f), end)
        weights = np.zeros(max(0, weight_end - weight_start), dtype=np.float32)
        for j in range(weight_start, weight_end):
            weight = 0.0
            if f[j] > central_freq[i - 1] and f[j] < central_freq[i]:
                weight = (f[j] - central_freq[i - 1]) / (
                    central_freq[i] - central_freq[i - 1]
                )
            elif f[j] > central_freq[i] and f[j] < central_freq[i + 1]:
                weight = (central_freq[i + 1] - f[j]) / (
                    central_freq[i + 1] - central_freq[i]
                )
            weights[j - weight_start] = weight
        bands.append({"row": int(i), "start": weight_start, "weights": weights})

    return {
        "rows": max(0, Nest - 1),
        "bands": bands,
        "central_freq": tuple(central_freq),
    }


def STFT(x, fr, fs, Hop, h, profile=None):
    t_stft = time.perf_counter()
    t = np.arange(0, np.ceil(len(x) / float(Hop)) * Hop, Hop)
    N = int(fs / float(fr))
    window_size = len(h)
    f = fs * np.linspace(0, 0.5, int(np.round(N / 2)), endpoint=True)
    Lh = int(np.floor(float(window_size - 1) / 2))
    tfr = np.zeros((int(N), len(t)), dtype=np.float32)
    _set_profile_value(profile, "stft_prepare_ms", _ms_since(t_stft))

    t_fill = time.perf_counter()
    for icol in range(0, len(t)):
        ti = int(t[icol])
        tau = np.arange(
            int(-min([round(N / 2.0) - 1, Lh, ti - 1])),
            int(min([round(N / 2.0) - 1, Lh, len(x) - ti])),
        )
        indices = np.mod(N + tau, N) + 1
        tfr[indices - 1, icol] = (
            x[ti + tau - 1] * h[Lh + tau - 1] / np.linalg.norm(h[Lh + tau - 1])
        )
    _set_profile_value(profile, "stft_fill_ms", _ms_since(t_fill))
    t_fft = time.perf_counter()
    tfr = abs(scipy.fftpack.fft(tfr, n=N, axis=0))
    _set_profile_value(profile, "stft_fft_ms", _ms_since(t_fft))
    _set_profile_value(profile, "stft_total_ms", _ms_since(t_stft))
    return tfr, f, t, N


def nonlinear_func(X, g, cutoff):
    cutoff = int(cutoff)
    if g != 0:
        X[X < 0] = 0
        X[:cutoff, :] = 0
        X[-cutoff:, :] = 0
        X = np.power(X, g)
    else:
        X = np.log(X)
        X[:cutoff, :] = 0
        X[-cutoff:, :] = 0
    return X


def Freq2LogFreqMapping(tfr, f, fr, fc, tc, NumPerOct, profile=None, profile_prefix="freq"):
    t0 = time.perf_counter()
    cache_key = _build_freq_mapping_cache_key(f, fr, fc, tc, NumPerOct)
    cached = FREQ_LOG_FREQ_CACHE.get(cache_key)
    if cached is None:
        cached = _build_freq_mapping_spec(f, fr, fc, tc, NumPerOct)
        FREQ_LOG_FREQ_CACHE[cache_key] = cached
        _set_profile_value(profile, f"{profile_prefix}_cache_hit", 0)
    else:
        _set_profile_value(profile, f"{profile_prefix}_cache_hit", 1)
    mapping_spec = cached
    _set_profile_value(profile, f"{profile_prefix}_map_build_ms", _ms_since(t0))
    t_dot = time.perf_counter()
    tfrL = _apply_sparse_banded_mapping(mapping_spec, tfr)
    _set_profile_value(profile, f"{profile_prefix}_dot_ms", _ms_since(t_dot))
    _set_profile_value(profile, f"{profile_prefix}_total_ms", _ms_since(t0))
    return tfrL, list(mapping_spec["central_freq"])


def Quef2LogFreqMapping(ceps, q, fs, fc, tc, NumPerOct, profile=None):
    t0 = time.perf_counter()
    cache_key = _build_quef_mapping_cache_key(q, fs, fc, tc, NumPerOct)
    cached = QUEF_LOG_FREQ_CACHE.get(cache_key)
    if cached is None:
        cached = _build_quef_mapping_spec(q, fs, fc, tc, NumPerOct)
        QUEF_LOG_FREQ_CACHE[cache_key] = cached
        _set_profile_value(profile, "quef_cache_hit", 0)
    else:
        _set_profile_value(profile, "quef_cache_hit", 1)
    mapping_spec = cached
    _set_profile_value(profile, "quef_map_build_ms", _ms_since(t0))
    t_dot = time.perf_counter()
    tfrL = _apply_sparse_banded_mapping(mapping_spec, ceps)
    _set_profile_value(profile, "quef_dot_ms", _ms_since(t_dot))
    _set_profile_value(profile, "quef_total_ms", _ms_since(t0))
    return tfrL, list(mapping_spec["central_freq"])


def CFP_filterbank(x, fr, fs, Hop, h, fc, tc, g, NumPerOctave, profile=None):
    t0 = time.perf_counter()
    NumofLayer = np.size(g)
    N = int(fs / float(fr))
    [tfr, f, t, N] = STFT(x, fr, fs, Hop, h, profile=profile)
    tfr = np.power(abs(tfr), g[0])
    tfr0 = tfr  # original STFT
    ceps = np.zeros(tfr.shape)
    _set_profile_value(profile, "cfp_initial_power_ms", _ms_since(t0))

    if NumofLayer >= 2:
        t_layers = time.perf_counter()
        for gc in range(1, NumofLayer):
            if np.remainder(gc, 2) == 1:
                tc_idx = round(fs * tc)
                ceps = np.real(np.fft.fft(tfr, axis=0)) / np.sqrt(N)
                ceps = nonlinear_func(ceps, g[gc], tc_idx)
            else:
                fc_idx = round(fc / fr)
                tfr = np.real(np.fft.fft(ceps, axis=0)) / np.sqrt(N)
                tfr = nonlinear_func(tfr, g[gc], fc_idx)
        _set_profile_value(profile, "cfp_layers_ms", _ms_since(t_layers))

    t_trim = time.perf_counter()
    tfr0 = tfr0[: int(round(N / 2)), :]
    tfr = tfr[: int(round(N / 2)), :]
    ceps = ceps[: int(round(N / 2)), :]

    HighFreqIdx = int(round((1 / tc) / fr) + 1)
    f = f[:HighFreqIdx]
    tfr0 = tfr0[:HighFreqIdx, :]
    tfr = tfr[:HighFreqIdx, :]
    HighQuefIdx = int(round(fs / fc) + 1)

    q = np.arange(HighQuefIdx) / float(fs)

    ceps = ceps[:HighQuefIdx, :]
    _set_profile_value(profile, "cfp_trim_ms", _ms_since(t_trim))

    tfrL0, central_frequencies = Freq2LogFreqMapping(
        tfr0, f, fr, fc, tc, NumPerOctave, profile=profile, profile_prefix="freq0"
    )
    tfrLF, central_frequencies = Freq2LogFreqMapping(
        tfr, f, fr, fc, tc, NumPerOctave, profile=profile, profile_prefix="freq"
    )
    tfrLQ, central_frequencies = Quef2LogFreqMapping(
        ceps, q, fs, fc, tc, NumPerOctave, profile=profile
    )

    _set_profile_value(profile, "cfp_filterbank_total_ms", _ms_since(t0))
    return tfrL0, tfrLF, tfrLQ, f, q, t, central_frequencies


def feature_extraction(
    x, fs, Hop=512, Window=2049, StartFreq=80.0, StopFreq=1000.0, NumPerOct=48, profile=None
):
    t0 = time.perf_counter()
    fr = 2.0  # frequency resolution
    t_window = time.perf_counter()
    h = scipy.signal.windows.blackmanharris(Window)  # window size
    _set_profile_value(profile, "window_ms", _ms_since(t_window))
    g = np.array([0.24, 0.6, 1])  # gamma value

    t_filter = time.perf_counter()
    tfrL0, tfrLF, tfrLQ, f, q, t, CenFreq = CFP_filterbank(
        x, fr, fs, Hop, h, StartFreq, 1 / StopFreq, g, NumPerOct, profile=profile
    )
    _set_profile_value(profile, "feature_filterbank_ms", _ms_since(t_filter))
    t_combine = time.perf_counter()
    Z = tfrLF * tfrLQ
    time_axis = t / fs
    _set_profile_value(profile, "feature_combine_ms", _ms_since(t_combine))
    _set_profile_value(profile, "feature_total_ms", _ms_since(t0))
    return Z, time_axis, CenFreq, tfrL0, tfrLF, tfrLQ


def midi2hz(midi):
    return 2 ** ((midi - 69) / 12.0) * 440


def hz2midi(hz):
    return 69 + 12 * np.log2(hz / 440.0)


def get_CenFreq(StartFreq=80, StopFreq=1000, NumPerOct=48):
    Nest = int(np.ceil(np.log2(StopFreq / StartFreq)) * NumPerOct)
    central_freq = []
    for i in range(0, Nest):
        CenFreq = StartFreq * pow(2, float(i) / NumPerOct)
        if CenFreq < StopFreq:
            central_freq.append(CenFreq)
        else:
            break
    return central_freq


def get_time(fs, Hop, end):
    return np.arange(Hop / fs, end, Hop / fs)


def lognorm(x):
    # 数值稳定：避免极小负值导致 log 出现异常
    x = np.asarray(x, dtype=np.float32)
    return np.log1p(np.maximum(x, 0.0))


def norm(x):
    x = np.asarray(x, dtype=np.float32)
    xmin = np.min(x)
    xmax = np.max(x)
    denom = xmax - xmin
    if not np.isfinite(denom) or denom < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - xmin) / (denom + 1e-8)
    return np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)


def preprocess_audio_for_cfp(x, fs, target_fs=8000):
    """
    在 CFP 前做与训练侧更一致的预处理：
    1) 强制 float32 + 一维
    2) 使用 scipy.signal.resample_poly 重采样到 target_fs
    """
    x = np.asarray(x, dtype=np.float32).flatten()
    fs = int(fs)
    target_fs = int(target_fs)

    if fs <= 0:
        raise ValueError(f"invalid fs: {fs}")
    if target_fs <= 0:
        raise ValueError(f"invalid target_fs: {target_fs}")

    if fs == target_fs:
        return x, fs

    g = math.gcd(fs, target_fs)
    up = target_fs // g
    down = fs // g
    x_rs = scipy.signal.resample_poly(x, up, down).astype(np.float32, copy=False)
    return x_rs, target_fs


def cfp_process_from_array(x, fs, hop=80, model_type="vocal", target_fs=8000):
    """
    x: numpy array (float32), fs: int
    其余参数同 cfp_process
    """
    global CFP_LAST_PROFILE
    profile = {
        "input_samples": int(np.asarray(x).size),
        "input_fs": int(fs),
        "target_fs": int(target_fs),
        "hop": int(hop),
        "model_type": str(model_type),
    }

    t_total = time.perf_counter()
    t_pre = time.perf_counter()
    x, fs = preprocess_audio_for_cfp(x, fs, target_fs=target_fs)
    _set_profile_value(profile, "preprocess_ms", _ms_since(t_pre))
    profile["resampled_samples"] = int(np.asarray(x).size)
    profile["effective_fs"] = int(fs)

    t_feat = time.perf_counter()
    Z, time_arr, CenFreq, tfrL0, tfrLF, tfrLQ = feature_extraction(
        x, fs, Hop=hop, Window=768, StartFreq=32, StopFreq=2050, NumPerOct=60, profile=profile
    )
    _set_profile_value(profile, "feature_extraction_call_ms", _ms_since(t_feat))

    t_post = time.perf_counter()
    t_post_0 = time.perf_counter()
    tfrL0 = norm(lognorm(tfrL0))[np.newaxis, :, :]
    _set_profile_value(profile, "post_tfrL0_ms", _ms_since(t_post_0))
    t_post_f = time.perf_counter()
    tfrLF = norm(lognorm(tfrLF))[np.newaxis, :, :]
    _set_profile_value(profile, "post_tfrLF_ms", _ms_since(t_post_f))
    t_post_q = time.perf_counter()
    tfrLQ = norm(lognorm(tfrLQ))[np.newaxis, :, :]
    _set_profile_value(profile, "post_tfrLQ_ms", _ms_since(t_post_q))
    _set_profile_value(profile, "post_norm_total_ms", _ms_since(t_post))

    t_concat = time.perf_counter()
    W = np.concatenate((tfrL0, tfrLF, tfrLQ), axis=0)
    _set_profile_value(profile, "concat_ms", _ms_since(t_concat))
    profile["output_shape"] = [int(v) for v in W.shape]
    # 保存shape和二进制float32
    t_export = time.perf_counter()
    shape = np.array(W.shape, dtype=np.int32)
    with open("cfp_out_shape.bin", "wb") as f:
        f.write(shape.tobytes())
    with open("cfp_out.bin", "wb") as f:
        f.write(W.astype(np.float32).tobytes())
    _set_profile_value(profile, "export_ms", _ms_since(t_export))
    _set_profile_value(profile, "total_ms", _ms_since(t_total))
    CFP_LAST_PROFILE = profile
    return W


def get_last_cfp_profile_json():
    return json.dumps(CFP_LAST_PROFILE, ensure_ascii=False)
