"""Requantize the routed experts of an MTP draft GGUF to Q2_0 with a per-block least-squares scale search.

ggml's reference Q2_0 quantizer sets the block scale to max|w| and never reaches the +2 level; Strata's packer
searches the scale per 64-weight block to minimize the squared error over the {-1, 0, 1, 2} x d grid, which is
what keeps its 2-bit draft experts at full acceptance. This does the same on the three expert tensors and copies
everything else, writing two files: one with the heads untouched (Q8_0) and one with token_embd/output in Q4_0.

usage: python -I requant_mtp_q2_0.py IN.gguf OUT_PREFIX
"""
import sys
import numpy as np

import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))
import gguf  # noqa: E402
from gguf import GGMLQuantizationType as T  # noqa: E402

QK = 64
BLOCK_BYTES = 18
FACTORS = np.concatenate([np.linspace(0.50, 1.00, 26), np.linspace(1.05, 1.30, 6)]).astype(np.float32)


def q2_0_mse(w: np.ndarray) -> np.ndarray:
    """w: (..., ncols) float32 -> uint8 (..., ncols/QK*18) in ggml's block_q2_0 layout, scale searched per block."""
    shape = w.shape
    blocks = w.reshape(-1, QK).astype(np.float32)
    amax = np.abs(blocks).max(axis=1, keepdims=True)
    best_err = np.full((blocks.shape[0], 1), np.inf, dtype=np.float32)
    best_d = np.zeros((blocks.shape[0], 1), dtype=np.float16)
    best_q = np.zeros(blocks.shape, dtype=np.int8)
    for f in FACTORS:
        d = (amax * f).astype(np.float16)          # the stored scale is fp16, search with the rounded value
        df = d.astype(np.float32)
        inv = np.where(df > 0, 1.0 / np.where(df > 0, df, 1.0), 0.0).astype(np.float32)
        q = np.clip(np.rint(blocks * inv), -1, 2)
        err = ((q * df - blocks) ** 2).sum(axis=1, keepdims=True)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_d = np.where(better, d, best_d)
        best_q = np.where(better, q.astype(np.int8), best_q)
    codes = (best_q + 1).astype(np.uint8)          # 0..3
    c = codes.reshape(-1, QK // 4, 4)
    packed = (c[:, :, 0] | (c[:, :, 1] << 2) | (c[:, :, 2] << 4) | (c[:, :, 3] << 6)).astype(np.uint8)
    out = np.concatenate([best_d.view(np.uint8).reshape(-1, 2), packed], axis=1)
    assert out.shape[1] == BLOCK_BYTES
    return out.reshape(shape[:-1] + (shape[-1] // QK * BLOCK_BYTES,)), float(best_err.sum())


def q2_0_ref_err(w: np.ndarray) -> float:
    blocks = w.reshape(-1, QK).astype(np.float32)
    d = np.abs(blocks).max(axis=1, keepdims=True).astype(np.float16).astype(np.float32)
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
    q = np.clip(np.rint(blocks * inv), -1, 2)
    return float(((q * d - blocks) ** 2).sum())


def main():
    src, prefix = sys.argv[1], sys.argv[2]
    reader = gguf.GGUFReader(src)
    arch = reader.fields["general.architecture"].contents()
    outs = {
        "q8heads": gguf.GGUFWriter(prefix + "-Q2_0mse-q8heads.gguf", arch),
        "q4heads": gguf.GGUFWriter(prefix + "-Q2_0mse-q4heads.gguf", arch),
    }
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        for w in outs.values():
            w.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)
    for w in outs.values():
        w.add_string("quantize.experts", "q2_0 per-block least-squares scale search (scripts/requant_mtp_q2_0.py)")

    plan = []  # (tensor, {out_name: (data, qtype)})
    tot_ref = tot_mse = tot_n = 0.0
    for t in reader.tensors:
        per_out = {}
        is_exp = t.name.endswith(("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"))
        is_head = t.name in ("token_embd.weight", "output.weight")
        if is_exp:
            f32 = gguf.quants.dequantize(t.data, t.tensor_type)
            data, err = q2_0_mse(f32)
            ref = q2_0_ref_err(f32)
            tot_ref += ref; tot_mse += err; tot_n += float((f32 ** 2).sum())
            print(f"{t.name}: {t.tensor_type.name} -> Q2_0  rel.err ref {ref / float((f32 ** 2).sum()):.4f}  searched {err / float((f32 ** 2).sum()):.4f}", flush=True)
            for k in outs:
                per_out[k] = (data, T.Q2_0)
        elif is_head:
            f32 = gguf.quants.dequantize(t.data, t.tensor_type)
            per_out["q8heads"] = (t.data, t.tensor_type)
            per_out["q4heads"] = (gguf.quants.quantize(f32, T.Q4_0), T.Q4_0)
            print(f"{t.name}: kept {t.tensor_type.name} / Q4_0 variant", flush=True)
        else:
            for k in outs:
                per_out[k] = (t.data, t.tensor_type)
        plan.append((t, per_out))
        for k, (data, qt) in per_out.items():
            outs[k].add_tensor_info(t.name, data.shape, data.dtype, data.nbytes, qt)
    print(f"experts total rel.err: reference {tot_ref / tot_n:.4f}, searched {tot_mse / tot_n:.4f}", flush=True)

    for k, w in outs.items():
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_ti_data_to_file()
        for t, per_out in plan:
            data, _ = per_out[k]
            w.write_tensor_data(data, tensor_endianess=reader.endianess)
        w.close()
        print("wrote", k, flush=True)


if __name__ == "__main__":
    main()
