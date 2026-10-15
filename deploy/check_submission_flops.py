"""Static design-space integrity check for EURS submissions: computes total
compute (GFLOPs) and parameter count directly from a submitted ONNX graph's
static shapes -- no runtime, no GPU, no benchmarking hardware needed.
Official MaCVi @ WACV 2027 checker fetched from https://www.macvi.org/EURS_check_submission_flops.py
"""
import argparse
import sys
from collections import defaultdict

import onnx
from onnx import shape_inference, numpy_helper


ALWAYS_ALLOWED = {
    'Conv', 'BatchNormalization', 'Relu', 'Clip', 'Sigmoid', 'Concat', 'Add',
    'Resize', 'Reshape', 'Transpose', 'Constant', 'Shape', 'Gather',
    'Unsqueeze', 'Squeeze', 'Cast', 'Slice', 'Mul', 'Div', 'Sub', 'Identity',
    'ConstantOfShape', 'Expand',
}


def _tensor_shape(value_info_map, name):
    if name not in value_info_map:
        return None
    dims = value_info_map[name].type.tensor_type.shape.dim
    shape = []
    for d in dims:
        if d.HasField('dim_value'):
            shape.append(d.dim_value)
        else:
            return None  # dynamic/symbolic dim -- can't compute exact FLOPs
    return shape


def analyze(onnx_path):
    model = onnx.load(onnx_path)
    inferred = shape_inference.infer_shapes(model)
    graph = inferred.graph

    value_info_map = {}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        value_info_map[vi.name] = vi

    initializer_map = {init.name: init for init in graph.initializer}

    total_gflops = 0.0
    op_gflops = defaultdict(float)
    op_counts = defaultdict(int)
    unrecognized_ops = set()

    for node in graph.node:
        op_counts[node.op_type] += 1
        if node.op_type not in ALWAYS_ALLOWED and node.op_type not in ('MatMul', 'Gemm'):
            unrecognized_ops.add(node.op_type)

        if node.op_type == 'Conv':
            weight_name = node.input[1]
            if weight_name not in initializer_map:
                continue  # weight not a static initializer -- can't count exactly
            weight = numpy_helper.to_array(initializer_map[weight_name])
            out_ch, in_ch_per_group, kh, kw = weight.shape
            out_shape = _tensor_shape(value_info_map, node.output[0])
            if out_shape is None or len(out_shape) != 4:
                continue
            _, _, out_h, out_w = out_shape
            macs = out_h * out_w * out_ch * in_ch_per_group * kh * kw
            gflops = 2 * macs / 1e9  # 2 FLOPs per MAC (multiply + add)
            total_gflops += gflops
            op_gflops[node.op_type] += gflops

        elif node.op_type in ('MatMul', 'Gemm'):
            a_name, b_name = node.input[0], node.input[1]
            b_shape = None
            if b_name in initializer_map:
                b_shape = list(numpy_helper.to_array(initializer_map[b_name]).shape)
            a_shape = _tensor_shape(value_info_map, a_name)
            if b_shape is None or a_shape is None:
                continue
            M = a_shape[-2] if len(a_shape) >= 2 else 1
            K = a_shape[-1]
            N = b_shape[-1]
            macs = M * K * N
            gflops = 2 * macs / 1e9
            total_gflops += gflops
            op_gflops[node.op_type] += gflops

    total_params = sum(
        numpy_helper.to_array(init).size for init in graph.initializer
    )

    return {
        'total_gflops': total_gflops,
        'total_params_m': total_params / 1e6,
        'op_gflops': dict(op_gflops),
        'op_counts': dict(op_counts),
        'unrecognized_ops': unrecognized_ops,
    }


DEFAULT_MAX_GFLOPS = 12.0
DEFAULT_MAX_PARAMS_M = 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', required=True)
    parser.add_argument('--max_gflops', type=float, default=DEFAULT_MAX_GFLOPS,
                         help=f'Reject if total Conv/Gemm GFLOPs exceeds this (default: {DEFAULT_MAX_GFLOPS}, locked 2026-08-30).')
    parser.add_argument('--max_params_m', type=float, default=DEFAULT_MAX_PARAMS_M,
                         help=f'Reject if total parameter count (millions) exceeds this (default: {DEFAULT_MAX_PARAMS_M}, locked 2026-08-30).')
    parser.add_argument('--check_ops', action='store_true',
                         help="Flag op types outside the baseline's known vocabulary (informational, not blocking by default).")
    args = parser.parse_args()

    result = analyze(args.onnx)

    print(f"Total compute: {result['total_gflops']:.4f} GFLOPs (Conv/Gemm/MatMul only)")
    print(f"Total parameters: {result['total_params_m']:.4f} M")
    print("Breakdown by op type (GFLOPs):")
    for op, g in sorted(result['op_gflops'].items(), key=lambda x: -x[1]):
        print(f"  {op}: {g:.4f} GFLOPs ({result['op_counts'][op]} nodes)")
    print("All node op types and counts:")
    for op, c in sorted(result['op_counts'].items()):
        print(f"  {op}: {c}")

    exit_code = 0

    if args.check_ops and result['unrecognized_ops']:
        print(f"\nWARNING: op types outside the baseline's known vocabulary: "
              f"{sorted(result['unrecognized_ops'])}")
        print("Not blocking by default -- review manually before allowing if this matters for your deployment.")

    if args.max_gflops is not None and result['total_gflops'] > args.max_gflops:
        print(f"\nREJECTED: {result['total_gflops']:.4f} GFLOPs exceeds the "
              f"{args.max_gflops:.4f} GFLOPs cap.")
        exit_code = 1

    if args.max_params_m is not None and result['total_params_m'] > args.max_params_m:
        print(f"\nREJECTED: {result['total_params_m']:.4f}M parameters exceeds the "
              f"{args.max_params_m:.4f}M parameter cap.")
        exit_code = 1

    if exit_code == 0 and (args.max_gflops is not None or args.max_params_m is not None):
        print("\nPASSED: within the configured compute/parameter caps.")

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
