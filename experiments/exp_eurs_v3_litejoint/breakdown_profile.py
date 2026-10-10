import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'experiments/exp_eurs_v3_litejoint')
import torch
from torch.profiler import profile, ProfilerActivity
from model_v3 import EURSNetV3LiteJoint

def main():
    model = EURSNetV3LiteJoint(num_classes=5, restoration_mode='bounded')
    model.eval()
    x = torch.randn(1, 3, 1024, 1024)

    # 1. Profile full model
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_flops=True) as prof:
        with torch.no_grad():
            model(x)

    events = prof.key_averages()
    total_flops = sum([e.flops for e in events])
    print(f"Total model FLOPs: {total_flops / 1e9:.3f} GFLOPs")

    # 2. Profile each submodule separately
    # FWB
    wb_x = model.fwb(x)

    # Encoder
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_flops=True) as prof:
        with torch.no_grad():
            stem, c1, c2, c3, c4, c5 = model.encoder(wb_x)
    print(f"Encoder FLOPs: {sum([e.flops for e in prof.key_averages()]) / 1e9:.3f} GFLOPs")

    # Restoration decoder
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_flops=True) as prof:
        with torch.no_grad():
            restored, rest_feat_4 = model.restoration_decoder(stem, c1, c2, c3, raw_x=x, wb_x=wb_x)
    print(f"Restoration Decoder FLOPs: {sum([e.flops for e in prof.key_averages()]) / 1e9:.3f} GFLOPs")

    # Segmentation decoder
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_flops=True) as prof:
        with torch.no_grad():
            seg_logits, _ = model.segmentation_decoder(stem, c1, c2, c3, c4, c5, rest_feat_4, target_size=(1024, 1024))
    print(f"Segmentation Decoder FLOPs: {sum([e.flops for e in prof.key_averages()]) / 1e9:.3f} GFLOPs")

if __name__ == '__main__':
    main()
