import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, ".")

from data.v18.dataset import V18EURSInvarianceDataset
from models.v18.restoration_refiner import build_v18_restoration_refiner
from models.v18.segmenter import build_v18_segmenter
from tools.train_track1_screening import build_dinov2_model, apply_rsuigm_batch_gpu
from models.convnext_fpn import build_convnext_fpn
from tools.v18.run_v18_pipeline import load_checkpoint_with_provenance_check, get_fold_data, validate_manifest_and_folds

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    manifest = validate_manifest_and_folds()
    fold = 0
    fold_data = get_fold_data(manifest, fold)
    train_scenes = fold_data["train_scenes"]
    val_scenes = fold_data["val_scenes"]

    # 1. Load V18
    table_path = f"checkpoints/exp4_1_scale_consistent/table_candidate_B_fold_{fold}.pt"
    rsuigm_table = torch.load(table_path, map_location=device)
    refiner = build_v18_restoration_refiner(rsuigm_table=rsuigm_table, use_neural_refiner=True).to(device)
    ref_weights = load_checkpoint_with_provenance_check(f"checkpoints/v18/fold_{fold}/refiner_best.pt", "ConditionedRestorationRefinerV18", fold, train_scenes, val_scenes)
    refiner.load_state_dict(ref_weights)
    refiner.eval()

    seg_v18 = build_v18_segmenter(num_classes=5, vit_pretrained=False, conv_pretrained=False, use_mock=False).to(device)
    seg_weights = load_checkpoint_with_provenance_check(f"checkpoints/v18/fold_{fold}/segmenter_best.pt", "EURSAPEXV18Segmenter", fold, train_scenes, val_scenes)
    seg_v18.load_state_dict(seg_weights)
    seg_v18.eval()

    # 2. Load DINOv2 Fold 0
    dino_path = f"checkpoints/v16_dinov2/fold_{fold}/best_model.pt"
    dino_model = build_dinov2_model(num_classes=5).to(device)
    dino_ckpt = torch.load(dino_path, map_location=device, weights_only=False)
    dino_model.load_state_dict(dino_ckpt["state_dict"] if "state_dict" in dino_ckpt else dino_ckpt["model_state_dict"])
    dino_model.eval()

    # 3. Load ConvNeXt Fold 0
    conv_path = f"checkpoints/v16_convnext/fold_{fold}/best_model.pt"
    conv_model = build_convnext_fpn(num_classes=5).to(device)
    conv_ckpt = torch.load(conv_path, map_location=device, weights_only=False)
    conv_model.load_state_dict(conv_ckpt["state_dict"] if "state_dict" in conv_ckpt else conv_ckpt["model_state_dict"])
    conv_model.eval()

    # Test on first batch
    ds = V18EURSInvarianceDataset(scenes=val_scenes[:1], crop_size=1024, is_train=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    mean_im = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_im = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    for b in loader:
        deg = b["deg_rgb"].to(device)
        wt = b["wt"].to(device)
        depth = b["depth"].to(device)
        gt = b["mask"].to(device)

        with torch.no_grad():
            # Input pipeline A: V18 refiner output
            rest_v18 = refiner(deg, wt, depth)["refined_image"]

            # Input pipeline B: Pure RSUIGM table B output (leak-free fold 0 table)
            rest_table = apply_rsuigm_batch_gpu(deg, wt, depth, rsuigm_table)

            # Test V18 forward
            s_v18 = seg_v18(rest_v18, deg, wt, depth)
            p_v18 = F.softmax(s_v18["logits"], dim=1)

            # Test DINOv2 with Pipeline A vs Pipeline B
            for name, rest_in in [("V18-Refined", rest_v18), ("RSUIGM-Table-B", rest_table)]:
                x_518 = F.interpolate(rest_in, size=(518, 518), mode="bilinear", align_corners=False)
                x_norm = (x_518 - mean_im) / std_im
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    l_dino = dino_model(x_norm)
                p_dino = F.softmax(l_dino.float(), dim=1)
                p_dino_1024 = F.interpolate(p_dino, size=(1024, 1024), mode="bilinear", align_corners=False)
                print(f"DINOv2 [{name}] prob mean: {p_dino_1024.mean(dim=(0,2,3)).tolist()}")

            # Test ConvNeXt with Pipeline A vs Pipeline B
            for name, rest_in in [("V18-Refined", rest_v18), ("RSUIGM-Table-B", rest_table)]:
                x_512 = F.interpolate(rest_in, size=(512, 512), mode="bilinear", align_corners=False)
                x_norm = (x_512 - mean_im) / std_im
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    l_conv = conv_model(x_norm)
                p_conv = F.softmax(l_conv.float(), dim=1)
                p_conv_1024 = F.interpolate(p_conv, size=(1024, 1024), mode="bilinear", align_corners=False)
                print(f"ConvNeXt [{name}] prob mean: {p_conv_1024.mean(dim=(0,2,3)).tolist()}")

        break
    print("Smoke test completed successfully!")

if __name__ == "__main__":
    main()
