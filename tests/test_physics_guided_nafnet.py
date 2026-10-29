import sys
sys.path.insert(0, ".")
import torch
from models.physics_guided_nafnet import PhysicsGuidedNAFNet
from losses.restoration_v7_loss import RestorationV7Loss

def test_nafnet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 76)
    print("TESTING PhysicsGuidedNAFNet & RestorationV7Loss")
    print(f"  Device: {device}")
    print("=" * 76)

    # 1. Shape & Bounds Verification
    print("\n[Test 1/4] Verifying forward shapes and physical [0, 1] bounds...")
    net = PhysicsGuidedNAFNet(residual_scale=0.25).to(device)
    b, h, w = 2, 256, 256
    i_deg = torch.rand(b, 3, h, w, device=device)
    j_rsuigm = torch.rand(b, 3, h, w, device=device)
    wt = torch.ones(b, 1, h, w, device=device) * 0.5
    dp = torch.ones(b, 1, h, w, device=device) * 0.3

    with torch.no_grad():
        out = net(i_deg, j_rsuigm, wt, dp)

    assert out.shape == (b, 3, h, w), f"Expected {(b, 3, h, w)}, got {out.shape}"
    assert out.min() >= 0.0 and out.max() <= 1.0, f"Bounds violated! min={out.min()}, max={out.max()}"
    print(f"  Shape check passed: {out.shape} | Range: [{out.min():.4f}, {out.max():.4f}] -> PASS")

    # 2. Zero-Residual Guarantee at Step 0
    print("\n[Test 2/4] Verifying Zero-Residual Guarantee at Step 0...")
    dev = (out - j_rsuigm).abs().max().item()
    print(f"  Max absolute deviation from RSUIGM prior: {dev:.8f}")
    assert dev < 1e-6, f"Initial deviation {dev} exceeds threshold! Ending layer must be zero-initialized."
    print("  Zero-Residual Anchor: PASS (Model starts identically at the RSUIGM baseline!)")

    # 3. Residual Boundedness Check
    print("\n[Test 3/4] Verifying maximum residual is bounded by 0.25...")
    # Add large weights to ending layer to test maximum possible residual
    with torch.no_grad():
        net.ending.weight.fill_(10.0)
        out_extreme = net(i_deg, j_rsuigm, wt, dp)
        max_res = (out_extreme - j_rsuigm).abs().max().item()
    print(f"  Maximum residual under extreme saturation: {max_res:.4f}")
    assert max_res <= 0.250001, f"Residual exceeded bound! max_res={max_res}"
    print("  Residual Boundedness: PASS (Strictly <= 0.25)")

    # 4. Gradient Flow & Loss Stability
    print("\n[Test 4/4] Verifying Gradient Flow & RestorationV7Loss...")
    net = PhysicsGuidedNAFNet(residual_scale=0.25).to(device)
    net.train()
    criterion = RestorationV7Loss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    optimizer.zero_grad()

    target = torch.rand(b, 3, h, w, device=device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred = net(i_deg, j_rsuigm, wt, dp)
        loss, loss_dict = criterion(pred.float(), target.float())

    loss.backward()

    # Step to make ending layer non-zero
    optimizer.step()
    optimizer.zero_grad()

    # Step 2: verify gradients flow to encoders
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred2 = net(i_deg, j_rsuigm, wt, dp)
        loss2, loss_dict2 = criterion(pred2.float(), target.float())
    loss2.backward()

    has_encoder_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.encoders.parameters())
    has_decoder_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.decoders.parameters())
    has_ending_grad = net.ending.weight.grad is not None and net.ending.weight.grad.abs().sum() > 0

    print(f"  Encoder Gradients: {has_encoder_grad} | Decoder Gradients: {has_decoder_grad} | Ending Gradients: {has_ending_grad}")
    assert has_encoder_grad and has_decoder_grad and has_ending_grad, "Gradient flow broken!"
    print(f"  Loss breakdown: {loss_dict2}")
    print("  Gradient Flow & Loss Stability: PASS")

    print("\n" + "=" * 76)
    print("ALL PhysicsGuidedNAFNet & Loss TESTS PASSED 100%!")
    print("=" * 76)

if __name__ == '__main__':
    test_nafnet()
