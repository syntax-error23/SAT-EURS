"""
Generate publication-grade, pristine architecture flowcharts for EURS Factsheet.
Designed with generous horizontal gaps (90px), zero label clipping,
high-contrast badges, and clean orthogonal routing.
"""

import os
import math
from PIL import Image, ImageDraw, ImageFont

WIDTH = 2400
HEIGHT = 1050

def get_font(size: int, bold: bool = False):
    font_name = "segoeuib.ttf" if bold else "segoeui.ttf"
    path = os.path.join("C:/Windows/Fonts", font_name)
    if os.path.exists(path):
        return ImageFont.truetype(path, size)
    alt = "arialbd.ttf" if bold else "arial.ttf"
    alt_path = os.path.join("C:/Windows/Fonts", alt)
    if os.path.exists(alt_path):
        return ImageFont.truetype(alt_path, size)
    return ImageFont.load_default()

def draw_card(draw: ImageDraw.ImageDraw, box, fill="#FFFFFF", outline="#E2E8F0", outline_width=2, radius=14):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=outline_width)

def draw_pill(draw: ImageDraw.ImageDraw, xy, text: str, bg_color: str = "#1E293B", text_color: str = "#FFFFFF", font_size: int = 13):
    font = get_font(font_size, bold=True)
    bbox = font.getbbox(text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x, y = xy
    pad_h = 10
    pad_v = 5
    box = [x, y, x + tw + pad_h * 2, y + th + pad_v * 2]
    draw.rounded_rectangle(box, radius=6, fill=bg_color)
    draw.text((x + pad_h, y + pad_v - bbox[1]), text, fill=text_color, font=font)
    return box[2]

def draw_text_lines(draw: ImageDraw.ImageDraw, xy, lines, font, fill="#334155", line_spacing=6):
    x, y = xy
    for line in lines:
        if line == "":
            y += line_spacing * 2
            continue
        draw.text((x, y), line, fill=fill, font=font)
        bbox = font.getbbox(line)
        h = (bbox[3] - bbox[1])
        y += h + line_spacing
    return y

def draw_arrow(draw: ImageDraw.ImageDraw, start, end, color="#475569", width=3, label=None, label_offset=(0, -13)):
    x0, y0 = start
    x1, y1 = end
    draw.line([start, end], fill=color, width=width)

    # Arrowhead
    arrow_size = 13
    angle = math.atan2(y1 - y0, x1 - x0)
    angle1 = angle + math.pi * 0.82
    angle2 = angle - math.pi * 0.82
    p1 = (x1 + arrow_size * math.cos(angle1), y1 + arrow_size * math.sin(angle1))
    p2 = (x1 + arrow_size * math.cos(angle2), y1 + arrow_size * math.sin(angle2))
    draw.polygon([end, p1, p2], fill=color)

    if label:
        f_lbl = get_font(12, bold=True)
        mx = (x0 + x1) / 2 + label_offset[0]
        my = (y0 + y1) / 2 + label_offset[1]
        bbox = f_lbl.getbbox(label)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.rounded_rectangle([mx - tw/2 - 5, my - 2, mx + tw/2 + 5, my + th + 2], radius=4, fill="#FFFFFF", outline=color, width=1)
        draw.text((mx - tw/2, my - bbox[1]), label, fill=color, font=f_lbl)

def draw_elbow_arrow(draw: ImageDraw.ImageDraw, points, color="#475569", width=3, label=None, label_idx=0, label_offset=(0, -13)):
    for i in range(len(points) - 1):
        draw.line([points[i], points[i+1]], fill=color, width=width)
    
    # Arrowhead at final segment
    x0, y0 = points[-2]
    x1, y1 = points[-1]
    arrow_size = 13
    angle = math.atan2(y1 - y0, x1 - x0)
    angle1 = angle + math.pi * 0.82
    angle2 = angle - math.pi * 0.82
    p1 = (x1 + arrow_size * math.cos(angle1), y1 + arrow_size * math.sin(angle1))
    p2 = (x1 + arrow_size * math.cos(angle2), y1 + arrow_size * math.sin(angle2))
    draw.polygon([points[-1], p1, p2], fill=color)

    if label and len(points) >= 2:
        f_lbl = get_font(12, bold=True)
        seg_start = points[label_idx]
        seg_end = points[label_idx + 1]
        mx = (seg_start[0] + seg_end[0]) / 2 + label_offset[0]
        my = (seg_start[1] + seg_end[1]) / 2 + label_offset[1]
        bbox = f_lbl.getbbox(label)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.rounded_rectangle([mx - tw/2 - 5, my - 2, mx + tw/2 + 5, my + th + 2], radius=4, fill="#FFFFFF", outline=color, width=1)
        draw.text((mx - tw/2, my - bbox[1]), label, fill=color, font=f_lbl)


# =============================================================================
# TRACK 1 ARCHITECTURE FLOWCHART
# =============================================================================
def generate_track1_pipeline():
    img = Image.new("RGB", (WIDTH, HEIGHT), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    f_title = get_font(27, bold=True)
    f_sub = get_font(15)
    f_box_h = get_font(17.5, bold=True)
    f_box_sub = get_font(13.5, bold=True)
    f_box_p = get_font(13)

    # Outer border / header
    draw.rounded_rectangle([20, 20, WIDTH - 20, HEIGHT - 20], radius=16, outline="#CBD5E1", width=2)
    draw.text((50, 42), "Team SAT (MIT-BLR) — Track 1: V33-Ensemble-F3Upgrade Complete Perception Pipeline", fill="#0F172A", font=f_title)
    draw.text((50, 80), "Decoupled Physical Radiative Inversion & Dual-Stream Semantic Segmentation | Official Score: 0.6410 Composite (#19359)", fill="#475569", font=f_sub)
    draw.line([(50, 112), (WIDTH - 50, 112)], fill="#E2E8F0", width=2)

    # Coordinates
    # Col 1: 50 .. 330 (w: 280)
    # Gap: 330 .. 420 (90px)
    # Col 2: 420 .. 830 (w: 410)
    # Gap: 830 .. 920 (90px)
    # Col 3: 920 .. 1330 (w: 410)
    # Gap: 1330 .. 1420 (90px)
    # Col 4: 1420 .. 1830 (w: 410)
    # Gap: 1830 .. 1920 (90px)
    # Col 5: 1920 .. 2350 (w: 430)

    # -------------------------------------------------------------------------
    # COL 1: INPUT DEGRADED IMAGERY (x: 50 .. 330, y: 310 .. 790)
    # -------------------------------------------------------------------------
    draw_card(draw, [50, 310, 330, 790], fill="#F8FAFC", outline="#3B82F6", outline_width=2)
    draw.text((68, 330), "Input Degraded Imagery", fill="#1E3A8A", font=f_box_h)
    draw.text((68, 358), "Turbid Multi-Station Capture", fill="#2563EB", font=f_box_sub)
    draw_text_lines(draw, (68, 388), [
        "Raw Underwater RGB Tensor",
        "• Dimensions: Full Test Resolution",
        "  (Aspect-preserved to 1024px)",
        "• 10 Optical Water Types (Jerlov I-9C)",
        "• Attenuation Stations: 5m to 100m",
        "• Severe red-wavelength extinction",
        "• Strong forward/backscatter veiling",
        "• High sensor ISO photon noise floor"
    ], f_box_p, fill="#334155", line_spacing=7)
    draw_pill(draw, (68, 705), "INPUT IMAGE TENSOR", bg_color="#1E3A8A", font_size=12.5)
    draw_pill(draw, (68, 745), "OPTICAL METADATA", bg_color="#0284C7", font_size=12.5)

    # -------------------------------------------------------------------------
    # TOP PATHWAY: RESTORATION CASCADE (y: 140 .. 490)
    # -------------------------------------------------------------------------
    # Stage 1: RSUIGM-19 Physical Basis (x: 420 .. 830)
    draw_card(draw, [420, 140, 830, 490], fill="#F0FDF4", outline="#10B981", outline_width=2)
    draw.text((438, 160), "Stage 1: Physical Radiative Basis", fill="#065F46", font=f_box_h)
    draw.text((438, 188), "RSUIGM-19 Closed-Form Matrix Inversion", fill="#059669", font=f_box_sub)
    draw_text_lines(draw, (438, 218), [
        "• Analytical expansion phi(r,g,b) in R^19 per pixel",
        "• Multi-scale veiling light (1/16 & 1/64 downsampled)",
        "• Closed-form Tikhonov matrix: lambda = 10^-4",
        "• Channel transmission map estimation",
        "• Removes dominant color cast reliably",
        "• Zero high-frequency noise amplification"
    ], f_box_p, fill="#14532D", line_spacing=7)
    draw_pill(draw, (438, 442), "PHYSICAL RADIATIVE INVERSION", bg_color="#059669", font_size=12.5)

    # Stage 2: Bounded NAFNet Refiner (x: 920 .. 1330)
    draw_card(draw, [920, 140, 1330, 490], fill="#FFFFFF", outline="#10B981", outline_width=2)
    draw.text((938, 160), "Stage 2: Bounded NAFNet Refiner", fill="#065F46", font=f_box_h)
    draw.text((938, 188), "Depthwise Gated Residual Restoration", fill="#059669", font=f_box_sub)
    draw_text_lines(draw, (938, 218), [
        "• 16 Non-Linear Activation Free (NAF) blocks",
        "• Ingests 8-ch: [Raw, RSUIGM, WaterType, Depth]",
        "• Condition FiLM modulation across depth levels",
        "• Bounded residual: I_base + 0.10*tanh(NAF)",
        "• Laplacian unsharp filter: alpha = 1.34, sigma = 1.5",
        "• Reaches 25.48 dB PSNR & 0.829 SSIM on test"
    ], f_box_p, fill="#334155", line_spacing=7)
    draw_pill(draw, (938, 442), "RESTORED RGB: 25.48 dB | 0.829 SSIM", bg_color="#047857", font_size=12.5)

    # Deliverable 1: Restored Imagery Output (x: 1420 .. 1830)
    draw_card(draw, [1420, 140, 1830, 490], fill="#ECFDF5", outline="#059669", outline_width=3)
    draw.text((1438, 160), "Output Deliverable 1: Restored RGB", fill="#065F46", font=f_box_h)
    draw.text((1438, 188), "Official PSNR & SSIM Evaluation Head", fill="#047857", font=f_box_sub)
    draw_text_lines(draw, (1438, 218), [
        "• Clean RGB PNG / JPEG at native test resolution",
        "• Restored red-channel luminance fidelity",
        "• Micro-texture preservation across coral substrate",
        "• Zero hallucinated artifacts (unlike GAN baselines)",
        "• Official Scores: 25.48 dB PSNR, 0.829 SSIM",
        "• Feeds directly into downstream Semantic Stream"
    ], f_box_p, fill="#064E3B", line_spacing=7)
    draw_pill(draw, (1438, 442), "OFFICIAL HEAD 1 (RESTORED IMAGES)", bg_color="#065F46", font_size=12.5)

    # -------------------------------------------------------------------------
    # BOTTOM PATHWAY: DUAL-STREAM SEGMENTATION (y: 545 .. 975)
    # -------------------------------------------------------------------------
    # Stream B: ConvNeXt-Base (x: 420 .. 830)
    draw_card(draw, [420, 545, 830, 975], fill="#FAF5FF", outline="#8B5CF6", outline_width=2)
    draw.text((438, 565), "Stream B: ConvNeXt-Base", fill="#581C87", font=f_box_h)
    draw.text((438, 593), "Boundary Inductive Stream (Raw RGB)", fill="#7C3AED", font=f_box_sub)
    draw_text_lines(draw, (438, 625), [
        "• Ingests: Raw Degraded RGB @ 512x512",
        "• 88.5M parameters with 2-stage DetailStem",
        "• Preserves fine organism edges & thin fins",
        "• Resolves the typical 10-15px ViT edge smearing",
        "• Continuous spatial inductive bias under haze",
        "• Retains sharp boundary mask gradients"
    ], f_box_p, fill="#4C1D95", line_spacing=7)
    draw_pill(draw, (438, 925), "88.5M PARAMS | BOUNDARY LOCALIZATION", bg_color="#6D28D9", font_size=12.5)

    # Stream A: DINOv3 ViT-Large (x: 920 .. 1330)
    draw_card(draw, [920, 545, 1330, 975], fill="#EFF6FF", outline="#3B82F6", outline_width=2)
    draw.text((938, 565), "Stream A: DINOv3 ViT-Large/16", fill="#1E3A8A", font=f_box_h)
    draw.text((938, 593), "Global Semantic Stream (Restored RGB)", fill="#2563EB", font=f_box_sub)
    draw_text_lines(draw, (938, 625), [
        "• Ingests: Restored RGB @ 518x518 (patch size 14)",
        "• 304M parameters pre-trained on LVD-142M dataset",
        "• Extracts invariant high-level semantic tokens",
        "• Multi-level intermediate taps: Blocks 5, 11, 17, 23",
        "• Broad receptive field for long-range water context",
        "• High categorical separability across 5 marine classes"
    ], f_box_p, fill="#1E3A8A", line_spacing=7)
    draw_pill(draw, (938, 925), "304M PARAMS | GLOBAL SEMANTIC CONTEXT", bg_color="#1D4ED8", font_size=12.5)

    # Fusion Neck: Learned Spatial Gate & UPerNet (x: 1420 .. 1830)
    draw_card(draw, [1420, 545, 1830, 975], fill="#FFFBEB", outline="#F59E0B", outline_width=2)
    draw.text((1438, 565), "Learned Spatial Gate & UPerNet", fill="#92400E", font=f_box_h)
    draw.text((1438, 593), "Dynamic Pixel-Wise Multi-Scale Fusion", fill="#D97706", font=f_box_sub)
    draw_text_lines(draw, (1438, 625), [
        "• Pixel-wise gating: f_fused = G*f_sem + (1-G)*f_conv",
        "• Gate G computed via 3x3 conv over joint features",
        "• Dynamically trusts ViT in open water, CNN on edges",
        "• Condition FiLM modulation with optical water type",
        "• Progressive multi-scale aggregation (strides 4, 8, 16, 32)",
        "• Auxiliary boundary & objectness supervision heads"
    ], f_box_p, fill="#78350F", line_spacing=7)
    draw_pill(draw, (1438, 925), "DYNAMIC PIXEL-WISE FUSION NECK", bg_color="#D97706", font_size=12.5)

    # -------------------------------------------------------------------------
    # RIGHT COLUMN: ENSEMBLE, CALIBRATION & DELIVERABLE 2 (x: 1920 .. 2350)
    # -------------------------------------------------------------------------
    draw_card(draw, [1920, 140, 2350, 975], fill="#FEF2F2", outline="#EF4444", outline_width=3)
    draw.text((1938, 160), "Ensemble & Deliverable 2", fill="#991B1B", font=f_box_h)
    draw.text((1938, 188), "5-Fold F3Upgrade & Logit Calibration", fill="#DC2626", font=f_box_sub)
    draw_text_lines(draw, (1938, 218), [
        "5-Fold Cross-Validation Ensemble:",
        "• Combines Folds 0, 1, 2, 4 + upgraded Fold 3",
        "• Fold 3 retrained with rare copy-paste fauna",
        "• Horizontal flip test-time augmentation (TTA)",
        "• 10 total forward passes per test image",
        "• Fused logits averaged across predictions",
        "",
        "Rare-Class Threshold Calibration:",
        "• Delta Z = [0.0, 0.0, 0.0, +0.10, +0.25]",
        "• Elasmobranch boost: +0.10 offset",
        "• Macro-Fauna boost: +0.25 offset",
        "• Protects against Zero-TP penalty collapse",
        "• argmax -> uint8 class masks {0, 1, 2, 3, 4}",
        "",
        "Official Competition Benchmark Results:",
        "• Composite Score: 0.6410 (#19359)",
        "• Q_seg: 0.5820   |  mIoU: 0.7110",
        "• Macro F1: 0.8180 |  PSNR: 25.48 dB",
        "• SSIM: 0.8290   |  5-Class Compliant"
    ], f_box_p, fill="#7F1D1D", line_spacing=6.5)
    draw_pill(draw, (1938, 885), "OFFICIAL HEAD 2 (5-CLASS MASKS)", bg_color="#991B1B", font_size=12.5)
    draw_pill(draw, (1938, 925), "CHAMPION SUBMISSION #19359", bg_color="#1E293B", font_size=12.5)

    # -------------------------------------------------------------------------
    # ARROWS & ROUTING (PRISTINE, NON-OVERLAPPING)
    # -------------------------------------------------------------------------
    # 1. Input -> Stage 1 Physical Basis (Top Branch)
    draw_elbow_arrow(draw, [(330, 420), (375, 420), (375, 315), (420, 315)], color="#10B981", width=3, label="Raw RGB", label_idx=0)

    # 2. Input -> Stream B ConvNeXt (Bottom Branch)
    draw_elbow_arrow(draw, [(330, 680), (375, 680), (375, 760), (420, 760)], color="#8B5CF6", width=3, label="Raw RGB", label_idx=0)

    # 3. Stage 1 -> Stage 2 Refiner (Horizontal)
    draw_arrow(draw, (830, 315), (920, 315), color="#10B981", width=3, label="RSUIGM")

    # 4. Stage 2 -> Deliverable 1 Restored Output (Horizontal)
    draw_arrow(draw, (1330, 315), (1420, 315), color="#059669", width=3, label="Restored RGB")

    # 5. Stage 2 -> Stream A DINOv3 (Straight Vertical Down!)
    draw_arrow(draw, (1125, 490), (1125, 545), color="#3B82F6", width=3, label="Restored RGB", label_offset=(54, 0))

    # 6. Stream B ConvNeXt -> Spatial Gate & UPerNet (Clean Orthogonal underneath Stream A)
    draw_elbow_arrow(draw, [
        (625, 975),
        (625, 1005),
        (1625, 1005),
        (1625, 975)
    ], color="#8B5CF6", width=3, label="ConvNeXt Inductive Boundary Features", label_idx=1)

    # 7. Stream A DINOv3 -> Spatial Gate & UPerNet (Horizontal)
    draw_arrow(draw, (1330, 760), (1420, 760), color="#3B82F6", width=3, label="DINOv3")

    # 8. Spatial Gate Neck -> Ensemble & Deliverable 2 (Horizontal)
    draw_arrow(draw, (1830, 760), (1920, 760), color="#EF4444", width=3, label="Fused Logits")

    out = img.resize((WIDTH // 2, HEIGHT // 2), Image.Resampling.LANCZOS)
    out.save("factsheet-template/images/method_overview.png")
    out.save("docs/track1_pipeline_diagram.png")
    print("[DIAGRAM] Generated perfected, non-overlapping Track 1 flowchart!")


# =============================================================================
# TRACK 2 ARCHITECTURE FLOWCHART (JETSON AGX ORIN)
# =============================================================================
def generate_track2_pipeline():
    img = Image.new("RGB", (WIDTH, HEIGHT), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    f_title = get_font(27, bold=True)
    f_sub = get_font(15)
    f_box_h = get_font(17.5, bold=True)
    f_box_sub = get_font(13.5, bold=True)
    f_box_p = get_font(13)

    # Outer border / header
    draw.rounded_rectangle([20, 20, WIDTH - 20, HEIGHT - 20], radius=16, outline="#CBD5E1", width=2)
    draw.text((50, 42), "Team SAT (MIT-BLR) — Track 2: track2-apex-orin-v3 Embedded Edge Pipeline", fill="#042F2E", font=f_title)
    draw.text((50, 80), "Single-Graph Multi-Task Perception on NVIDIA Jetson AGX Orin | 0.884M Params (Cap: 1.0M) | 9.82 GFLOPs (Cap: 12.0G)", fill="#0F766E", font=f_sub)
    draw.line([(50, 112), (WIDTH - 50, 112)], fill="#E2E8F0", width=2)

    # Coordinates
    # Col 1: 50 .. 330 (w: 280)
    # Gap: 330 .. 420 (90px)
    # Col 2: 420 .. 830 (w: 410)
    # Gap: 830 .. 920 (90px)
    # Col 3: 920 .. 1330 (w: 410)
    # Gap: 1330 .. 1420 (90px)
    # Col 4: 1420 .. 1830 (w: 410)
    # Gap: 1830 .. 1920 (90px)
    # Col 5: 1920 .. 2350 (w: 430)

    # -------------------------------------------------------------------------
    # COL 1: STATIC INPUT TENSOR (x: 50 .. 330, y: 310 .. 790)
    # -------------------------------------------------------------------------
    draw_card(draw, [50, 310, 330, 790], fill="#F0FDFA", outline="#0D9488", outline_width=2)
    draw.text((68, 330), "Static Input Tensor", fill="#042F2E", font=f_box_h)
    draw.text((68, 358), "NVIDIA Jetson AGX Orin Engine", fill="#0F766E", font=f_box_sub)
    draw_text_lines(draw, (68, 388), [
        "Fixed-Shape Engine Ingestion:",
        "• Shape: [1, 3, 1024, 1024]",
        "• Format: Float32 [0.0, 1.0]",
        "• Static memory allocation eliminates",
        "  runtime TensorRT buffer fragmentation",
        "• Bound zero-copy GPU shared memory",
        "• Dual consumption: Feeds radiometric",
        "  router and multi-task shared stem",
        "• 100% Whitelist operator graph"
    ], f_box_p, fill="#134E4A", line_spacing=7)
    draw_pill(draw, (68, 705), "STATIC INPUT TENSOR", bg_color="#0F766E", font_size=12.5)
    draw_pill(draw, (68, 745), "ZERO DYNAMIC ALLOCS", bg_color="#042F2E", font_size=12.5)

    # -------------------------------------------------------------------------
    # TOP PATHWAY: RESTORATION ENGINE (y: 140 .. 490)
    # -------------------------------------------------------------------------
    # Box 1: 18-D Radiometric Router (x: 420 .. 830)
    draw_card(draw, [420, 140, 830, 490], fill="#FFFFFF", outline="#0D9488", outline_width=2)
    draw.text((438, 160), "1. 18-D Radiometric Router", fill="#042F2E", font=f_box_h)
    draw.text((438, 188), "Fast Matrix Contraction (<0.30 GFLOPs)", fill="#0D9488", font=f_box_sub)
    draw_text_lines(draw, (438, 218), [
        "• 18 translation-invariant global statistics",
        "  (channel means, std, min/max, Sobel edges)",
        "• Downsampled to 256x256 via stride-4 pool",
        "• 2-layer LayerNorm MLP with OnnxSafeHardswish",
        "• Dynamic routing over 20-scene buffer [200, 57]",
        "• Analytical 19-term polynomial basis evaluation",
        "• 95% compute saving over spatial 2D convs"
    ], f_box_p, fill="#134E4A", line_spacing=7)
    draw_pill(draw, (438, 442), "<0.30 GFLOPS | DYNAMIC ROUTER", bg_color="#0D9488", font_size=12.5)

    # Box 2: FiLM Spatial Refiner @ 256px (x: 920 .. 1330)
    draw_card(draw, [920, 140, 1330, 490], fill="#FFFFFF", outline="#059669", outline_width=2)
    draw.text((938, 160), "2. FiLM Spatial Refiner @ 256px", fill="#065F46", font=f_box_h)
    draw.text((938, 188), "Low-Resolution Residual Polish (+6.15 dB)", fill="#059669", font=f_box_sub)
    draw_text_lines(draw, (938, 218), [
        "• 3 depthwise-separable inverted bottleneck blocks",
        "• Operates exclusively on downsampled 256x256 grid",
        "• 32-D condition FiLM modulation with depth levels",
        "• Bounded offset: I_ref = I_base + 0.25*tanh(Conv)",
        "• In-graph 3x3 Gaussian unsharp filter (gamma = 0.16)",
        "• Bilinear upsampling to 1024x1024 restored image",
        "• Lifts PSNR from 17.1 dB to 23.25 dB (+6.15 dB)"
    ], f_box_p, fill="#064E3B", line_spacing=7)
    draw_pill(draw, (938, 442), "RESTORED RGB: 23.25 dB | 0.847 SSIM", bg_color="#059669", font_size=12.5)

    # Box 3: Output Deliverable 1: Edge Restored RGB (x: 1420 .. 1830)
    draw_card(draw, [1420, 140, 1830, 490], fill="#ECFDF5", outline="#059669", outline_width=3)
    draw.text((1438, 160), "Output Deliverable 1: Restored", fill="#065F46", font=f_box_h)
    draw.text((1438, 188), "TensorRT Native Output Head 1", fill="#047857", font=f_box_sub)
    draw_text_lines(draw, (1438, 218), [
        "• Output tensor: [1, 3, 1024, 1024] Float32 in [0, 1]",
        "• Recovers true natural color balance and contrast",
        "• Edge-sharpened high-frequency coral textures",
        "• 100% compliant with Jetson AGX Orin whitelist",
        "• Zero post-processing or external unsharp tools",
        "• Evaluated officially: 23.25 dB PSNR, 0.847 SSIM",
        "• Re-ingested into downstream dual stem in-graph"
    ], f_box_p, fill="#047857", line_spacing=7)
    draw_pill(draw, (1438, 442), "OFFICIAL HEAD 1: RESTORED IMAGE", bg_color="#047857", font_size=12.5)

    # -------------------------------------------------------------------------
    # BOTTOM PATHWAY: MULTI-TASK SEGMENTATION (y: 545 .. 975)
    # -------------------------------------------------------------------------
    # Box 4: Shared Dual Stem (x: 420 .. 830)
    draw_card(draw, [420, 545, 830, 975], fill="#F0F9FF", outline="#0284C7", outline_width=2)
    draw.text((438, 565), "3. Shared Dual Stem & Fusion", fill="#075985", font=f_box_h)
    draw.text((438, 593), "Parallel Ingestion (Raw RGB + Restored RGB)", fill="#0284C7", font=f_box_sub)
    draw_text_lines(draw, (438, 625), [
        "• Restored RGB (from Refiner) + Raw RGB (from Input)",
        "• Dual 3x3 stride-2 convs with zero-init BatchNorm gamma",
        "• Preserves ImageNet pretrained trunk weights at step 0",
        "• Zero representational drift during multi-task training",
        "• Merges restored color clarity with raw edge gradients",
        "• Feeds directly into truncated MobileNetV3 trunk"
    ], f_box_p, fill="#0C4A6E", line_spacing=7)
    draw_pill(draw, (438, 925), "ZERO INITIAL DRIFT DUAL STEM", bg_color="#0284C7", font_size=12.5)

    # Box 5: Truncated MobileNetV3 Backbone (x: 920 .. 1330)
    draw_card(draw, [920, 545, 1330, 975], fill="#FFFFFF", outline="#0284C7", outline_width=2)
    draw.text((938, 565), "4. Truncated MobileNetV3-Large", fill="#075985", font=f_box_h)
    draw.text((938, 593), "Param-Capped Edge Backbone (Cut @ Block 11)", fill="#0369A1", font=f_box_sub)
    draw_text_lines(draw, (938, 625), [
        "• Truncated strictly after Block 11 (features[:12])",
        "• Eliminates Block 12 heavy 386k pointwise expansion",
        "• Backbone parameters: only 406.4k (Total model: 884.1k)",
        "• Passes official 1.0M cap with 11.6% safety headroom",
        "• Multi-scale stage taps: S4 (24ch), S8 (40ch), S16 (112ch)",
        "• Mobile inverted bottlenecks with depthwise 3x3/5x5"
    ], f_box_p, fill="#334155", line_spacing=7)
    draw_pill(draw, (938, 925), "0.884M TOTAL PARAMS (11.6% BUFFER)", bg_color="#0369A1", font_size=12.5)

    # Box 6: ASPP, FPN & Speckle Suppressor (x: 1420 .. 1830)
    draw_card(draw, [1420, 545, 1830, 975], fill="#FFFBEB", outline="#D97706", outline_width=2)
    draw.text((1438, 565), "5. ASPP, FPN & Speckle Filter", fill="#92400E", font=f_box_h)
    draw.text((1438, 593), "In-Graph Speckle Suppressor (TensorRT-Native)", fill="#D97706", font=f_box_sub)
    draw_text_lines(draw, (1438, 625), [
        "• Dilated ASPP (rates 1, 2, 4) + ScenePriorGate context",
        "• 4-scale progressive FPN with 72-ch depthwise skips",
        "• Pointwise 5-class linear classifier head @ 256x256",
        "• In-Graph Spatial Speckle Suppressor:",
        "  P = Softmax(Z_256, dim=1)",
        "  Support = AvgPool2d(P, 11, stride=1, pad=5)",
        "  Cleaned Z = Z - (1.2/0.04)*ReLU(0.04 - Support)",
        "• Zero CPU latency: 100% inside TensorRT FP16 engine"
    ], f_box_p, fill="#78350F", line_spacing=7)
    draw_pill(draw, (1438, 925), "0 EXTRA PARAMS | TENSORRT-NATIVE", bg_color="#D97706", font_size=12.5)

    # -------------------------------------------------------------------------
    # RIGHT COLUMN: DELIVERABLE 2 (SEG LOGITS) & JETSON BENCHMARK (x: 1920 .. 2350)
    # -------------------------------------------------------------------------
    draw_card(draw, [1920, 140, 2350, 975], fill="#FEF2F2", outline="#DC2626", outline_width=3)
    draw.text((1938, 160), "Deliverable 2: Segmentation", fill="#991B1B", font=f_box_h)
    draw.text((1938, 188), "Jetson AGX Orin Verified Benchmarks", fill="#DC2626", font=f_box_sub)
    draw_text_lines(draw, (1938, 218), [
        "Final Segmentation Output Head:",
        "• Output Logits: [1, 5, 1024, 1024] Float32",
        "• Bilinear upsampling from 256px cleaned logits",
        "• 5 discrete categorical classes:",
        "  {Background, Substrate, Fish,",
        "   Elasmobranch, Macro-Fauna}",
        "",
        "NVIDIA Jetson AGX Orin Edge SOTA:",
        "• Complexity: 9.82 GFLOPs",
        "  (Official cap: 12.0 GFLOPs, 18.2% headroom)",
        "• Model Parameters: 0.884 M",
        "  (Official cap: 1.0 M, 11.6% headroom)",
        "• Latency: 5.8 ms / frame (172.4 FPS)",
        "  on TensorRT 8.6 FP16 engine (2048-core GPU)",
        "• CPU Latency: 43.6 ms (22.9 FPS) on Cortex-A78AE",
        "• Energy: ~0.17 J / frame on 30W power mode",
        "• Official Leaderboard Score: 0.4070 Composite",
        "• Verified Bit-Identical ONNX Export (#19447)"
    ], f_box_p, fill="#7F1D1D", line_spacing=6.5)
    draw_pill(draw, (1938, 885), "OUTPUT HEAD 2: SEG_LOGITS", bg_color="#DC2626", font_size=12.5)
    draw_pill(draw, (1938, 925), "SUBMISSION #19447 (0.4070)", bg_color="#042F2E", font_size=12.5)

    # -------------------------------------------------------------------------
    # ARROWS & ROUTING (PRISTINE, NON-OVERLAPPING)
    # -------------------------------------------------------------------------
    # 1. Input -> Router (Top Branch)
    draw_elbow_arrow(draw, [(330, 420), (375, 420), (375, 315), (420, 315)], color="#0D9488", width=3, label="Downsampled", label_idx=0)

    # 2. Input -> Dual Stem (Bottom Branch)
    draw_elbow_arrow(draw, [(330, 680), (375, 680), (375, 760), (420, 760)], color="#0284C7", width=3, label="Raw RGB", label_idx=0)

    # 3. Router -> Spatial Refiner (Horizontal)
    draw_arrow(draw, (830, 315), (920, 315), color="#0D9488", width=3, label="Basis Matrix")

    # 4. Spatial Refiner -> Deliverable 1 Restored RGB (Horizontal)
    draw_arrow(draw, (1330, 315), (1420, 315), color="#059669", width=3, label="Restored RGB")

    # 5. Spatial Refiner -> Shared Dual Stem (Clean Orthogonal Down-Left to Dual Stem!)
    draw_elbow_arrow(draw, [
        (1125, 490),
        (1125, 518),
        (625, 518),
        (625, 545)
    ], color="#059669", width=3, label="Restored RGB (In-Graph)", label_idx=1)

    # 6. Shared Dual Stem -> Truncated MobileNetV3 Backbone (Horizontal)
    draw_arrow(draw, (830, 760), (920, 760), color="#0284C7", width=3, label="Fused Stem")

    # 7. Truncated MobileNetV3 -> ASPP, FPN & Speckle Suppressor (Horizontal)
    draw_arrow(draw, (1330, 760), (1420, 760), color="#D97706", width=3, label="Multi-Scale")

    # 8. FPN & Speckle Suppressor -> Deliverable 2 Segmentation Output (Horizontal)
    draw_arrow(draw, (1830, 760), (1920, 760), color="#DC2626", width=3, label="5-ch Logits")

    out = img.resize((WIDTH // 2, HEIGHT // 2), Image.Resampling.LANCZOS)
    out.save("factsheet-template/images/track2_pipeline.png")
    out.save("docs/track2_pipeline_diagram.png")
    print("[DIAGRAM] Generated perfected, non-overlapping Track 2 flowchart!")


if __name__ == "__main__":
    generate_track1_pipeline()
    generate_track2_pipeline()
