"""
Generate ultra-crisp, publication-grade high-DPI architecture flowcharts for EURS Factsheet.
Native 2520x1100 resolution (360 DPI in ReportLab at 504x220 pt), large high-contrast fonts,
zero blur, zero clipping, and generous margins.
"""

import os
import math
from PIL import Image, ImageDraw, ImageFont

WIDTH = 2520
HEIGHT = 1100

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

def draw_card(draw: ImageDraw.ImageDraw, box, fill="#FFFFFF", outline="#CBD5E1", outline_width=3, radius=16):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=outline_width)

def draw_pill(draw: ImageDraw.ImageDraw, xy, text: str, bg_color: str = "#1E293B", text_color: str = "#FFFFFF", font_size: int = 18):
    font = get_font(font_size, bold=True)
    bbox = font.getbbox(text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x, y = xy
    pad_h = 16
    pad_v = 7
    box = [x, y, x + tw + pad_h * 2, y + th + pad_v * 2]
    draw.rounded_rectangle(box, radius=8, fill=bg_color)
    draw.text((x + pad_h, y + pad_v - bbox[1]), text, fill=text_color, font=font)
    return box[2]

def draw_text_lines(draw: ImageDraw.ImageDraw, xy, lines, font, fill="#1E293B", line_spacing=8):
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

def draw_arrow(draw: ImageDraw.ImageDraw, start, end, color="#475569", width=4, label=None, label_offset=(0, -18)):
    x0, y0 = start
    x1, y1 = end
    draw.line([start, end], fill=color, width=width)

    # Arrowhead
    arrow_size = 18
    angle = math.atan2(y1 - y0, x1 - x0)
    angle1 = angle + math.pi * 0.82
    angle2 = angle - math.pi * 0.82
    p1 = (x1 + arrow_size * math.cos(angle1), y1 + arrow_size * math.sin(angle1))
    p2 = (x1 + arrow_size * math.cos(angle2), y1 + arrow_size * math.sin(angle2))
    draw.polygon([end, p1, p2], fill=color)

    if label:
        f_lbl = get_font(17, bold=True)
        mx = (x0 + x1) / 2 + label_offset[0]
        my = (y0 + y1) / 2 + label_offset[1]
        bbox = f_lbl.getbbox(label)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.rounded_rectangle([mx - tw/2 - 8, my - 4, mx + tw/2 + 8, my + th + 4], radius=6, fill="#FFFFFF", outline=color, width=2)
        draw.text((mx - tw/2, my - bbox[1]), label, fill=color, font=f_lbl)

def draw_elbow_arrow(draw: ImageDraw.ImageDraw, points, color="#475569", width=4, label=None, label_idx=0, label_offset=(0, -18)):
    for i in range(len(points) - 1):
        draw.line([points[i], points[i+1]], fill=color, width=width)
    
    # Arrowhead at final segment
    x0, y0 = points[-2]
    x1, y1 = points[-1]
    arrow_size = 18
    angle = math.atan2(y1 - y0, x1 - x0)
    angle1 = angle + math.pi * 0.82
    angle2 = angle - math.pi * 0.82
    p1 = (x1 + arrow_size * math.cos(angle1), y1 + arrow_size * math.sin(angle1))
    p2 = (x1 + arrow_size * math.cos(angle2), y1 + arrow_size * math.sin(angle2))
    draw.polygon([points[-1], p1, p2], fill=color)

    if label and len(points) >= 2:
        f_lbl = get_font(17, bold=True)
        seg_start = points[label_idx]
        seg_end = points[label_idx + 1]
        mx = (seg_start[0] + seg_end[0]) / 2 + label_offset[0]
        my = (seg_start[1] + seg_end[1]) / 2 + label_offset[1]
        bbox = f_lbl.getbbox(label)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.rounded_rectangle([mx - tw/2 - 8, my - 4, mx + tw/2 + 8, my + th + 4], radius=6, fill="#FFFFFF", outline=color, width=2)
        draw.text((mx - tw/2, my - bbox[1]), label, fill=color, font=f_lbl)


# =============================================================================
# TRACK 1 ARCHITECTURE FLOWCHART
# =============================================================================
def generate_track1_pipeline():
    img = Image.new("RGB", (WIDTH, HEIGHT), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    f_title = get_font(36, bold=True)
    f_sub = get_font(21)
    f_box_h = get_font(25, bold=True)
    f_box_sub = get_font(18.5, bold=True)
    f_box_p = get_font(18)

    # Outer border / header
    draw.rounded_rectangle([20, 20, WIDTH - 20, HEIGHT - 20], radius=18, outline="#CBD5E1", width=3)
    draw.text((50, 42), "Team SAT (MIT-BLR) — Track 1: V33-Ensemble-F3Upgrade Perception Pipeline", fill="#0F172A", font=f_title)
    draw.text((50, 88), "Physical Radiative Inversion & Dual-Stream Semantic Segmentation | Official Score: 0.6410 Composite (#19359)", fill="#475569", font=f_sub)
    draw.line([(50, 122), (WIDTH - 50, 122)], fill="#E2E8F0", width=2)

    # Coordinates:
    # Col 1: 50 .. 350 (w: 300)
    # Gap: 350 .. 450 (100px)
    # Col 2: 450 .. 880 (w: 430)
    # Gap: 880 .. 980 (100px)
    # Col 3: 980 .. 1410 (w: 430)
    # Gap: 1410 .. 1510 (100px)
    # Col 4: 1510 .. 1940 (w: 430)
    # Gap: 1940 .. 2040 (100px)
    # Col 5: 2040 .. 2470 (w: 430)

    # -------------------------------------------------------------------------
    # COL 1: INPUT DEGRADED IMAGERY (x: 50 .. 350, y: 310 .. 830)
    # -------------------------------------------------------------------------
    draw_card(draw, [50, 310, 350, 830], fill="#F8FAFC", outline="#3B82F6", outline_width=3)
    draw.text((70, 332), "Input Degraded Imagery", fill="#1E3A8A", font=f_box_h)
    draw.text((70, 365), "Multi-Station Underwater Capture", fill="#2563EB", font=f_box_sub)
    draw_text_lines(draw, (70, 402), [
        "Raw Underwater RGB Tensor",
        "• Dimensions: Native Test Res",
        "  (Aspect-preserved to 1024px)",
        "• 10 Optical Water Types (1 to 10)",
        "• Attenuation: 5m to 100m depth",
        "• Severe red-spectrum extinction",
        "• Forward & backscatter haze",
        "• High sensor ISO photon noise"
    ], f_box_p, fill="#1E293B", line_spacing=8)
    draw_pill(draw, (70, 725), "INPUT IMAGE TENSOR", bg_color="#1E3A8A", font_size=16)
    draw_pill(draw, (70, 775), "OPTICAL METADATA", bg_color="#0284C7", font_size=16)

    # -------------------------------------------------------------------------
    # TOP PATHWAY: RESTORATION CASCADE (y: 150 .. 520)
    # -------------------------------------------------------------------------
    # Stage 1: RSUIGM-19 Physical Basis (x: 450 .. 880)
    draw_card(draw, [450, 150, 880, 520], fill="#F0FDF4", outline="#10B981", outline_width=3)
    draw.text((470, 172), "Stage 1: Physical Basis", fill="#065F46", font=f_box_h)
    draw.text((470, 205), "RSUIGM-19 Polynomial Inversion", fill="#059669", font=f_box_sub)
    draw_text_lines(draw, (470, 242), [
        "• Analytical expansion phi(r,g,b) in R^19",
        "• Multi-scale veiling light (1/16, 1/64 pool)",
        "• Closed-form Tikhonov: lambda = 10^-4",
        "• Global transmission per channel",
        "• Removes severe wavelength color cast",
        "• Zero high-frequency noise amplification"
    ], f_box_p, fill="#14532D", line_spacing=8)
    draw_pill(draw, (470, 465), "PHYSICAL RADIATIVE INVERSION", bg_color="#059669", font_size=16)

    # Stage 2: Bounded NAFNet Refiner (x: 980 .. 1410)
    draw_card(draw, [980, 150, 1410, 520], fill="#FFFFFF", outline="#10B981", outline_width=3)
    draw.text((1000, 172), "Stage 2: Bounded NAFNet", fill="#065F46", font=f_box_h)
    draw.text((1000, 205), "Depthwise Gated Neural Refiner", fill="#059669", font=f_box_sub)
    draw_text_lines(draw, (1000, 242), [
        "• 16 Non-Linear Activation Free blocks",
        "• Ingests 8-ch: [Raw, RSUIGM, WT, Depth]",
        "• Condition FiLM across depth stations",
        "• Bounded offset: I_base + 0.10*tanh(NAF)",
        "• Laplacian unsharp: alpha = 1.34, sigma = 1.5",
        "• Reaches 25.48 dB PSNR & 0.829 SSIM"
    ], f_box_p, fill="#1E293B", line_spacing=8)
    draw_pill(draw, (1000, 465), "RESTORED RGB: 25.48 dB | 0.829 SSIM", bg_color="#047857", font_size=16)

    # Deliverable 1: Restored Imagery Output (x: 1510 .. 1940)
    draw_card(draw, [1510, 150, 1940, 520], fill="#ECFDF5", outline="#059669", outline_width=3)
    draw.text((1530, 172), "Output 1: Restored RGB", fill="#065F46", font=f_box_h)
    draw.text((1530, 205), "Official PSNR & SSIM Evaluation Head", fill="#047857", font=f_box_sub)
    draw_text_lines(draw, (1530, 242), [
        "• Clean RGB PNG / JPEG at native test res",
        "• Restored red-channel luminance fidelity",
        "• Micro-texture preservation on coral reef",
        "• Zero hallucinated artifacts (unlike GANs)",
        "• Official Scores: 25.48 dB PSNR, 0.829 SSIM",
        "• Feeds directly into Semantic Stream"
    ], f_box_p, fill="#064E3B", line_spacing=8)
    draw_pill(draw, (1530, 465), "OFFICIAL HEAD 1: RESTORED IMAGES", bg_color="#065F46", font_size=16)

    # -------------------------------------------------------------------------
    # BOTTOM PATHWAY: DUAL-STREAM SEGMENTATION (y: 575 .. 1025)
    # -------------------------------------------------------------------------
    # Stream B: ConvNeXt-Base (x: 450 .. 880)
    draw_card(draw, [450, 575, 880, 1025], fill="#FAF5FF", outline="#8B5CF6", outline_width=3)
    draw.text((470, 597), "Stream B: ConvNeXt-Base", fill="#581C87", font=f_box_h)
    draw.text((470, 630), "Boundary Inductive Stream (Raw RGB)", fill="#7C3AED", font=f_box_sub)
    draw_text_lines(draw, (470, 667), [
        "• Ingests: Raw Degraded RGB @ 512x512",
        "• 88.5M parameters with 2-stage DetailStem",
        "• Preserves fine organism edges & thin fins",
        "• Resolves typical 10-15px ViT edge smearing",
        "• Continuous spatial inductive bias under haze",
        "• Retains sharp boundary mask gradients"
    ], f_box_p, fill="#4C1D95", line_spacing=8)
    draw_pill(draw, (470, 965), "88.5M PARAMS | BOUNDARY LOCALIZATION", bg_color="#6D28D9", font_size=16)

    # Stream A: DINOv3 ViT-Large (x: 980 .. 1410)
    draw_card(draw, [980, 575, 1410, 1025], fill="#EFF6FF", outline="#3B82F6", outline_width=3)
    draw.text((1000, 597), "Stream A: DINOv3 ViT-Large", fill="#1E3A8A", font=f_box_h)
    draw.text((1000, 630), "Global Semantic Stream (Restored RGB)", fill="#2563EB", font=f_box_sub)
    draw_text_lines(draw, (1000, 667), [
        "• Ingests: Restored RGB @ 518x518 (patch 14)",
        "• 304M parameters pre-trained on LVD-142M",
        "• Invariant high-level semantic tokens",
        "• Multi-level intermediate taps: 5, 11, 17, 23",
        "• Broad receptive field for global water context",
        "• High separability across 5 marine classes"
    ], f_box_p, fill="#1E3A8A", line_spacing=8)
    draw_pill(draw, (1000, 965), "304M PARAMS | GLOBAL SEMANTIC CONTEXT", bg_color="#1D4ED8", font_size=16)

    # Fusion Neck: Learned Spatial Gate & UPerNet (x: 1510 .. 1940)
    draw_card(draw, [1510, 575, 1940, 1025], fill="#FFFBEB", outline="#F59E0B", outline_width=3)
    draw.text((1530, 597), "Spatial Gate & UPerNet", fill="#92400E", font=f_box_h)
    draw.text((1530, 630), "Dynamic Pixel-Wise Multi-Scale Fusion", fill="#D97706", font=f_box_sub)
    draw_text_lines(draw, (1530, 667), [
        "• Pixel-wise gate: f_fused = G*f_sem + (1-G)*f_conv",
        "• Gate G: 3x3 conv over joint features",
        "• Dynamically trusts ViT in open water, CNN on edges",
        "• Condition FiLM modulation with water type",
        "• Multi-scale progressive aggregation (4, 8, 16, 32)",
        "• Auxiliary boundary & objectness heads"
    ], f_box_p, fill="#78350F", line_spacing=8)
    draw_pill(draw, (1530, 965), "DYNAMIC PIXEL-WISE FUSION NECK", bg_color="#D97706", font_size=16)

    # -------------------------------------------------------------------------
    # RIGHT COLUMN: ENSEMBLE, CALIBRATION & DELIVERABLE 2 (x: 2040 .. 2470)
    # -------------------------------------------------------------------------
    draw_card(draw, [2040, 150, 2470, 1025], fill="#FEF2F2", outline="#EF4444", outline_width=3)
    draw.text((2060, 172), "Ensemble & Output 2", fill="#991B1B", font=f_box_h)
    draw.text((2060, 205), "5-Fold F3Upgrade & Calibration", fill="#DC2626", font=f_box_sub)
    draw_text_lines(draw, (2060, 242), [
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
    ], f_box_p, fill="#7F1D1D", line_spacing=7.5)
    draw_pill(draw, (2060, 915), "OFFICIAL HEAD 2: 5-CLASS MASKS", bg_color="#991B1B", font_size=16)
    draw_pill(draw, (2060, 965), "SUBMISSION #19359 (0.6410)", bg_color="#1E293B", font_size=16)

    # -------------------------------------------------------------------------
    # ARROWS & ROUTING (PRISTINE, HIGH-DPI)
    # -------------------------------------------------------------------------
    # 1. Input -> Stage 1 Physical Basis (Top Branch)
    draw_elbow_arrow(draw, [(350, 440), (400, 440), (400, 335), (450, 335)], color="#10B981", width=4, label="Raw RGB", label_idx=0)

    # 2. Input -> Stream B ConvNeXt (Bottom Branch)
    draw_elbow_arrow(draw, [(350, 710), (400, 710), (400, 800), (450, 800)], color="#8B5CF6", width=4, label="Raw RGB", label_idx=0)

    # 3. Stage 1 -> Stage 2 Refiner (Horizontal)
    draw_arrow(draw, (880, 335), (980, 335), color="#10B981", width=4, label="RSUIGM")

    # 4. Stage 2 -> Deliverable 1 Restored Output (Horizontal)
    draw_arrow(draw, (1410, 335), (1510, 335), color="#059669", width=4, label="Restored RGB")

    # 5. Stage 2 -> Stream A DINOv3 (Straight Vertical Down!)
    draw_arrow(draw, (1195, 520), (1195, 575), color="#3B82F6", width=4, label="Restored RGB", label_offset=(64, 0))

    # 6. Stream B ConvNeXt -> Spatial Gate & UPerNet (Clean Orthogonal underneath Stream A)
    draw_elbow_arrow(draw, [
        (665, 1025),
        (665, 1060),
        (1725, 1060),
        (1725, 1025)
    ], color="#8B5CF6", width=4, label="ConvNeXt Inductive Boundary Features", label_idx=1)

    # 7. Stream A DINOv3 -> Spatial Gate & UPerNet (Horizontal)
    draw_arrow(draw, (1410, 800), (1510, 800), color="#3B82F6", width=4, label="DINOv3")

    # 8. Spatial Gate Neck -> Ensemble & Deliverable 2 (Horizontal)
    draw_arrow(draw, (1940, 800), (2040, 800), color="#EF4444", width=4, label="Fused Logits")

    # Save full resolution PNG directly (no downsampling!)
    img.save("factsheet-template/images/method_overview.png", dpi=(360, 360))
    img.save("docs/track1_pipeline_diagram.png", dpi=(360, 360))
    print("[DIAGRAM] Generated ultra-crisp, native 360 DPI Track 1 flowchart!")


# =============================================================================
# TRACK 2 ARCHITECTURE FLOWCHART (JETSON AGX ORIN)
# =============================================================================
def generate_track2_pipeline():
    img = Image.new("RGB", (WIDTH, HEIGHT), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    f_title = get_font(36, bold=True)
    f_sub = get_font(21)
    f_box_h = get_font(25, bold=True)
    f_box_sub = get_font(18.5, bold=True)
    f_box_p = get_font(18)

    # Outer border / header
    draw.rounded_rectangle([20, 20, WIDTH - 20, HEIGHT - 20], radius=18, outline="#CBD5E1", width=3)
    draw.text((50, 42), "Team SAT (MIT-BLR) — Track 2: track2-apex-orin-v3 Embedded Edge Pipeline", fill="#042F2E", font=f_title)
    draw.text((50, 88), "Single-Graph Multi-Task Perception on NVIDIA Jetson AGX Orin | 0.884M Params (Cap: 1.0M) | 9.82 GFLOPs (Cap: 12.0G)", fill="#0F766E", font=f_sub)
    draw.line([(50, 122), (WIDTH - 50, 122)], fill="#E2E8F0", width=2)

    # Coordinates:
    # Col 1: 50 .. 350 (w: 300)
    # Gap: 350 .. 450 (100px)
    # Col 2: 450 .. 880 (w: 430)
    # Gap: 880 .. 980 (100px)
    # Col 3: 980 .. 1410 (w: 430)
    # Gap: 1410 .. 1510 (100px)
    # Col 4: 1510 .. 1940 (w: 430)
    # Gap: 1940 .. 2040 (100px)
    # Col 5: 2040 .. 2470 (w: 430)

    # -------------------------------------------------------------------------
    # COL 1: STATIC INPUT TENSOR (x: 50 .. 350, y: 310 .. 830)
    # -------------------------------------------------------------------------
    draw_card(draw, [50, 310, 350, 830], fill="#F0FDFA", outline="#0D9488", outline_width=3)
    draw.text((70, 332), "Static Input Tensor", fill="#042F2E", font=f_box_h)
    draw.text((70, 365), "NVIDIA Jetson AGX Orin Engine", fill="#0F766E", font=f_box_sub)
    draw_text_lines(draw, (70, 402), [
        "Fixed-Shape Engine Ingestion:",
        "• Shape: [1, 3, 1024, 1024]",
        "• Precision: Float32 [0.0, 1.0]",
        "• Static memory allocation eliminates",
        "  runtime TensorRT buffer fragmentation",
        "• Bound zero-copy GPU shared memory",
        "• Feeds radiometric router and dual stem",
        "• 100% Whitelist operator graph"
    ], f_box_p, fill="#134E4A", line_spacing=8)
    draw_pill(draw, (70, 725), "STATIC INPUT TENSOR", bg_color="#0F766E", font_size=16)
    draw_pill(draw, (70, 775), "ZERO DYNAMIC ALLOCS", bg_color="#042F2E", font_size=16)

    # -------------------------------------------------------------------------
    # TOP PATHWAY: RESTORATION ENGINE (y: 150 .. 520)
    # -------------------------------------------------------------------------
    # Box 1: 18-D Radiometric Router (x: 450 .. 880)
    draw_card(draw, [450, 150, 880, 520], fill="#FFFFFF", outline="#0D9488", outline_width=3)
    draw.text((470, 172), "1. 18-D Radiometric Router", fill="#042F2E", font=f_box_h)
    draw.text((470, 205), "Fast Matrix Contraction (<0.30 GFLOPs)", fill="#0D9488", font=f_box_sub)
    draw_text_lines(draw, (470, 242), [
        "• 18 translation-invariant global statistics",
        "  (channel means, std, min/max, Sobel edges)",
        "• Downsampled to 256x256 via stride-4 pool",
        "• 2-layer LayerNorm MLP with Hardswish",
        "• Dynamic routing over 20-scene buffer",
        "• Analytical 19-term polynomial basis",
        "• 95% compute saving over spatial 2D convs"
    ], f_box_p, fill="#134E4A", line_spacing=8)
    draw_pill(draw, (470, 465), "<0.30 GFLOPS | DYNAMIC ROUTER", bg_color="#0D9488", font_size=16)

    # Box 2: FiLM Spatial Refiner @ 256px (x: 980 .. 1410)
    draw_card(draw, [980, 150, 1410, 520], fill="#FFFFFF", outline="#059669", outline_width=3)
    draw.text((1000, 172), "2. FiLM Spatial Refiner", fill="#065F46", font=f_box_h)
    draw.text((1000, 205), "Low-Resolution Residual Polish (+6.15 dB)", fill="#059669", font=f_box_sub)
    draw_text_lines(draw, (1000, 242), [
        "• 3 depthwise-separable inverted bottlenecks",
        "• Operates exclusively on 256x256 grid",
        "• 32-D condition FiLM modulation with depth",
        "• Bounded offset: I_base + 0.25*tanh(Conv)",
        "• In-graph 3x3 Gaussian unsharp (gamma = 0.16)",
        "• Bilinear upsampling to 1024x1024 output",
        "• Lifts PSNR from 17.1 dB to 23.25 dB (+6.15 dB)"
    ], f_box_p, fill="#064E3B", line_spacing=8)
    draw_pill(draw, (1000, 465), "RESTORED RGB: 23.25 dB | 0.847 SSIM", bg_color="#059669", font_size=16)

    # Box 3: Output Deliverable 1: Edge Restored RGB (x: 1510 .. 1940)
    draw_card(draw, [1510, 150, 1940, 520], fill="#ECFDF5", outline="#059669", outline_width=3)
    draw.text((1530, 172), "Output 1: Edge Restored", fill="#065F46", font=f_box_h)
    draw.text((1530, 205), "TensorRT Native Output Head 1", fill="#047857", font=f_box_sub)
    draw_text_lines(draw, (1530, 242), [
        "• Output tensor: [1, 3, 1024, 1024] Float32 in [0, 1]",
        "• Recovers natural color balance and contrast",
        "• Edge-sharpened high-frequency textures",
        "• 100% compliant with Jetson Orin whitelist",
        "• Zero external post-processing tools",
        "• Evaluated officially: 23.25 dB PSNR, 0.847 SSIM",
        "• Re-ingested into downstream dual stem"
    ], f_box_p, fill="#047857", line_spacing=8)
    draw_pill(draw, (1530, 465), "OFFICIAL HEAD 1: RESTORED IMAGE", bg_color="#047857", font_size=16)

    # -------------------------------------------------------------------------
    # BOTTOM PATHWAY: MULTI-TASK SEGMENTATION (y: 575 .. 1025)
    # -------------------------------------------------------------------------
    # Box 4: Shared Dual Stem (x: 450 .. 880)
    draw_card(draw, [450, 575, 880, 1025], fill="#F0F9FF", outline="#0284C7", outline_width=3)
    draw.text((470, 597), "3. Shared Dual Stem", fill="#075985", font=f_box_h)
    draw.text((470, 630), "Parallel Ingestion (Raw RGB + Restored RGB)", fill="#0284C7", font=f_box_sub)
    draw_text_lines(draw, (470, 667), [
        "• Restored RGB (from Refiner) + Raw RGB (Input)",
        "• Dual 3x3 stride-2 convs with zero-init gamma",
        "• Preserves ImageNet pretrained weights at step 0",
        "• Zero representational drift during multi-task",
        "• Merges restored color with raw edge gradients",
        "• Feeds directly into truncated MobileNetV3"
    ], f_box_p, fill="#0C4A6E", line_spacing=8)
    draw_pill(draw, (470, 965), "ZERO INITIAL DRIFT DUAL STEM", bg_color="#0284C7", font_size=16)

    # Box 5: Truncated MobileNetV3 Backbone (x: 980 .. 1410)
    draw_card(draw, [980, 575, 1410, 1025], fill="#FFFFFF", outline="#0284C7", outline_width=3)
    draw.text((1000, 597), "4. Truncated MobileNetV3", fill="#075985", font=f_box_h)
    draw.text((1000, 630), "Param-Capped Edge Backbone (Cut @ Block 11)", fill="#0369A1", font=f_box_sub)
    draw_text_lines(draw, (1000, 667), [
        "• Truncated strictly after Block 11 (features[:12])",
        "• Eliminates Block 12 heavy 386k expansion",
        "• Backbone parameters: only 406.4k (Total: 884.1k)",
        "• Passes official 1.0M cap with 11.6% headroom",
        "• Multi-scale stage taps: S4 (24ch), S8, S16 (112ch)",
        "• Mobile inverted bottlenecks with depthwise 3x3/5x5"
    ], f_box_p, fill="#1E293B", line_spacing=8)
    draw_pill(draw, (1000, 965), "0.884M TOTAL PARAMS (11.6% BUFFER)", bg_color="#0369A1", font_size=16)

    # Box 6: ASPP, FPN & Speckle Suppressor (x: 1510 .. 1940)
    draw_card(draw, [1510, 575, 1940, 1025], fill="#FFFBEB", outline="#D97706", outline_width=3)
    draw.text((1530, 597), "5. ASPP, FPN & Filter", fill="#92400E", font=f_box_h)
    draw.text((1530, 630), "In-Graph Speckle Filter (TensorRT)", fill="#D97706", font=f_box_sub)
    draw_text_lines(draw, (1530, 667), [
        "• Dilated ASPP (1, 2, 4) + ScenePrior context",
        "• 4-scale FPN with 72-ch depthwise skips",
        "• Pointwise 5-class linear classifier @ 256x256",
        "• In-Graph Spatial Speckle Suppressor:",
        "  P = Softmax(Z_256, dim=1)",
        "  Support = AvgPool2d(P, 11, stride=1, pad=5)",
        "  Cleaned Z = Z - (1.2/0.04)*ReLU(0.04 - Support)",
        "• Zero CPU latency: 100% inside TensorRT FP16 engine"
    ], f_box_p, fill="#78350F", line_spacing=8)
    draw_pill(draw, (1530, 965), "0 EXTRA PARAMS | TENSORRT-NATIVE", bg_color="#D97706", font_size=16)

    # -------------------------------------------------------------------------
    # RIGHT COLUMN: DELIVERABLE 2 & JETSON BENCHMARK (x: 2040 .. 2470)
    # -------------------------------------------------------------------------
    draw_card(draw, [2040, 150, 2470, 1025], fill="#FEF2F2", outline="#DC2626", outline_width=3)
    draw.text((2060, 172), "Output 2: Segmentation", fill="#991B1B", font=f_box_h)
    draw.text((2060, 205), "Jetson AGX Orin Verified Benchmarks", fill="#DC2626", font=f_box_sub)
    draw_text_lines(draw, (2060, 242), [
        "Final Segmentation Output Head:",
        "• Output Logits: [1, 5, 1024, 1024] Float32",
        "• Bilinear upsampling from 256px cleaned logits",
        "• 5 discrete categorical classes:",
        "  {Background, Substrate, Fish,",
        "   Elasmobranch, Macro-Fauna}",
        "",
        "NVIDIA Jetson AGX Orin Edge Benchmarks:",
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
    ], f_box_p, fill="#7F1D1D", line_spacing=7.5)
    draw_pill(draw, (2060, 915), "OUTPUT HEAD 2: SEG_LOGITS", bg_color="#DC2626", font_size=16)
    draw_pill(draw, (2060, 965), "SUBMISSION #19447 (0.4070)", bg_color="#042F2E", font_size=16)

    # -------------------------------------------------------------------------
    # ARROWS & ROUTING (PRISTINE, HIGH-DPI)
    # -------------------------------------------------------------------------
    # 1. Input -> Router (Top Branch)
    draw_elbow_arrow(draw, [(350, 440), (400, 440), (400, 335), (450, 335)], color="#0D9488", width=4, label="Downsampled", label_idx=0)

    # 2. Input -> Dual Stem (Bottom Branch)
    draw_elbow_arrow(draw, [(350, 710), (400, 710), (400, 800), (450, 800)], color="#0284C7", width=4, label="Raw RGB", label_idx=0)

    # 3. Router -> Spatial Refiner (Horizontal)
    draw_arrow(draw, (880, 335), (980, 335), color="#0D9488", width=4, label="Basis Matrix")

    # 4. Spatial Refiner -> Deliverable 1 Restored RGB (Horizontal)
    draw_arrow(draw, (1410, 335), (1510, 335), color="#059669", width=4, label="Restored RGB")

    # 5. Spatial Refiner -> Shared Dual Stem (Clean Orthogonal Down-Left to Dual Stem!)
    draw_elbow_arrow(draw, [
        (1195, 520),
        (1195, 548),
        (665, 548),
        (665, 575)
    ], color="#059669", width=4, label="Restored RGB (In-Graph)", label_idx=1)

    # 6. Shared Dual Stem -> Truncated MobileNetV3 Backbone (Horizontal)
    draw_arrow(draw, (880, 800), (980, 800), color="#0284C7", width=4, label="Fused Stem")

    # 7. Truncated MobileNetV3 -> ASPP, FPN & Speckle Suppressor (Horizontal)
    draw_arrow(draw, (1410, 800), (1510, 800), color="#D97706", width=4, label="Skips")

    # 8. FPN & Speckle Suppressor -> Deliverable 2 Segmentation Output (Horizontal)
    draw_arrow(draw, (1940, 800), (2040, 800), color="#DC2626", width=4, label="5-ch Logits")

    # Save full resolution PNG directly (no downsampling!)
    img.save("factsheet-template/images/track2_pipeline.png", dpi=(360, 360))
    img.save("docs/track2_pipeline_diagram.png", dpi=(360, 360))
    print("[DIAGRAM] Generated ultra-crisp, native 360 DPI Track 2 flowchart!")


if __name__ == "__main__":
    generate_track1_pipeline()
    generate_track2_pipeline()
