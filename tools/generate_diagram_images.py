"""
Publication-Grade Visual Flowchart Generator for SAT-EURS Track 1 & Track 2
Generates 4 high-resolution (300 DPI equivalent) anti-aliased architecture diagrams:
  1. docs/track1_pipeline_diagram.png
  2. docs/track1_subsystem_diagram.png
  3. docs/track2_pipeline_diagram.png
  4. docs/track2_subsystem_diagram.png
"""

import os
from PIL import Image, ImageDraw, ImageFont

# Canvas dimensions (2x super-sampled for razor-sharp downscaling)
WIDTH = 2160
HEIGHT_MAIN = 960
HEIGHT_SUB = 840

def get_font(size: int, bold: bool = False):
    font_name = "arialbd.ttf" if bold else "arial.ttf"
    try:
        return ImageFont.truetype(font_name, size)
    except Exception:
        return ImageFont.load_default()

def draw_shadowed_box(
    draw: ImageDraw.ImageDraw,
    xy,
    radius: int = 16,
    fill: str = "#FFFFFF",
    outline: str = "#CBD5E1",
    outline_width: int = 3,
    shadow_offset: int = 4,
    shadow_color: str = "#E2E8F0"
):
    x0, y0, x1, y1 = xy
    # Draw soft shadow
    draw.rounded_rectangle(
        [x0 + shadow_offset, y0 + shadow_offset, x1 + shadow_offset, y1 + shadow_offset],
        radius=radius,
        fill=shadow_color
    )
    # Draw main box
    draw.rounded_rectangle(
        [x0, y0, x1, y1],
        radius=radius,
        fill=fill,
        outline=outline,
        width=outline_width
    )

def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple,
    end: tuple,
    color: str = "#475569",
    width: int = 4,
    arrow_size: int = 12
):
    x0, y0 = start
    x1, y1 = end
    draw.line([start, end], fill=color, width=width)
    
    # Draw arrowhead based on orientation
    import math
    angle = math.atan2(y1 - y0, x1 - x0)
    angle1 = angle + math.pi * 0.82
    angle2 = angle - math.pi * 0.82
    p1 = (x1 + arrow_size * math.cos(angle1), y1 + arrow_size * math.sin(angle1))
    p2 = (x1 + arrow_size * math.cos(angle2), y1 + arrow_size * math.sin(angle2))
    draw.polygon([end, p1, p2], fill=color)

def draw_badge(draw: ImageDraw.ImageDraw, xy, text: str, bg_color: str = "#4338CA", text_color: str = "#FFFFFF", font_size: int = 18):
    font = get_font(font_size, bold=True)
    bbox = font.getbbox(text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x, y = xy
    pad_h = 10
    pad_v = 4
    box = [x, y, x + tw + pad_h * 2, y + th + pad_v * 2]
    draw.rounded_rectangle(box, radius=8, fill=bg_color)
    draw.text((x + pad_h, y + pad_v - bbox[1]), text, fill=text_color, font=font)
    return box[2]  # right edge


# =============================================================================
# TRACK 1 PIPELINE DIAGRAM
# =============================================================================
def generate_track1_pipeline():
    img = Image.new("RGB", (WIDTH, HEIGHT_MAIN), "#F8FAFC")
    draw = ImageDraw.Draw(img)

    f_title = get_font(34, bold=True)
    f_sub = get_font(20)
    f_box_title = get_font(24, bold=True)
    f_box_desc = get_font(18)
    f_tag = get_font(15, bold=True)

    # Header
    draw.text((60, 35), "SAT-EURS Track 1: V33-Ensemble-F3Upgrade Complete Perception Pipeline", fill="#1E1B4B", font=f_title)
    draw.text((60, 80), "Unconstrained High-Capacity Architecture: Dual-Stream DINOv3 + ConvNeXt with Physics-Guided NAFNet Refiner", fill="#4338CA", font=f_sub)
    draw.line([(60, 115), (WIDTH - 60, 115)], fill="#CBD5E1", width=2)

    # 1. Input Node (Left)
    draw_shadowed_box(draw, [60, 380, 310, 560], fill="#FFFFFF", outline="#6366F1", outline_width=4)
    draw.text((85, 400), "Degraded Input", fill="#1E1B4B", font=f_box_title)
    draw.text((85, 435), "Raw Optical Image\n[B, 3, H, W] in [0, 1]\nAbsorption + Scattering", fill="#475569", font=f_box_desc)
    draw_badge(draw, (85, 510), "RAW INPUT", bg_color="#4F46E5")

    # Flow arrows from input
    draw_arrow(draw, (310, 430), (410, 230), color="#4F46E5", width=4) # to restoration
    draw_arrow(draw, (310, 470), (410, 470), color="#0284C7", width=4) # to ConvNeXt
    draw_arrow(draw, (310, 510), (410, 710), color="#0D9488", width=4) # to Detail Stem

    # 2. Top Branch: Radiometric Restoration Engine
    draw_shadowed_box(draw, [410, 150, 820, 310], fill="#EEF2FF", outline="#4338CA", outline_width=3)
    draw.text((430, 165), "1. Physics Radiative Transfer", fill="#1E1B4B", font=f_box_title)
    draw.text((430, 200), "19-Term Non-Linear RSUIGM Basis\n+ 5-Fold NAFNet Neural Refiners (16 blks)\n+ Laplacian Unsharp Masking (gamma = 1.34)", fill="#312E81", font=f_box_desc)
    draw_badge(draw, (430, 270), "PSNR: 25.48 dB | SSIM: 0.829 | Sharp: 0.973", bg_color="#4338CA", font_size=15)

    # Arrow from Restoration to Restored Image Node
    draw_arrow(draw, (820, 230), (900, 230), color="#4338CA", width=4)

    # Restored Image Node
    draw_shadowed_box(draw, [900, 160, 1180, 300], fill="#FFFFFF", outline="#10B981", outline_width=4)
    draw.text((920, 175), "Restored Image", fill="#065F46", font=f_box_title)
    draw.text((920, 210), "Radiometrically Inverted RGB\n[B, 3, H, W] in [0, 1]\nFree of Chromatic Cast", fill="#047857", font=f_box_desc)
    draw_badge(draw, (920, 265), "STAGE 1 OUTPUT", bg_color="#059669")

    # Arrow from Restored to ViT Stream
    draw_arrow(draw, (1180, 230), (1260, 230), color="#059669", width=4)

    # 3. Main Semantic Stream: DINOv3 ViT-L/16
    draw_shadowed_box(draw, [1260, 150, 1680, 310], fill="#FFFFFF", outline="#4338CA", outline_width=3)
    draw.text((1280, 165), "2. DINOv3 ViT-L/16 Backbone", fill="#1E1B4B", font=f_box_title)
    draw.text((1280, 200), "24 Transformer Blocks (D = 1024, 16 Heads)\nDense feature taps from Blocks 5, 11, 17, 23\nHigh-Order Global Semantic Context", fill="#312E81", font=f_box_desc)
    draw_badge(draw, (1280, 270), "304.4M PARAMS | RESTORED STREAM", bg_color="#4338CA", font_size=15)

    # 4. Middle Branch: ConvNeXt-Base (Inductive Stream)
    draw_shadowed_box(draw, [410, 390, 820, 550], fill="#F0F9FF", outline="#0284C7", outline_width=3)
    draw.text((430, 405), "3. ConvNeXt-Base Stream", fill="#0369A1", font=f_box_title)
    draw.text((430, 440), "Official DINOv3 ConvNeXt-Base (88.5M params)\nStrides 4, 8, 16, 32 ([128, 256, 512, 1024]ch)\nEliminates 15px ViT Boundary Smearing", fill="#075985", font=f_box_desc)
    draw_badge(draw, (430, 510), "RAW DEGRADED STREAM", bg_color="#0284C7")

    # Arrow from ConvNeXt to Neck
    draw_arrow(draw, (820, 470), (1260, 470), color="#0284C7", width=4)

    # 5. Bottom Branch: Two-Stage Detail Stem & Condition Router
    draw_shadowed_box(draw, [410, 630, 820, 790], fill="#F0FDFA", outline="#0D9488", outline_width=3)
    draw.text((430, 645), "4. Raw Detail Stem & Cond-Enc", fill="#0F766E", font=f_box_title)
    draw.text((430, 680), "Two-Stage Conv Stem (Strides 2 & 4, 128ch)\nConditionEncoder: Water Type (11x64)\n+ Depth MLP (1->64) -> 64-D FiLM vector", fill="#115E59", font=f_box_desc)
    draw_badge(draw, (430, 750), "HIGH-RES BOUNDARIES", bg_color="#0D9488")

    # Arrow from Detail Stem to Neck
    draw_arrow(draw, (820, 710), (1260, 610), color="#0D9488", width=4)

    # 6. Central Fusion: UPerNet Neck with Learned Spatial Gating
    draw_shadowed_box(draw, [1260, 390, 1680, 710], fill="#FAF5FF", outline="#7E22CE", outline_width=4)
    draw.text((1280, 410), "5. FiLM-Gated UPerNet Neck", fill="#581C87", font=f_box_title)
    draw.text((1280, 450), "Learned Spatial Gate:\nG = sigmoid(Conv2d([f_sem, f_conv]))\nf_fused = G * f_sem + (1 - G) * f_conv\n\nBounded Decoder FiLM Gate:\n(1 + 0.20*tanh(gamma)) * f + 0.20*tanh(beta)\n\nMulti-Scale Feature Pyramid (256ch):\nP2 (S4), P3 (S8), P4 (S16), P5 (S32)", fill="#6B21A8", font=f_box_desc)
    draw_badge(draw, (1280, 660), "DYNAMIC MULTI-MODAL FUSION", bg_color="#7E22CE")

    # Connect ViT to Neck
    draw_arrow(draw, (1470, 310), (1470, 390), color="#4338CA", width=4)

    # Arrow from Neck to Ensemble Node
    draw_arrow(draw, (1680, 550), (1750, 550), color="#7E22CE", width=5)

    # 7. Final Stage: 5-Fold F3Upgrade Ensemble & Zero-TP Calibration
    draw_shadowed_box(draw, [1750, 320, 2100, 780], fill="#FFFFFF", outline="#B91C1C", outline_width=4)
    draw.text((1770, 345), "6. V33 F3Upgrade Ensemble", fill="#991B1B", font=f_box_title)
    draw.text((1770, 385), "5-Fold Scene-Disjoint Models:\nFold 0: v27 Reef Specialist\nFold 1: v18 Fauna Anchor\nFold 2: v18 Turbidity Specialist\nFold 3: v28-F3 Retrained Upgrade\nFold 4: v18 Deep Scatter Specialist\n\nCalibration & TTA:\n+ Calibrated Offsets [0, 0, 0, +0.10, +0.25]\n+ Horizontal-Flip TTA (10 passes/img)\n+ Single-Image Independence", fill="#7F1D1D", font=f_box_desc)
    draw_badge(draw, (1770, 690), "OFFICIAL WINNER #19359", bg_color="#B91C1C", font_size=16)
    draw_badge(draw, (1770, 730), "COMPOSITE: 0.641 | Qseg: 0.582", bg_color="#1E1B4B", font_size=16)

    # Resize with LANCZOS downsampling to smooth all edges
    out = img.resize((WIDTH // 2, HEIGHT_MAIN // 2), Image.Resampling.LANCZOS)
    os.makedirs("docs", exist_ok=True)
    out.save("docs/track1_pipeline_diagram.png")
    print("[DIAGRAM] Successfully generated docs/track1_pipeline_diagram.png")


# =============================================================================
# TRACK 1 SUBSYSTEM DIAGRAM: DUAL-STREAM GATING & RARE-CLASS DEFENSE
# =============================================================================
def generate_track1_subsystem():
    img = Image.new("RGB", (WIDTH, HEIGHT_SUB), "#F8FAFC")
    draw = ImageDraw.Draw(img)

    f_title = get_font(32, bold=True)
    f_sub = get_font(19)
    f_box_title = get_font(23, bold=True)
    f_box_desc = get_font(17)

    # Header
    draw.text((60, 30), "SAT-EURS Track 1: Subsystem Deep Dive & Mathematical Mechanics", fill="#1E1B4B", font=f_title)
    draw.text((60, 72), "Left: Learned Spatial Gating & FiLM Modulation | Right: Zero-TP Defense & Calibration Vector Dynamics", fill="#4338CA", font=f_sub)
    draw.line([(60, 105), (WIDTH - 60, 105)], fill="#CBD5E1", width=2)

    # Left Container: Learned Spatial Gate & FiLM Neck
    draw_shadowed_box(draw, [60, 140, 1040, 780], fill="#FFFFFF", outline="#4338CA", outline_width=3)
    draw.text((85, 160), "Subsystem A: Learned Spatial Gating & FiLM Dynamics", fill="#1E1B4B", font=f_box_title)
    
    # Internal component 1: Spatial Gate
    draw_shadowed_box(draw, [90, 210, 1010, 450], fill="#EEF2FF", outline="#6366F1", outline_width=2)
    draw.text((110, 225), "Learned Spatial Gate Architecture (Pixel-Wise Blending)", fill="#312E81", font=get_font(20, bold=True))
    draw.text((110, 260), "• Inputs: f_sem ∈ R^[B, 256, H/s, W/s] (from ViT) & f_conv ∈ R^[B, 256, H/s, W/s] (from ConvNeXt)\n• Concatenation: Cat[f_sem, f_conv] ∈ R^[B, 512, H/s, W/s]\n• Gate CNN: Conv2d(512, 128, 3, p=1) -> BN -> ReLU -> Conv2d(128, 256, 1) -> Sigmoid\n• Gate Output: G ∈ [0, 1]^[B, 256, H/s, W/s]\n• Fused Features: f_fused = G ⊙ f_sem + (1 - G) ⊙ f_conv\n• Zero-Init Bias: Starts at exact 50/50 balance at Step 0, dynamically adapts to image content.", fill="#1E1B4B", font=f_box_desc)

    # Internal component 2: FiLM Neck
    draw_shadowed_box(draw, [90, 480, 1010, 740], fill="#F5F3FF", outline="#7C3AED", outline_width=2)
    draw.text((110, 495), "Bounded Decoder FiLM Gate (Optical Invariance)", fill="#581C87", font=get_font(20, bold=True))
    draw.text((110, 530), "• Inputs: f_fused ∈ R^[B, 256, H/s, W/s] & Condition Vector e_cond ∈ R^[B, 64]\n• FiLM Generator: Linear(64, 512) -> splits into raw gamma, beta ∈ R^[B, 256]\n• Bounded Mapping: gamma = 0.20 * tanh(raw_gamma), beta = 0.20 * tanh(raw_beta)\n• Feature Modulation: f_mod = (1.0 + gamma) ⊙ f_fused + beta\n• Mathematical Advantage: Prevents condition shift from destabilizing semantic representation\n  while adapting classification thresholds across light attenuation depths (5m to 100m).", fill="#3B0764", font=f_box_desc)

    # Right Container: Zero-TP Defense & Calibration Vector
    draw_shadowed_box(draw, [1120, 140, 2100, 780], fill="#FFFFFF", outline="#059669", outline_width=3)
    draw.text((1145, 160), "Subsystem B: Rare-Class Calibration & Zero-TP Penalty Gate", fill="#065F46", font=f_box_title)

    # Internal component 3: Formula & Failure Gate
    draw_shadowed_box(draw, [1150, 210, 2070, 450], fill="#ECFDF5", outline="#10B981", outline_width=2)
    draw.text((1170, 225), "The Strict Zero-TP Catastrophic Failure Mode", fill="#047857", font=get_font(20, bold=True))
    draw.text((1170, 260), "• Competition Rule: Q_seg = mIoU * Macro_F1 * Coverage_Penalty\n• Coverage Penalty = ∏_c 1(TP_c > 0)\n• Critical Vulnerability: If any class present in test ground-truth achieves TP = 0,\n  Coverage Penalty collapses to 0.0, wiping out 60% of the composite score.\n• Rare marine classes (Class 3: Elasmobranch <1.5%, Class 4: Macro-Fauna <2.2%)\n  are easily suppressed by default argmax in murky, scattering water conditions.", fill="#064E3B", font=f_box_desc)

    # Internal component 4: Solution
    draw_shadowed_box(draw, [1150, 480, 2070, 740], fill="#FEF3C7", outline="#F59E0B", outline_width=2)
    draw.text((1170, 495), "Empirically Calibrated Logit Bias Vector [0.0, 0.0, 0.0, +0.10, +0.25]", fill="#92400E", font=get_font(20, bold=True))
    draw.text((1170, 530), "• Injected statically into raw logits prior to argmax: Z_calib = Z_raw + ΔZ\n• Background (C0): +0.00 | Coral/Substrate (C1): +0.00 | Fish (C2): +0.00\n• Elasmobranch (C3): +0.10 -> Recovers faint stingray/shark boundaries in deep water\n• Macro-Fauna (C4): +0.25 -> Captures cryptic camouflage starfish, seahorses, turtles\n• Empirical Outcome: Achieved 100% True-Positive Class Coverage across all 1,200 test\n  images, elevating mIoU to 0.711 and Macro F1 to 0.818.", fill="#78350F", font=f_box_desc)

    out = img.resize((WIDTH // 2, HEIGHT_SUB // 2), Image.Resampling.LANCZOS)
    out.save("docs/track1_subsystem_diagram.png")
    print("[DIAGRAM] Successfully generated docs/track1_subsystem_diagram.png")


# =============================================================================
# TRACK 2 PIPELINE DIAGRAM (JETSON AGX ORIN)
# =============================================================================
def generate_track2_pipeline():
    img = Image.new("RGB", (WIDTH, HEIGHT_MAIN), "#F8FAFC")
    draw = ImageDraw.Draw(img)

    f_title = get_font(34, bold=True)
    f_sub = get_font(20)
    f_box_title = get_font(24, bold=True)
    f_box_desc = get_font(18)

    # Header
    draw.text((60, 35), "SAT-EURS Track 2: track2-apex-orin-v3 Complete Embedded Pipeline", fill="#042F2E", font=f_title)
    draw.text((60, 80), "Embedded Jetson AGX Orin Edge SOTA: 0.884M Parameters | 9.82 GFLOPs | 100% ONNX Whitelist Compliant", fill="#0D9488", font=f_sub)
    draw.line([(60, 115), (WIDTH - 60, 115)], fill="#CBD5E1", width=2)

    # 1. Input Node (Left)
    draw_shadowed_box(draw, [60, 380, 310, 560], fill="#FFFFFF", outline="#0D9488", outline_width=4)
    draw.text((85, 400), "Static Input", fill="#042F2E", font=f_box_title)
    draw.text((85, 435), "Fixed Static Tensor\n[1, 3, 1024, 1024] float32\nZero Dynamic Reallocations", fill="#475569", font=f_box_desc)
    draw_badge(draw, (85, 510), "STATIC EDGE INPUT", bg_color="#0D9488")

    # Flow arrows from input
    draw_arrow(draw, (310, 430), (410, 230), color="#0D9488", width=4) # to restoration
    draw_arrow(draw, (310, 480), (410, 480), color="#0284C7", width=4) # to dual-stem
    draw_arrow(draw, (310, 520), (410, 710), color="#7C3AED", width=4) # to radiometric router

    # 2. Top Branch: 18-D Radiometric Router & Physical Contraction
    draw_shadowed_box(draw, [410, 140, 860, 320], fill="#F0FDFA", outline="#0D9488", outline_width=3)
    draw.text((430, 155), "1. 18-D Radiometric Router", fill="#042F2E", font=f_box_title)
    draw.text((430, 190), "Global Summary Vector s(x) ∈ R^18 @ 256x256\n+ 2-Layer LayerNorm MLP with OnnxSafeHardswish\nRoutes over Static Condition Buffer [200, 57]\nBatch Matrix Contraction in <0.30 GFLOPs", fill="#115E59", font=f_box_desc)
    draw_badge(draw, (430, 275), "95% RESTORATION COMPUTE SAVED", bg_color="#0D9488", font_size=15)

    # Arrow from Router to Refiner
    draw_arrow(draw, (860, 230), (940, 230), color="#0D9488", width=4)

    # 3. Spatial Residual Refiner & Unsharp Filtering
    draw_shadowed_box(draw, [940, 140, 1370, 320], fill="#FFFFFF", outline="#10B981", outline_width=3)
    draw.text((960, 155), "2. FiLM Spatial Refiner", fill="#065F46", font=f_box_title)
    draw.text((960, 190), "3 DW-Sep Blocks @ 256x256 + 32-D FiLM\nZero-Init Bounded Offset: 0.25 * tanh(Conv)\n+ In-Graph 3x3 Gaussian Unsharp (gamma=0.16)\nPSNR: 23.25 dB (+6.15 dB) | SSIM: 0.847", fill="#047857", font=f_box_desc)
    draw_badge(draw, (960, 275), "RESTORED RGB: 23.25 dB", bg_color="#059669", font_size=15)

    # Arrow from Restored to Restored Image Output & to Dual Stem
    draw_arrow(draw, (1370, 230), (1450, 230), color="#059669", width=4) # to restored output
    draw_arrow(draw, (1155, 320), (1155, 380), color="#059669", width=4) # down to dual stem

    # Restored Image Output Box
    draw_shadowed_box(draw, [1450, 160, 1720, 300], fill="#ECFDF5", outline="#059669", outline_width=3)
    draw.text((1470, 175), "Restored Output", fill="#065F46", font=f_box_title)
    draw.text((1470, 210), "[1, 3, 1024, 1024]\nZero Non-Whitelisted Ops\nEdge-Sharpened RGB", fill="#047857", font=f_box_desc)
    draw_badge(draw, (1470, 265), "OUTPUT HEAD 1", bg_color="#059669")

    # 4. Middle Branch: Truncated MobileNetV3-Large Encoder
    draw_shadowed_box(draw, [410, 390, 860, 560], fill="#F0F9FF", outline="#0284C7", outline_width=3)
    draw.text((430, 405), "3. Truncated MobileNetV3", fill="#0369A1", font=f_box_title)
    draw.text((430, 440), "Cut strictly after Block 11 (features[:12])\nSaves 386k parameters while retaining rich\nmulti-scale features: S4 (24ch), S8 (40ch), S16 (112ch)\nPretrained ImageNet Initialization", fill="#075985", font=f_box_desc)
    draw_badge(draw, (430, 520), "406.4k PARAMS | STRIDES 4, 8, 16", bg_color="#0284C7", font_size=15)

    # Dual-Input Stem Box
    draw_shadowed_box(draw, [940, 390, 1370, 560], fill="#FFFFFF", outline="#0284C7", outline_width=3)
    draw.text((960, 405), "4. Dual-Input Stem & Fusion", fill="#0369A1", font=f_box_title)
    draw.text((960, 440), "Dual Stem: Restored RGB + Raw Degraded RGB\nAux Degraded Conv (3x3, s=2, 16ch)\nZero-Init BatchNorm Gamma preserves ImageNet\nweights at Step 0 with zero representational drift", fill="#075985", font=f_box_desc)
    draw_badge(draw, (960, 520), "ZERO INITIAL DRIFT", bg_color="#0284C7")

    draw_arrow(draw, (860, 475), (940, 475), color="#0284C7", width=4)
    draw_arrow(draw, (1370, 475), (1450, 475), color="#0284C7", width=4)

    # 5. Bottom Branch: Stride-32 ASPP & Bounded ScenePriorGate
    draw_shadowed_box(draw, [410, 630, 860, 800], fill="#FAF5FF", outline="#7C3AED", outline_width=3)
    draw.text((430, 645), "5. Dilated ASPP & ScenePriorGate", fill="#581C87", font=f_box_title)
    draw.text((430, 680), "Stride-32 Context Stage (112 -> 128ch)\nDilated branches: d=1, 2, 4 + Global Pooling\nBounded ScenePriorGate:\nΔz = 0.35 * tanh(MLP([g32, e_cond]))", fill="#6B21A8", font=f_box_desc)
    draw_badge(draw, (430, 760), "WIDE CONTEXT RECEPTIVE FIELD", bg_color="#7C3AED", font_size=15)

    # Arrow from ASPP to FPN Decoder
    draw_arrow(draw, (860, 715), (1450, 620), color="#7C3AED", width=4)

    # 6. Progressive FPN & Spatial Speckle Suppressor
    draw_shadowed_box(draw, [1450, 390, 1850, 710], fill="#FFFBEB", outline="#D97706", outline_width=4)
    draw.text((1470, 410), "6. Progressive FPN & Speckle Filter", fill="#92400E", font=f_box_title)
    draw.text((1470, 450), "72-Channel Lateral Skips & DW-Sep Fusions\nS32 -> S16 -> S8 -> S4 (256x256)\nPointwise Classifier Head (5ch)\n\nIn-Graph Spatial Speckle Suppressor:\nSupport = AvgPool2d(Probs, 11, s=1, p=5)\nPenalty = (1.2 / 0.04) * ReLU(0.04 - Support)\nEliminates isolated false-positive speckles\non rare classes without postprocessing latency.", fill="#78350F", font=f_box_desc)
    draw_badge(draw, (1470, 665), "0 EXTRA PARAMS | 0.05 GFLOPs", bg_color="#D97706")

    # Arrow from FPN to Output
    draw_arrow(draw, (1850, 550), (1910, 550), color="#D97706", width=5)

    # 7. Final Output Head 2 (Segmentation)
    draw_shadowed_box(draw, [1910, 380, 2120, 710], fill="#FFFFFF", outline="#042F2E", outline_width=4)
    draw.text((1925, 405), "Seg Logits", fill="#042F2E", font=f_box_title)
    draw.text((1925, 445), "[1, 5, 1024, 1024]\nBilinear Upsample\n5-Class Discrete\nMask Prediction\n\nOfficial Metric:\nComposite: 0.407\nmIoU: 0.406\nMacro F1: 0.560\nQseg: 0.227", fill="#1E293B", font=f_box_desc)
    draw_badge(draw, (1925, 630), "OUTPUT HEAD 2", bg_color="#042F2E")
    draw_badge(draw, (1925, 665), "WINNER #19447", bg_color="#B91C1C", font_size=14)

    out = img.resize((WIDTH // 2, HEIGHT_MAIN // 2), Image.Resampling.LANCZOS)
    out.save("docs/track2_pipeline_diagram.png")
    print("[DIAGRAM] Successfully generated docs/track2_pipeline_diagram.png")


# =============================================================================
# TRACK 2 SUBSYSTEM DIAGRAM: RADIOMETRIC ROUTER & IN-GRAPH SPECKLE SUPPRESSION
# =============================================================================
def generate_track2_subsystem():
    img = Image.new("RGB", (WIDTH, HEIGHT_SUB), "#F8FAFC")
    draw = ImageDraw.Draw(img)

    f_title = get_font(32, bold=True)
    f_sub = get_font(19)
    f_box_title = get_font(23, bold=True)
    f_box_desc = get_font(17)

    # Header
    draw.text((60, 30), "SAT-EURS Track 2: Subsystem Deep Dive & Mathematical Mechanics", fill="#042F2E", font=f_title)
    draw.text((60, 72), "Left: 18-D Radiometric Router & Physical Matrix Contraction | Right: In-Graph Spatial Speckle Suppressor", fill="#0D9488", font=f_sub)
    draw.line([(60, 105), (WIDTH - 60, 105)], fill="#CBD5E1", width=2)

    # Left Container: 18-D Router
    draw_shadowed_box(draw, [60, 140, 1040, 780], fill="#FFFFFF", outline="#0D9488", outline_width=3)
    draw.text((85, 160), "Subsystem A: 18-D Radiometric Router & Matrix Bank", fill="#042F2E", font=f_box_title)

    # Internal component 1: 18-D Features
    draw_shadowed_box(draw, [90, 210, 1010, 450], fill="#F0FDFA", outline="#14B8A6", outline_width=2)
    draw.text((110, 225), "18-D Geometry-Invariant Global Radiometric Summary Vector s(x)", fill="#0F766E", font=get_font(20, bold=True))
    draw.text((110, 260), "Computed on 256x256 pooled representation with zero spatial conv overhead:\n1. 3 Channel Means: [μ_R, μ_G, μ_B] (global illumination levels)\n2. 3 Channel Standard Deviations: [σ_R, σ_G, σ_B] (turbidity / scattering spread)\n3. 3 Soft-Mins & 3 Soft-Maxs on 32x32 average-pooled spatial patches\n4. 3 Cross-Channel Attenuation Differences: [μ_B - μ_R, μ_G - μ_R, μ_B - μ_G]\n5. 3 Spatial Gradient Means: Horizontal and vertical Sobel differences\n• Invariant to camera rotation, object translation, and zoom levels.", fill="#134E4A", font=f_box_desc)

    # Internal component 2: Contraction
    draw_shadowed_box(draw, [90, 480, 1010, 740], fill="#ECFDF5", outline="#10B981", outline_width=2)
    draw.text((110, 495), "Real-Time Matrix Contraction (<0.30 GFLOPs)", fill="#065F46", font=get_font(20, bold=True))
    draw.text((110, 530), "• Condition Trunk: Linear(18, 64) -> LayerNorm -> Hardswish -> Linear(64, 64) -> LayerNorm\n• Predicts 32-D Condition Embedding e_cond + 200 Condition Logits\n• Routes over static buffer [200, 57] (tbl19_20sc_scale_consistent.pt, 10 water types x 20 depths)\n• Batch Matrix Multiplication: J_poly = MatMul(W_eff [19, 3], PolyBasis [19, 1024, 1024])\n• Compute Footprint: 0.28 GFLOPs (vs. 5.8 GFLOPs on 2D conv restorer)\n• Official PSNR surged from 17.10 dB to 23.25 dB (+6.15 dB restoration leap).", fill="#064E3B", font=f_box_desc)

    # Right Container: In-Graph Speckle Suppressor
    draw_shadowed_box(draw, [1120, 140, 2100, 780], fill="#FFFFFF", outline="#D97706", outline_width=3)
    draw.text((1145, 160), "Subsystem B: In-Graph Spatial Speckle Suppressor", fill="#92400E", font=f_box_title)

    # Internal component 3: The Speckle Problem
    draw_shadowed_box(draw, [1150, 210, 2070, 450], fill="#FFFBEB", outline="#F59E0B", outline_width=2)
    draw.text((1170, 225), "The Edge Elasmobranch False-Alarm Dilemma", fill="#B45309", font=get_font(20, bold=True))
    draw.text((1170, 260), "• Severe Failure Mode in Baseline #18699: In deep murky water (Water Type 10, Depth 75m),\n  dark crevices in coral reefs triggered high false-positive activations for elasmobranchs.\n• Why standard postprocessing fails: Connected-component labeling or morphological\n  opening scripts cannot be exported to ONNX or accelerated by Jetson TensorRT.\n• The entire suppression mechanism MUST reside natively inside the ONNX graph.", fill="#78350F", font=f_box_desc)

    # Internal component 4: The Convolutional Solution
    draw_shadowed_box(draw, [1150, 480, 2070, 740], fill="#FDF2F8", outline="#EC4899", outline_width=2)
    draw.text((1170, 495), "Embedded Convolutional Density Penalty Algorithm", fill="#9D174D", font=get_font(20, bold=True))
    draw.text((1170, 530), "1. Normalized Probabilities: P = Softmax(Z_256, dim=1) ∈ R^[1, 5, 256, 256]\n2. Regional Density: Support = AvgPool2d(P, kernel_size=11, stride=1, padding=5)\n3. Continuous Penalty Calculation:\n   Penalty = (1.2 / 0.04) * ReLU(0.04 - Support) * Mask_{Class 3 & 4}\n4. Cleaned Logits: Z_clean = Z_256 - Penalty\n• Real marine organisms have cluster support > 0.20 and suffer ZERO penalty.\n• Isolated single-pixel reef false alarms have support < 0.02 and receive a -1.2 logit suppression.", fill="#831843", font=f_box_desc)

    out = img.resize((WIDTH // 2, HEIGHT_SUB // 2), Image.Resampling.LANCZOS)
    out.save("docs/track2_subsystem_diagram.png")
    print("[DIAGRAM] Successfully generated docs/track2_subsystem_diagram.png")


if __name__ == "__main__":
    generate_track1_pipeline()
    generate_track1_subsystem()
    generate_track2_pipeline()
    generate_track2_subsystem()
