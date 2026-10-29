"""
Publication-Grade Factsheet PDF Generator for EURS @ MaCVi WACV 2027
Humanized, mathematically precise, publication-quality layout.
Matches the official EURS factsheet template specifications.
"""

import os
import sys
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable, Image as RLImage
)
from reportlab.pdfgen import canvas


class FactsheetNumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        # Running Header (Pages > 1)
        if self._pageNumber > 1:
            self.setFont("Times-Italic", 8.5)
            self.setFillColor(colors.HexColor("#334155"))
            self.drawString(54, 745, "MaCVi@WACV 2027 EURS Challenge — Factsheet")
            self.drawRightString(558, 745, "Team SAT (MIT-BLR)")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 739, 558, 739)

        # Running Footer (All Pages)
        self.setFont("Times-Roman", 9)
        self.setFillColor(colors.HexColor("#475569"))
        self.drawRightString(306, 32, f"— {self._pageNumber} —")
        self.restoreState()


def build_factsheet_pdf(output_path: str):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    PRIMARY = colors.HexColor("#0F172A")
    ACCENT = colors.HexColor("#1E3A8A")
    TEXT = colors.HexColor("#1E293B")
    TEXT_MUTED = colors.HexColor("#475569")
    BORDER = colors.HexColor("#94A3B8")
    
    p_title = ParagraphStyle('FTitle', fontName='Times-Bold', fontSize=14.5, leading=18, textColor=PRIMARY, alignment=1, spaceAfter=3)
    p_sub = ParagraphStyle('FSub', fontName='Times-Bold', fontSize=11.5, leading=15, textColor=ACCENT, alignment=1, spaceAfter=5)
    p_author = ParagraphStyle('FAuthor', fontName='Times-Roman', fontSize=9.5, leading=13, textColor=TEXT, alignment=1, spaceAfter=2)
    p_affil = ParagraphStyle('FAffil', fontName='Times-Italic', fontSize=8.0, leading=10.8, textColor=TEXT_MUTED, alignment=1, spaceAfter=4)
    p_date = ParagraphStyle('FDate', fontName='Times-Roman', fontSize=8.2, leading=10.5, textColor=TEXT_MUTED, alignment=1, spaceAfter=8)
    
    p_h1 = ParagraphStyle('FH1', fontName='Times-Bold', fontSize=10.8, leading=14, textColor=PRIMARY, spaceBefore=7, spaceAfter=3, keepWithNext=True)
    p_h2 = ParagraphStyle('FH2', fontName='Times-Bold', fontSize=9.2, leading=12, textColor=ACCENT, spaceBefore=5, spaceAfter=2.5, keepWithNext=True)
    
    p_body = ParagraphStyle('FBody', fontName='Times-Roman', fontSize=7.9, leading=10.8, textColor=TEXT, spaceAfter=3.0)
    p_bullet = ParagraphStyle('FBullet', fontName='Times-Roman', fontSize=7.9, leading=10.8, textColor=TEXT, leftIndent=12, firstLineIndent=-8, spaceAfter=2.2)
    p_caption = ParagraphStyle('FCap', fontName='Times-Italic', fontSize=7.3, leading=9.8, textColor=TEXT_MUTED, alignment=1, spaceBefore=3, spaceAfter=4)
    p_code = ParagraphStyle('FCode', fontName='Courier', fontSize=7.0, leading=9.0, textColor=colors.HexColor("#0F172A"), leftIndent=10, spaceAfter=2.5)
    
    th_style = ParagraphStyle('FTH', fontName='Times-Bold', fontSize=7.2, leading=9.2, textColor=PRIMARY, alignment=1)
    td_style = ParagraphStyle('FTD', fontName='Times-Roman', fontSize=7.2, leading=9.2, textColor=TEXT)
    td_center = ParagraphStyle('FTDC', fontName='Times-Roman', fontSize=7.2, leading=9.2, textColor=TEXT, alignment=1)
    td_bold = ParagraphStyle('FTDB', fontName='Times-Bold', fontSize=7.2, leading=9.2, textColor=PRIMARY)
    td_bold_center = ParagraphStyle('FTDBC', fontName='Times-Bold', fontSize=7.2, leading=9.2, textColor=PRIMARY, alignment=1)

    story = []

    # =========================================================================
    # TITLE & HEADER
    # =========================================================================
    story.append(Paragraph("MaCVi@WACV 2027 EURS Challenge", p_title))
    story.append(Paragraph("Embedded Underwater Restoration and Segmentation — Official Factsheet", p_sub))
    story.append(Paragraph("<b>Arin Pattnaik, Tanmay Das, Shubham Kulkarni, Dr. Chaitra Desai</b>", p_author))
    story.append(Paragraph("Department of Computer Science & Engineering, Manipal Institute of Technology (MIT-BLR),<br/>Manipal Academy of Higher Education, Bengaluru, Karnataka, India", p_affil))
    story.append(Paragraph("Challenge Submission Date: October 28, 2026", p_date))
    story.append(HRFlowable(width="100%", thickness=1.0, color=BORDER, spaceBefore=0, spaceAfter=6))

    # =========================================================================
    # SECTION 1: TEAM DETAILS
    # =========================================================================
    story.append(Paragraph("1. Team Details", p_h1))
    
    t1_details = [
        "<b>Team name (as on macvi.org):</b> SAT (MIT-BLR)",
        "<b>Team leader name:</b> Arin Pattnaik (Faculty Advisor & Mentor: Dr. Chaitra Desai)",
        "<b>Team leader institution and email:</b> Department of Computer Science & Engineering, Manipal Institute of Technology (MIT-BLR), Manipal Academy of Higher Education, Bengaluru, Karnataka, India. Email: <code>arinp2018@gmail.com</code> (Faculty: <code>chaitra.desai@manipal.edu</code>)",
        "<b>Rest of the team members (name, macvi.org user ID):</b> Arin Pattnaik (Team Owner, User ID <code>2044</code>), Tanmay Das (User ID <code>2043</code>), Shubham Kulkarni (User ID <code>2035</code>), Dr. Chaitra Desai (Faculty Advisor & Mentor)",
        "<b>Team website URL:</b> <font color='#1E3A8A'><u>https://github.com/syntax-error23/SAT-EURS</u></font>",
        "<b>Affiliations:</b> Department of Computer Science & Engineering, Manipal Institute of Technology Bengaluru (MIT-BLR), Manipal Academy of Higher Education, Bengaluru, Karnataka, India",
        "<b>Team leader's user ID on the submission server (macvi.org):</b> <code>2044</code>",
        "<b>User IDs on the submission server of all team members (leader included; accounts that submitted):</b> <code>2044</code>, <code>2035</code>, <code>2043</code>",
        "<b>Track(s) entered:</b> Both (Track 1 and Track 2)",
        "<b>Best-scoring testing-phase submission per track (reported for reproduction):</b><br/>"
        "&bull; <b>Track 1 (Quality / Unconstrained):</b> Submission ID <b>#19359</b>, submitted by user ID <code>2035</code> (<code>shubham.kulkarni232007@gmail.com</code>), <b>Composite Score: 0.6410</b> (<i>Q</i><sub>seg</sub>: 0.5820, mIoU: 0.7110, Macro <i>F</i><sub>1</sub>: 0.8180, PSNR: 25.48 dB, SSIM: 0.8290)<br/>"
        "&bull; <b>Track 2 (Jetson AGX Orin Edge):</b> Submission ID <b>#19447</b>, submitted by user ID <code>2044</code> (<code>arinp2018@gmail.com</code>), <b>Composite Score: 0.4070</b> (<i>Q</i><sub>seg</sub>: 0.2270, mIoU: 0.4060, Macro <i>F</i><sub>1</sub>: 0.5600, PSNR: 23.25 dB, SSIM: 0.8470, Complexity: 9.82 GFLOPs, 0.884 M params)",
        "<b>Link to reproduction repository (one per track entered, official EURS template):</b><br/>"
        "&bull; Track 1 (Quality): <font color='#1E3A8A'><u>https://github.com/syntax-error23/SAT-EURS-Track1-Repro</u></font><br/>"
        "&bull; Track 2 (Jetson AGX Orin): <font color='#1E3A8A'><u>https://github.com/syntax-error23/SAT-EURS-Track2-Repro</u></font>",
        "<b>Artifact traceability:</b> Primary repository: <code>https://github.com/syntax-error23/SAT-EURS</code>. Track 1 final archive: <code>deploy/v33/submission_v33_ensemble_f3upgrade.zip</code> (SHA-256: <code>de741a84ce4601f82286ef156458b07e9fe2d4c4f75559a4589010f90ae6510d</code>). Track 2 ONNX binary: <code>onnx/submitted_model.onnx</code> (SHA-256: <code>addf76f170a0c09e0a3af50ebabb530b3f542080d2f3921c171ea3ae3cd80247</code>). Link access: Public GitHub repositories, permanent access."
    ]
    for d in t1_details:
        story.append(Paragraph(f"&bull; {d}", p_bullet))
    story.append(Spacer(1, 4))

    # =========================================================================
    # SECTION 2: GLOBAL METHOD DESCRIPTION
    # =========================================================================
    story.append(Paragraph("2. Global Method Description", p_h1))
    
    story.append(Paragraph(
        "Initial experiments with end-to-end multi-task training on the EURS benchmark indicated trade-offs between image restoration and semantic segmentation objectives. Jointly optimizing pixel-level reconstruction losses (L1 and Laplacian penalties) reduced segmentation mIoU along fine boundaries (e.g., seahorse tails and starfish appendages). Conversely, optimizing segmentation alone on uncorrected imagery required early network stages to compensate for strong wavelength attenuation rather than learning discriminative geometric features. Given the differing computational constraints of the competition tracks, we developed separate architectures for each setting:",
        p_body
    ))
    story.append(Paragraph(
        "<b>Track 1 (Unconstrained Quality):</b> We structure Track 1 as a two-stage pipeline separating image restoration from dense semantic segmentation. For restoration, we first regress an analytical 19-term polynomial basis (&phi;(<i>r,g,b</i>) &isin; &real;<sup>19</sup>) via closed-form Tikhonov regularisation (&lambda; = 10<sup>&minus;4</sup>), providing a physically grounded baseline that avoids high-frequency noise amplification. A neural refiner composed of 16 Non-Linear Activation Free (NAF) blocks with depthwise gated convolutions processes the concatenated 8-channel tensor [Raw RGB, RSUIGM RGB, Water Type, Depth]. To prevent the neural network from drifting away from physical optical priors or hallucinating artifacts on coral substrate, we clamped the refiner residual strictly to &plusmn;0.10 &times; tanh and sharpened high-frequency transitions using a Laplacian edge filter (&alpha;=1.34, &sigma;=1.5), reaching 25.48 dB PSNR and 0.829 SSIM on the test benchmark. For segmentation, vision transformers provide extensive receptive fields but consistently smear object boundaries by 10 to 15 pixels, whereas standard convolutional networks preserve sharp edges but miss global context in turbid waters. We resolved this trade-off using our asymmetric dual-stream <code>EURSAPEXV18Segmenter</code>: (1) DINOv3 ViT-Large/16 (304M params, patch 14) ingests restored RGB at 518&times;518 for global semantics, (2) ConvNeXt-Base (88.5M params) ingests raw degraded RGB at 512&times;512 through a 2-stage DetailStem to preserve fine anatomical boundaries, and (3) learned spatial gating (<i>f</i><sub>fused</sub> = &sigma;(<i>G</i>)&middot;<i>f</i><sub>sem</sub> + (1&minus;&sigma;(<i>G</i>))&middot;<i>f</i><sub>conv</sub>) dynamically blends features across a 4-level UPerNet neck paired with condition FiLM modulation.",
        p_body
    ))
    story.append(Paragraph(
        "<b>Track 2 (Jetson AGX Orin Edge):</b> To generate both restored imagery and 5-class segmentations within a single ONNX graph under the 1.0M parameter and 12.0 GFLOP limits, we implement a unified multi-task architecture: (1) an 18-D radiometric router computes translation-invariant global statistics from a downsampled 256&times;256 representation and contracts against a scale-consistent 20-scene calibration bank in &lt;0.30 GFLOPs (a 95% compute saving over 2D convs), (2) a 3-block depthwise-separable inverted bottleneck refiner operating at 256&times;256 with condition FiLM modulation lifts PSNR from 17.1 dB to 23.25 dB (+6.15 dB boost), (3) a shared dual stem feeds a MobileNetV3-Large trunk <b>truncated strictly after Block 11</b> (<code>features[:12]</code>), eliminating Block 12's heavy 386k parameter expansion and keeping total model parameters at 884.1k (passing the 1.0M ceiling with 11.6% headroom), (4) a dilated ASPP context module and 4-scale progressive FPN, and (5) an in-graph average pooling speckle suppressor (<code>AvgPool2d(11, 1, 5)</code>) that penalizes rare-class logits where regional support is below &tau;=0.04, executing natively on TensorRT with 0 extra parameters and 0.05 GFLOPs.",
        p_body
    ))

    method_disclosures_1 = [
        "<b>Which pre-trained or external methods / models have been used:</b> DINOv3 ViT-Large/16 pre-trained on LVD-142M via self-supervised learning; ConvNeXt-Base pre-trained on ImageNet-1k; MobileNetV3-Large pre-trained on ImageNet-1k (stages up to Block 11); NAFNet restoration blocks.",
        "<b>Which additional data has been used in addition to provided EURS training data:</b> <b>None</b>. Strictly trained exclusively on the provided EURS benchmark scenes (20 training scenes, 6 dev scenes). No external imagery, synthetic generators, or unannounced data were used.",
        "<b>Training description:</b> PyTorch 2.11 / 2.6 with CUDA 12.8, torchvision 0.26.0, timm 1.0.30. Optimizer: AdamW (&beta;<sub>1</sub>=0.9, &beta;<sub>2</sub>=0.999, weight decay 10<sup>&minus;4</sup>). LR schedule: Cosine annealing with 3 warmup epochs; peak lr 3&times;10<sup>&minus;5</sup> for Track 1, 2&times;10<sup>&minus;4</sup> for Track 2; min lr 10<sup>&minus;6</sup>. Batch size: Effective batch size 16 for Track 1 (batch size 4 with 4 gradient accumulation steps); batch size 8 for Track 2. Epochs: 25 per fold for Track 1; 30 for Track 2. Input size: 518&times;518 (ViT), 512&times;512 (ConvNeXt), 1024&times;1024 (Restoration and Track 2). Augmentations: Random flips, scale jitter [0.8, 1.25], random crop, color jitter (&plusmn;0.1), and copy-paste rare organism oversampling on elasmobranchs. Loss functions: Present-Class Lov&aacute;sz-Softmax (w=1.0), Class-Weighted Focal Cross-Entropy (&gamma;=2.0, &alpha;=[0.70, 0.90, 1.20, 3.00, 3.50], w=1.0), Soft Dice Loss (w=0.5), Auxiliary Boundary BCE (w=0.2), Restoration L1 + Laplacian Loss (w=0.1). Track 2 incorporates Multi-Teacher Ensemble Distillation (temperature &tau;=3.0, w=1.0). Hardware: Single NVIDIA GeForce RTX 5090 (32GB VRAM, Blackwell architecture). Training time: Track 1 &approx; 18h across all 5 folds; Track 2 &approx; 3.5h for 30 epochs.",
        "<b>Testing description:</b> Inputs evaluated at native resolution, resized such that longest side is exactly 1024 px with preserved aspect ratio matching official scoring. Track 1 uses 5-fold ensemble (Folds 0, 1, 2, 3-Upgrade, 4) with horizontal-flip TTA (10 forward passes total). Track 2 runs single-pass deterministic inference with zero TTA or ensemble directly in TensorRT / ONNX Runtime."
    ]
    for m in method_disclosures_1:
        story.append(Paragraph(f"&bull; {m}", p_bullet))
    story.append(Spacer(1, 4))

    # Figure 1: Track 1 Overview (Aspect ratio exactly 1200x525 -> 2.285: 504 / 2.285 = 220.5 pt)
    diag1 = "factsheet-template/images/method_overview.png"
    if os.path.exists(diag1):
        story.append(RLImage(diag1, width=504, height=220))
        story.append(Paragraph("<b>Figure 1:</b> Overview of the Track 1 perception architecture (<code>V33-Ensemble-F3Upgrade</code>, Submission #19359, 0.641 Composite). Degraded underwater imagery is first restored via a two-stage physical formulation combining an analytical 19-term RSUIGM radiative basis expansion and a bounded 16-block NAFNet residual refiner. The restored RGB feeds a DINOv3 ViT-Large dense transformer stream ($518\\times518$), while the raw degraded RGB concurrently feeds a ConvNeXt-Base convolutional hierarchy ($512\\times512$) and DetailStem. Features are fused via learned spatial gating and condition FiLM modulation in a UPerNet neck, averaged across a 5-fold ensemble with calibrated logit offsets.", p_caption))
    story.append(Spacer(1, 4))

    # Figure 2: Track 2 Overview
    diag2 = "factsheet-template/images/track2_pipeline.png"
    if os.path.exists(diag2):
        story.append(RLImage(diag2, width=504, height=220))
        story.append(Paragraph("<b>Figure 2:</b> Overview of the Track 2 embedded edge architecture (<code>track2-apex-orin-v3</code>, Submission #19447, 0.407 Composite). Designed for the NVIDIA Jetson AGX Orin under strict 1.0M parameter and 12.0 GFLOP limits: an 18-D radiometric summary router dynamically contracts a precomputed scale-consistent 20-scene polynomial buffer in &lt;0.30 GFLOPs, paired with a $256\\times256$ FiLM spatial refiner (+6.15 dB PSNR boost). Multi-task segmentation uses a truncated MobileNetV3 trunk (cut strictly after Block 11), dilated ASPP context module, 4-scale progressive FPN, ScenePriorGate, and an in-graph convolutional spatial speckle suppressor.", p_caption))
    story.append(Spacer(1, 4))

    method_disclosures_2 = [
        "<b>How training and development scenes were used:</b><br/>"
        "&bull; <i>Exact scene IDs used for training (20 scenes):</i> <code>fish1, fish2, fish4, fish5, fish7, fish9, fish11, fish12, fish13, fish14, fish15, fish17, fish18, fish20, fish21, ray1, seahorse1, sharkFish, starfish2, turtles1</code>.<br/>"
        "&bull; <i>Validation & model selection:</i> 5-fold cross-validation across these 20 training scenes, stratified by optical water type (Types 1–10) and depth level.<br/>"
        "&bull; <i>Development scenes:</i> The 6 development scenes (<code>fish3, fish6, fish10, fish19, sharRayFish, starfish1</code>) were held out strictly for out-of-domain validation, hyperparameter tuning, and post-submission discrepancy audits. Neither development nor testing scenes were ever used in gradient backpropagation. <b>Compliance with EURS rules: Yes.</b>",
        "<b>Were released testing-phase images used for training/adaptation?</b> <b>No.</b> Zero test pixels were used for training, fine-tuning, domain adaptation, self-supervision, or pseudo-labelling. All test predictions were generated strictly blind via feed-forward inference.",
        "<b>Leaderboard score guided choices:</b> Rare-class logit calibration offsets (&Delta;Z<sub>3</sub> = +0.10 for elasmobranchs, &Delta;Z<sub>4</sub> = +0.25 for macro-fauna) and Laplacian unsharp boost (&alpha;=1.34, &sigma;=1.5) were set as fixed constant scalars applied uniformly to every testing image. Selection of the Fold 3 retrained elasmobranch pilot checkpoint over baseline fold 3 was validated on the leaderboard test response (moving composite from 0.634 to 0.641).",
        "<b>Novelty degree of the solution:</b> We highlight four engineering decisions developed for the EURS benchmark constraints: (1) an 18-D radiometric summary router that contracts over precomputed optical basis tables in &lt;0.30 GFLOPs, substituting spatial 2D convolutions with global statistical routing for low-power edge restoration, (2) an asymmetric dual-stream design combining dense self-supervised representations from DINOv3 with boundary-preserving convolutional features from ConvNeXt via learned spatial gating, (3) an in-graph spatial speckle suppressor that runs entirely inside TensorRT FP16 via average pooling to eliminate false elasmobranch detections without host CPU intervention, and (4) a threshold calibration strategy that safeguards the evaluation pipeline against the severe Zero-True-Positive penalty cliff on rare marine classes. Developed exclusively for MaCVi@WACV 2027.",
        "<b>Other methods and baselines tested:</b> Generative Adversarial Networks (CycleGAN, pix2pixHD) produced textural hallucinations on coral substrates (SSIM < 0.65); Single-stream ViT-Large segmenters exhibited 10–15 pixel boundary dilation around fine structures; Full MobileNetV3-Large for Track 2 exceeded parameter limits (1.27M vs 1.0M cap) due to Block 12, resolved by truncating after Block 11."
    ]
    for m in method_disclosures_2:
        story.append(Paragraph(f"&bull; {m}", p_bullet))
    story.append(Spacer(1, 4))

    # =========================================================================
    # SECTION 3: TRACK 1: RESTORATION AND SEGMENTATION OUTPUTS
    # =========================================================================
    story.append(Paragraph("3. Track 1: Restoration and Segmentation Outputs", p_h1))
    
    t1_points = [
        "<b>Submission ID reported for final ranking:</b> <b>#19359</b> (Composite: <b>0.6410</b>, <i>Q</i><sub>seg</sub>: 0.5820, mIoU: 0.7110, Macro <i>F</i><sub>1</sub>: 0.8180, PSNR: 25.48 dB, SSIM: 0.8290)",
        "<b>Model checkpoints used:</b> 5-fold ensemble bundle in <code>deploy/v33/</code> (Fold 0: <code>checkpoints/v27/segmenter_v27_fold0_best.pt</code>, Fold 1: <code>checkpoints/v18/fold_1/segmenter_best.pt</code>, Fold 2: <code>checkpoints/v18/fold_2/segmenter_best.pt</code>, Fold 3 Upgrade: <code>checkpoints/overnight_pilots/fold_3/segmenter_v28_fold3_best.pt</code>, Fold 4: <code>checkpoints/v18/fold_4/segmenter_best.pt</code>; Refiner: <code>checkpoints/v18/restoration/refiner_best.pt</code>)",
        "<b>How outputs are produced:</b> Two-stage pipeline operating independently on each single image. The restoration engine computes RSUIGM polynomial features and runs the 16-block NAFNet refiner with Laplacian unsharp filtering to yield clean RGB output. The segmentation engine passes restored RGB through DINOv3 and raw RGB through ConvNeXt and DetailStem. Logits are averaged across all 5 folds and horizontal flips, calibrated via &Delta;<i>Z</i>, and mapped to uint8 masks via argmax.",
        "<b>Post-processing:</b> Static rare-class logit calibration offsets (&Delta;<i>Z</i> = [0.0, 0.0, 0.0, +0.10, +0.25]) added before argmax to eliminate false negatives on Class 3 (Elasmobranch) and Class 4 (Macro-Fauna). High-frequency Laplacian unsharp filtering on restored RGB. No morphological opening/closing or CRF applied.",
        "<b>Inference time per image:</b> &approx; 1030 ms per image on a single NVIDIA GeForce RTX 5090 at native test resolution (including 10 forward passes across 5 folds and horizontal flip TTA, plus full-resolution restoration)."
    ]
    for p in t1_points:
        story.append(Paragraph(f"&bull; {p}", p_bullet))
    story.append(Spacer(1, 3))

    # Table 1: Track 1 Specs
    t1_table_data = [
        [Paragraph("<b>Model</b>", th_style), Paragraph("<b>Input size</b>", th_style), Paragraph("<b>Epochs</b>", th_style), Paragraph("<b>Extra data</b>", th_style), Paragraph("<b>Ensemble / TTA</b>", th_style), Paragraph("<b># Params. (M)</b>", th_style), Paragraph("<b>GPU</b>", th_style)],
        [Paragraph("V33-Ensemble-F3Upgrade", td_bold), Paragraph("518&times;518 / 512&times;512", td_style), Paragraph("25 / fold", td_center), Paragraph("No", td_center), Paragraph("Yes (5 Folds + Flip TTA)", td_style), Paragraph("412.3 / 2,061.5", td_center), Paragraph("RTX 5090 (32GB)", td_style)]
    ]
    t1_t = Table(t1_table_data, colWidths=[120, 85, 45, 45, 95, 60, 54])
    t1_t.setStyle(TableStyle([
        ('LINEABOVE', (0, 0), (-1, 0), 1.0, PRIMARY),
        ('LINEBELOW', (0, 0), (-1, 0), 0.5, PRIMARY),
        ('LINEBELOW', (0, -1), (-1, -1), 1.0, PRIMARY),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t1_t)
    story.append(Paragraph("<b>Table 1:</b> Training statistics and computational metrics for Track 1 (# Params: 412.3M per single model, 2,061.5M total ensemble).", p_caption))
    story.append(Spacer(1, 4))

    # =========================================================================
    # SECTION 4: TRACK 2: ONNX MODEL ON THE JETSON AGX ORIN
    # =========================================================================
    story.append(Paragraph("4. Track 2: ONNX Model on the Jetson AGX Orin", p_h1))
    
    t2_points = [
        "<b>Submission ID reported for final ranking:</b> <b>#19447</b> (Composite: <b>0.4070</b>, <i>Q</i><sub>seg</sub>: 0.2270, mIoU: 0.4060, Macro <i>F</i><sub>1</sub>: 0.5600, PSNR: 23.25 dB, SSIM: 0.8470)",
        "<b>How ONNX file is produced:</b> Exported from PyTorch checkpoint using <code>torch.onnx.export</code> with <code>opset_version=17</code> and <code>do_constant_folding=True</code>. Native PyTorch <code>nn.Hardswish</code> and <code>nn.Hardsigmoid</code> were replaced with elementary arithmetic clipping nodes: <code>x * relu6(x + 3.0) / 6.0</code> and <code>relu6(x + 3.0) / 6.0</code>, guaranteeing 100% adherence to the official Jetson operator whitelist. Zero external graph post-processing or quantizers required.",
        "<b>Copy of submitted ONNX file:</b> Path: <code>onnx/submitted_model.onnx</code> (also mirrored at <code>deploy/track2_apex_orin_v3.onnx</code>). SHA-256: <code>addf76f170a0c09e0a3af50ebabb530b3f542080d2f3921c171ea3ae3cd80247</code> (6,790,009 bytes). This is the exact byte file submitted to the leaderboard.",
        "<b>Software used:</b> PyTorch 2.6.0, ONNX 1.31.0, opset 17.",
        "<b>Does <code>python repro_check.py . --export-onnx</code> regenerate submitted file?</b> <b>Yes (same graph and weights, header differs).</b> The regenerated model possesses identical topology, layers, and weights, with minor difference restricted to the PyTorch version string in the ONNX file header.",
        "<b>Static input shape:</b> 1&times;3&times;1024&times;1024; <b>outputs:</b> <code>restored_image</code> (1&times;3&times;1024&times;1024) and <code>seg_logits</code> (1&times;5&times;1024&times;1024).",
        "<b>Complexity of submitted model:</b> <b>9.82 GFLOPs</b> (official limit 12.0 GFLOPs, 18.2% headroom) and <b>0.884 M parameters</b> (official limit 1.0 M parameters, 11.6% headroom).",
        "<b>Efficiency measures reported on Jetson AGX Orin:</b> Latency <b>5.8 ms / frame (172.4 FPS)</b> on TensorRT FP16 engine; Latency <b>43.6 ms / frame (22.9 FPS)</b> on ARM Cortex-A78AE CPU; Energy &approx; 0.17 J / frame on 30W operational power mode.",
        "<b>Is model identical to Track 1?</b> <b>No.</b> Completely different architecture. Track 1 utilizes large ViT-Large and ConvNeXt-Base ensembles (412M params, &gt;150 GFLOPs). Track 2 utilizes a compact single-network architecture (0.884M params, 9.82 GFLOPs) specifically engineered for edge deployment.",
        "<b>Efficiency strategies:</b> (1) 18-D Radiometric Router contracting in &lt;0.30 GFLOPs, (2) Low-resolution FiLM refiner restricted to 256&times;256, (3) Truncated MobileNetV3 backbone cut after Block 11, saving 386k params, (4) In-Graph spatial speckle suppressor running natively on TensorRT with 0 extra params and 0.05 GFLOPs."
    ]
    for p in t2_points:
        story.append(Paragraph(f"&bull; {p}", p_bullet))
    story.append(Spacer(1, 3))

    # Table 2: Track 2 Specs
    t2_table_data = [
        [Paragraph("<b>Model</b>", th_style), Paragraph("<b>Input (H,W)</b>", th_style), Paragraph("<b>Precision</b>", th_style), Paragraph("<b># Params. (M)</b>", th_style), Paragraph("<b>GFLOPs</b>", th_style), Paragraph("<b>Training time</b>", th_style), Paragraph("<b>Epochs</b>", th_style), Paragraph("<b>Extra data</b>", th_style)],
        [Paragraph("track2-apex-orin-v3", td_bold), Paragraph("1024&times;1024", td_center), Paragraph("FP32 (ONNX) / FP16 (TRT)", td_style), Paragraph("0.884 M", td_center), Paragraph("9.82 G", td_center), Paragraph("3.5 h", td_center), Paragraph("30", td_center), Paragraph("No", td_center)]
    ]
    t2_t = Table(t2_table_data, colWidths=[95, 65, 95, 55, 45, 50, 45, 54])
    t2_t.setStyle(TableStyle([
        ('LINEABOVE', (0, 0), (-1, 0), 1.0, PRIMARY),
        ('LINEBELOW', (0, 0), (-1, 0), 0.5, PRIMARY),
        ('LINEBELOW', (0, -1), (-1, -1), 1.0, PRIMARY),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t2_t)
    story.append(Paragraph("<b>Table 2:</b> Training statistics and computational metrics for Track 2.", p_caption))
    story.append(Spacer(1, 4))

    # =========================================================================
    # SECTION 5: IMPLEMENTATION AND ENVIRONMENT
    # =========================================================================
    story.append(Paragraph("5. Implementation and Environment", p_h1))
    
    t5_points = [
        "<b>Programming language and framework:</b> Python 3.11, PyTorch 2.11 / 2.6.0, torchvision 0.26.0, timm 1.0.30, numpy 2.4.6.",
        "<b>Export and runtime versions:</b> ONNX 1.31.0, ONNX Runtime 1.31.0, TensorRT 8.6 / 10.0, opset 17.",
        "<b>Operating system and hardware used:</b> Training: Ubuntu 22.04 LTS / Windows 11, NVIDIA GeForce RTX 5090 (32GB VRAM, Blackwell architecture, TF32 and BF16 enabled), 64GB DDR5 host RAM, 24-core CPU. Edge Deployment Platform: NVIDIA Jetson AGX Orin 64GB Industrial Module running JetPack 6.0.",
        "<b>External code and dependencies:</b> PyTorch (BSD-3-Clause), torchvision (BSD-3-Clause), timm (Apache-2.0), ONNX Runtime (MIT). Pinned dependency list provided in repository <code>requirements.txt</code>.",
        "<b>Efficiency optimization strategies:</b> Static memory allocation preventing TensorRT memory fragmentation; replacing transcendental activations with piecewise linear clipping nodes (<code>relu6</code>); fused depthwise-separable convolutions; batch tensor contraction over precomputed radiative transfer tables."
    ]
    for p in t5_points:
        story.append(Paragraph(f"&bull; {p}", p_bullet))
    story.append(Spacer(1, 4))

    # =========================================================================
    # SECTION 6: REPRODUCTION PROCEDURE
    # =========================================================================
    story.append(Paragraph("6. Reproduction Procedure", p_h1))
    
    t6_points = [
        "<b>Installation commands:</b><br/>"
        "<code>git clone https://github.com/syntax-error23/SAT-EURS.git</code><br/>"
        "<code>cd SAT-EURS && conda create -n eurs_repro python=3.11 -y && conda activate eurs_repro</code><br/>"
        "<code>pip install -r requirements.txt</code>",
        "<b>Track 2: ONNX file regeneration command:</b><br/>"
        "<code>python deploy/export_track2_apex_orin.py --model-dir weights/ --out-onnx onnx/submitted_model.onnx</code><br/>"
        "<code>python repro_check.py . --export-onnx</code>",
        "<b>Inference commands:</b><br/>"
        "Track 1: <code>python tools/v33/run_v33_candidate.py --input-dir /path/to/test_images --output-dir /path/to/outputs --model-dir checkpoints/</code><br/>"
        "Track 2: <code>python deploy/run_onnx_inference.py --onnx onnx/submitted_model.onnx --input-dir /path/to/test_images --output-dir /path/to/outputs</code>",
        "<b>Checkpoint file names and SHA-256 hashes:</b><br/>"
        "&bull; Track 1 archive: <code>deploy/v33/submission_v33_ensemble_f3upgrade.zip</code> (SHA-256: <code>de741a84ce4601f82286ef156458b07e9fe2d4c4f75559a4589010f90ae6510d</code>)<br/>"
        "&bull; Track 2 ONNX: <code>onnx/submitted_model.onnx</code> (SHA-256: <code>addf76f170a0c09e0a3af50ebabb530b3f542080d2f3921c171ea3ae3cd80247</code>)<br/>"
        "&bull; Track 2 PyTorch master checkpoint: <code>checkpoints/track2_apex/track2_apex_v3_best.pt</code> (SHA-256: <code>c6e3b567d1d261e4795a2872ff7e4a7c1e576be88a24898144b6dc51cf216bbd</code>)",
        "<b>Expected input and output structure:</b> Input: folder of <code>&lt;scene&gt;_&lt;type&gt;_&lt;level&gt;.png</code> degraded images. Output: <code>restored/&lt;stem&gt;.png</code> (RGB, longest side 1024 px) and <code>seg_masks/&lt;stem&gt;.png</code> (uint8 mask in {0, 1, 2, 3, 4}, longest side 1024 px).",
        "<b>Fixed seeds and deterministic operations:</b> Deterministic execution enforced via <code>torch.manual_seed(42)</code>, <code>torch.cuda.manual_seed_all(42)</code>, <code>torch.backends.cudnn.deterministic = True</code>, and explicit <code>model.eval()</code> invocation. Zero test-time randomness.",
        "<b>Expected reproduced result:</b> Track 1: Composite &approx; <b>0.641</b> (<i>Q</i><sub>seg</sub> &approx; 0.582, mIoU &approx; 0.711, Macro <i>F</i><sub>1</sub> &approx; 0.818, PSNR &approx; 25.48 dB, SSIM &approx; 0.829). Track 2: Composite &approx; <b>0.407</b> (<i>Q</i><sub>seg</sub> &approx; 0.227, mIoU &approx; 0.406, Macro <i>F</i><sub>1</sub> &approx; 0.560, PSNR &approx; 23.25 dB, SSIM &approx; 0.847, Complexity &approx; 9.82 GFLOPs, 0.884 M params).",
        "<b>Approximate run time of full reproduction:</b> Track 1: &approx; 20 minutes for 1,200 images on RTX 5090. Track 2: &approx; 4 minutes on ONNX Runtime; under 15 seconds on TensorRT FP16 on Jetson AGX Orin."
    ]
    for p in t6_points:
        story.append(Paragraph(f"&bull; {p}", p_bullet))
    story.append(Spacer(1, 4))

    # =========================================================================
    # SECTION 7: OTHER DETAILS & DECLARATION
    # =========================================================================
    story.append(Paragraph("7. Other Details", p_h1))
    
    t7_points = [
        "<b>Planned submission of solution description paper to MaCVi@WACV 2027:</b> <b>Yes.</b> The team plans to submit a full technical paper detailing our findings to the workshop proceedings.",
        "<b>General comments and impressions of the EURS challenge:</b> The challenge benchmark provides a realistic evaluation setting for underwater robotics by assessing joint optical restoration and dense semantic segmentation under embedded compute constraints. The Coverage Penalty formulation (<i>Q</i><sub>seg</sub> &times; &prod;<sub>c</sub> <b>1</b>(TP<sub>c</sub> &gt; 0)) was an effective mechanism to deter models from ignoring ecologically critical rare organisms, though it created a steep sensitivity cliff where missing just one pixel of an elasmobranch across a test split causes a catastrophic score collapse. This necessitated threshold calibration rather than relying purely on maximum-likelihood posteriors. Track 2's operator whitelist and FLOP ceilings effectively mirrored real-world deployability constraints on the Jetson AGX Orin platform.",
        "<b>What do you expect from a new challenge in underwater image restoration and segmentation?</b> Future editions would benefit from: (1) multi-frame temporal video evaluation (incorporating AUV optical flow and temporal mask consistency), (2) active turbidity and backscatter sensor fusion (e.g., matching acoustic/sonar feeds or polarized lighting), and (3) physical energy metering during edge benchmarking (measuring peak milliwatts and thermal throttling during sustained dive profiles).",
        "<b>Known limitations and failure cases:</b> In extreme water types (Types 9 and 10) at maximum attenuation depths (&gt;25 m), optical light extinction in the red spectrum is near-total, leading occasionally to subtle contrast suppression in deep blue sea floor regions. Additionally, sub-5-pixel micro-appendages (such as seahorse tails) can occasionally experience boundary dilation under heavy backscatter haze."
    ]
    for p in t7_points:
        story.append(Paragraph(f"&bull; {p}", p_bullet))
    story.append(Spacer(1, 5))

    # Declaration Box
    story.append(Paragraph("<b>Declaration:</b>", p_h2))
    story.append(Paragraph(
        "I confirm that all external datasets, pretrained models, ensembles and post-processing used for the reported submission(s) are disclosed in this factsheet; that the submitted code and weights correspond to the named testing-phase submission(s); and that the method complies with the official EURS rules.",
        p_body
    ))
    story.append(Spacer(1, 6))
    
    decl_data = [
        [Paragraph("<b>Team Leader:</b> Arin Pattnaik (Faculty Advisor: Dr. Chaitra Desai)", p_body), Paragraph("<b>Date:</b> October 28, 2026", td_bold_center)]
    ]
    t_decl = Table(decl_data, colWidths=[350, 154])
    t_decl.setStyle(TableStyle([
        ('LINEABOVE', (0, 0), (-1, 0), 0.5, BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_decl)
    story.append(Spacer(1, 6))

    # References
    story.append(Paragraph("<b>References:</b>", p_h2))
    refs = [
        "[1] O. Ronneberger, P. Fischer, T. Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation. In <i>MICCAI</i>, 2015.",
        "[2] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, S. Xie. A ConvNet for the 2020s. In <i>CVPR</i>, 2022.",
        "[3] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, et al. DINOv2: Learning Robust Visual Features without Supervision. In <i>arXiv:2304.07193</i>, 2023.",
        "[4] A. Howard, M. Sandler, G. Chu, L.-C. Chen, B. Chen, M. Tan, et al. Searching for MobileNetV3. In <i>ICCV</i>, 2019.",
        "[5] L. Chen, X. Chu, X. Zhang, J. Sun. Simple Baselines for Image Restoration. In <i>ECCV</i>, 2022.",
        "[6] M. Berman, A. R. Triki, M. B. Blaschko. The Lov&aacute;sz-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks. In <i>CVPR</i>, 2018.",
        "[7] T.-Y. Lin, P. Goyal, R. Girshick, K. He, P. Doll&aacute;r. Focal Loss for Dense Object Detection. In <i>ICCV</i>, 2017.",
        "[8] E. Perez, F. Strub, H. de Vries, V. Chaplot, A. Courville. FiLM: Visual Reasoning with a General Conditioning Layer. In <i>AAAI</i>, 2018."
    ]
    for r in refs:
        story.append(Paragraph(r, p_body))

    doc.build(story, canvasmaker=FactsheetNumberedCanvas)
    print(f"[FACTSHEET PDF] Built publication-grade factsheet PDF: {output_path} ({os.path.getsize(output_path)/1024:.1f} KB)")


if __name__ == "__main__":
    for name in ["EURS_factsheet_SAT.pdf", "EURS_factsheet_SAT-EURS.pdf"]:
        build_factsheet_pdf(name)
        build_factsheet_pdf(os.path.join("factsheet-template", name))
        build_factsheet_pdf(os.path.join("reports", name))
        build_factsheet_pdf(os.path.join("docs", name))
