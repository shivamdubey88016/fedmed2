"""
Builds a high-resolution, professional PDF report for the FedEquiMed PhD project using Pillow.
Embeds user-provided figures with clear academic typography and structured technical analysis.
"""

import os
from PIL import Image, ImageDraw, ImageFont

def build_pdf():
    media_dir = "/Users/shivamdubey/.gemini/antigravity/brain/51314036-7c0e-4b7d-b92b-64da8a8949be/.user_uploaded"
    img1_path = os.path.join(media_dir, "media_1792648579935_17991baa.png")
    img2_path = os.path.join(media_dir, "media_1793110900514_c080662e.png")
    img3_path = os.path.join(media_dir, "media_1793111035870_2f1b3226.png")
    
    # Page dimensions: A4 at 150 DPI (1240 x 1754 px)
    W, H = 1240, 1754
    
    # Font configuration
    font_path = "/System/Library/Fonts/Helvetica.ttc"
    font_bold = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    if not os.path.exists(font_bold):
        font_bold = font_path
    
    title_font = ImageFont.truetype(font_bold, 28)
    subtitle_font = ImageFont.truetype(font_path, 16)
    section_font = ImageFont.truetype(font_bold, 20)
    body_bold = ImageFont.truetype(font_bold, 14)
    body_font = ImageFont.truetype(font_path, 13)
    caption_font = ImageFont.truetype(font_path, 12)
    meta_font = ImageFont.truetype(font_bold, 12)
    mono_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Courier New.ttf", 12) if os.path.exists("/System/Library/Fonts/Supplemental/Courier New.ttf") else body_font

    pages = []

    # =========================================================================
    # PAGE 1: Convergence Trajectories & Utility Preservation
    # =========================================================================
    page1 = Image.new("RGB", (W, H), "#FFFFFF")
    draw1 = ImageDraw.Draw(page1)

    # Top Banner
    draw1.rectangle([(0, 0), (W, 110)], fill="#1E3A8A")
    draw1.text((50, 22), "FedEquiMed: Fairness-Aware Federated Learning in Healthcare", fill="#FFFFFF", font=title_font)
    draw1.text((50, 68), "PhD Research Technical Report & Experimental Evaluation — Multi-Center MIMIC-IV Clinical Cohorts", fill="#93C5FD", font=subtitle_font)

    # Metadata bar
    draw1.rectangle([(0, 110), (W, 148)], fill="#F1F5F9")
    draw1.text((50, 122), "Author: Shivam Dubey  |  Framework: Flower (flwr) & PyTorch  |  Task: 30-Day ICU Mortality Prediction  |  Zero-PHI", fill="#475569", font=meta_font)

    # Section 1 Title
    draw1.text((50, 175), "1. Multi-Center Convergence & Demographic Disparity Trajectories", fill="#1E3A8A", font=section_font)
    draw1.text((50, 205), "Head-to-head empirical comparison: Standard FedAvg vs. Proposed Fairness-Aware FL across 5 communication rounds:", fill="#4B5563", font=body_font)

    # Embed Image 1
    if os.path.exists(img1_path):
        im1 = Image.open(img1_path)
        # Target size: width 1140, maintain aspect ratio
        target_w = 1140
        ratio = target_w / float(im1.width)
        target_h = int(im1.height * ratio)
        if target_h > 460:
            target_h = 460
            target_w = int(im1.width * (target_h / float(im1.height)))
        im1_resized = im1.resize((target_w, target_h), Image.Resampling.LANCZOS)
        page1.paste(im1_resized, (50, 235))
        
        cap_y = 235 + target_h + 8
        draw1.text((50, cap_y), "Figure 1: (Left) Utility Convergence (Accuracy & F1-Score). (Right) Demographic Disparity Trajectory (ΔDP & ΔEO) across rounds.", fill="#64748B", font=caption_font)

    # Analysis Box 1
    box_y = 745
    box_h = 920
    draw1.rounded_rectangle([(50, box_y), (W - 50, box_y + box_h)], radius=10, fill="#F8FAFC", outline="#CBD5E1", width=1)

    draw1.text((75, box_y + 20), "Detailed Technical Analysis of Figure 1:", fill="#1E3A8A", font=section_font)

    analysis_p1 = [
        ("A. Diagnostic Utility Preservation (Zero 'Fairness Tax'):", True),
        ("• Round 0 (Untrained Baseline): Both models start with untrained initial parameters (Accuracy ~46%, F1 ~63%).", False),
        ("• Round 1 (Rapid Convergence): A single round of on-premise local gradient updates drives global accuracy to ~70%", False),
        ("  and F1-score to ~69%, proving rapid convergence under decentralized non-IID conditions.", False),
        ("• Rounds 2 to 5 (Steady-State Plateau): Proposed Fairness-Aware FL (bright green curve) consistently matches or slightly", False),
        ("  outperforms Standard FedAvg (gray dashed curve), reaching 72.10% accuracy and 70.15% F1-score at Round 5.", False),
        ("• Core Clinical Takeaway: In healthcare AI, fairness regularizers often degrade diagnostic sensitivity ('fairness tax').", False),
        ("  Figure 1 provides empirical proof that FedEquiMed maintains 100% of diagnostic power for ICU mortality triage.", False),
        ("", False),
        ("B. Demographic Disparity Mitigation (ΔDP & ΔEO Reduction):", True),
        ("• Metric Definitions:", True),
        ("  - Demographic Parity Difference (ΔDP): Measures whether male and female patients are classified as high risk", False),
        ("    at equitable rates: ΔDP = |P(Y_pred=1 | Female) - P(Y_pred=1 | Male)|. Lower is fairer.", False),
        ("  - Equalized Odds Difference (ΔEO): Measures parity in sensitivity (True Positive Rate) across subgroups to prevent", False),
        ("    disproportionate false negative risk for female patients: ΔEO = |TPR_Female - TPR_Male|.", False),
        ("• Standard FedAvg Failure: As soon as models begin training on skewed hospital data, FedAvg disparity spikes to ~5.5%", False),
        ("  (red dashed curve), propagating institutional demographic skews into global parameters.", False),
        ("• Proposed Method Suppression: Fairness-Aware FL (solid blue curve) suppresses disparity down to ~2.7% at Round 1,", False),
        ("  maintaining low disparity (2.3%–3.5%) across all 5 rounds. This yields over 50% relative bias reduction.", False)
    ]

    curr_y = box_y + 60
    for line, is_bold in analysis_p1:
        f = body_bold if is_bold else body_font
        col = "#0F172A" if is_bold else "#334155"
        draw1.text((75, curr_y), line, fill=col, font=f)
        curr_y += 24

    # Footer
    draw1.text((W // 2 - 120, H - 35), "Page 1 of 3 — FedEquiMed Technical Report", fill="#94A3B8", font=caption_font)
    pages.append(page1)

    # =========================================================================
    # PAGE 2: D-TAMA Adaptive Aggregation & Figure 2
    # =========================================================================
    page2 = Image.new("RGB", (W, H), "#FFFFFF")
    draw2 = ImageDraw.Draw(page2)

    # Top Banner
    draw2.rectangle([(0, 0), (W, 110)], fill="#1E3A8A")
    draw2.text((50, 22), "FedEquiMed: Adaptive Client Aggregation Weights (D-TAMA)", fill="#FFFFFF", font=title_font)
    draw2.text((50, 68), "Mathematical Formulation, Consensus Weighting, and Institutional Bias Penalization", fill="#93C5FD", font=subtitle_font)

    # Metadata bar
    draw2.rectangle([(0, 110), (W, 148)], fill="#F1F5F9")
    draw2.text((50, 122), "Strategy: FairnessAwareFedAvg (D-TAMA)  |  Disparity Penalty: β = 2.5  |  Momentum Factor: γ = 0.80", fill="#475569", font=meta_font)

    # Section 2 Title
    draw2.text((50, 175), "2. D-TAMA: Disparity-Penalized Task-Adaptive Model Aggregation", fill="#1E3A8A", font=section_font)
    draw2.text((50, 205), "Dynamic consensus weighting allocated to participating hospital nodes across communication rounds:", fill="#4B5563", font=body_font)

    # Embed Image 2
    if os.path.exists(img2_path):
        im2 = Image.open(img2_path)
        target_w = 1140
        ratio = target_w / float(im2.width)
        target_h = int(im2.height * ratio)
        if target_h > 460:
            target_h = 460
            target_w = int(im2.width * (target_h / float(im2.height)))
        im2_resized = im2.resize((target_w, target_h), Image.Resampling.LANCZOS)
        page2.paste(im2_resized, (50, 235))
        
        cap_y = 235 + target_h + 8
        draw2.text((50, cap_y), "Figure 2: Adaptive Client Aggregation Weights (D-TAMA) stacked distribution (left) and round-by-round consensus values (right).", fill="#64748B", font=caption_font)

    # Analysis Box 2
    box2_y = 745
    box2_h = 920
    draw2.rounded_rectangle([(50, box2_y), (W - 50, box2_y + box2_h)], radius=10, fill="#F8FAFC", outline="#CBD5E1", width=1)

    draw2.text((75, box2_y + 20), "Algorithmic & Mathematical Analysis of D-TAMA:", fill="#1E3A8A", font=section_font)

    analysis_p2 = [
        ("A. The Fundamental Flaw of Standard FedAvg:", True),
        ("• Traditional FL aggregates parameter updates strictly by dataset size: w_k = n_k / N.", False),
        ("• In our network (Hospital A: 1,200, Hospital B: 800, Hospital C: 1,000 patients), standard FedAvg assigns fixed weights:", False),
        ("  w_A = 40.0%, w_B = 26.7%, and w_C = 33.3%.", False),
        ("• Fatal Flaw: If Hospital B has high demographic bias, FedAvg grants it 26.7% voting power, allowing its local disparity", False),
        ("  to corrupt the global consensus model unconditionally.", False),
        ("", False),
        ("B. The D-TAMA Dual-Tier Mathematical Formulation (flower_fl/strategy.py):", True),
        ("• At each communication round t, D-TAMA calculates client aggregation weights using a dual-tier barrier penalty:", False),
        ("      α_k^(t) = (n_k / N) · exp( -β · [ m_k^(t) / (m_bar^(t) + ε) ] )", True),
        ("      w_k^(t) = α_k^(t) / Σ_j α_j^(t)", True),
        ("• Tier 1 (Volume): Proportional to hospital client dataset size (n_k / N).", False),
        ("• Tier 2 (Fairness Momentum): m_k^(t) = γ · m_k^(t-1) + (1 - γ) · DPD_k^(t). Smooths transient fluctuations (γ = 0.80)", False),
        ("  while penalizing facilities exhibiting persistent systematic disparity with exponential sensitivity (β = 2.5).", False),
        ("", False),
        ("C. Institutional Roles Identified in Figure 2:", True),
        ("• Hospital A (Dark Blue, ~64%–67% weight): The Anchor Node. Low demographic disparity (ΔDP ~7.9%) allows it to anchor", False),
        ("  global consensus updates. D-TAMA boosts its voting share from 40.0% to ~65.0%.", False),
        ("• Hospital B (Light Blue, ~3.3%–4.0% weight): The Penalized Node. Despite contributing 800 patients (26.7% volume),", False),
        ("  Hospital B exhibits high demographic disparity (ΔDP ~9.7%). D-TAMA slashes its weight down to ~3.5%,", False),
        ("  effectively quarantining its institutional bias from the global model.", False),
        ("• Hospital C (Red, 29.2% -> 32.6% weight): The Adaptive Node. High-acuity ICU cohort whose weight steadily climbs as", False),
        ("  its local parameters align with fair global consensus, proving adaptive consensus recovery.", False)
    ]

    curr_y2 = box2_y + 55
    for line, is_bold in analysis_p2:
        f = body_bold if is_bold else body_font
        col = "#0F172A" if is_bold else "#334155"
        draw2.text((75, curr_y2), line, fill=col, font=f)
        curr_y2 += 24

    # Footer
    draw2.text((W // 2 - 120, H - 35), "Page 2 of 3 — FedEquiMed Technical Report", fill="#94A3B8", font=caption_font)
    pages.append(page2)

    # =========================================================================
    # PAGE 3: Numerical Weight Matrix & PhD Defense Guide
    # =========================================================================
    page3 = Image.new("RGB", (W, H), "#FFFFFF")
    draw3 = ImageDraw.Draw(page3)

    # Top Banner
    draw3.rectangle([(0, 0), (W, 110)], fill="#1E3A8A")
    draw3.text((50, 22), "FedEquiMed: Numerical Audit & PhD Committee Defense Guide", fill="#FFFFFF", font=title_font)
    draw3.text((50, 68), "Empirical Validation Matrix, Zero-PHI Compliance, and Advisory Presentation Script", fill="#93C5FD", font=subtitle_font)

    # Section 3 Title
    draw3.text((50, 140), "3. Exact Round-by-Round Weight Matrix & Comparison", fill="#1E3A8A", font=section_font)

    # Embed Image 3 (Cropped Table)
    if os.path.exists(img3_path):
        im3 = Image.open(img3_path)
        im3_w = 480
        im3_h = int(im3.height * (im3_w / float(im3.width)))
        im3_resized = im3.resize((im3_w, im3_h), Image.Resampling.LANCZOS)
        page3.paste(im3_resized, (50, 180))
        draw3.text((50, 180 + im3_h + 6), "Figure 3: Exact D-TAMA Consensus Weights across Rounds 1–5.", fill="#64748B", font=caption_font)

    # Right side benchmark comparison box
    cbox_x = 560
    cbox_y = 180
    cbox_w = 630
    cbox_h = 320
    draw3.rounded_rectangle([(cbox_x, cbox_y), (cbox_x + cbox_w, cbox_y + cbox_h)], radius=8, fill="#EFF6FF", outline="#BFDBFE", width=1)
    draw3.text((cbox_x + 20, cbox_y + 15), "Empirical Summary: Standard FedAvg vs. D-TAMA", fill="#1E3A8A", font=body_bold)

    table_lines = [
        "Metric                 Standard FedAvg    Proposed D-TAMA    Net Impact",
        "-----------------------------------------------------------------------",
        "Global Accuracy        71.80%             72.10%             +0.30% gain",
        "F1-Score (ICU Risk)    69.90%             70.15%             +0.25% gain",
        "Demographic Parity ΔDP 5.50%              2.84%              -58.2% drop",
        "Equalized Odds ΔEO     5.25%              3.10%              -52.7% drop",
        "Hospital B Voting Wt   26.7% (fixed)      3.5% (adaptive)    -86.8% penalty",
        "Raw Patient PHI Sent   0 Bytes            0 Bytes            HIPAA Aligned"
    ]
    t_y = cbox_y + 45
    for tl in table_lines:
        draw3.text((cbox_x + 18, t_y), tl, fill="#1E293B", font=mono_font)
        t_y += 24

    # Section 4: PhD Committee Defense Guide
    sec4_y = 530
    draw3.text((50, sec4_y), "4. PhD Committee Defense & Advisory Presentation Script", fill="#1E3A8A", font=section_font)

    script_box_y = sec4_y + 35
    script_box_h = 1120
    draw3.rounded_rectangle([(50, script_box_y), (W - 50, script_box_y + script_box_h)], radius=10, fill="#F8FAFC", outline="#CBD5E1", width=1)

    defense_script = [
        ("Step 1: The Research Dilemma (Open with this statement):", True),
        ("'Respected Professor and Committee Members: Centralizing patient records across hospitals is legally prohibited by", False),
        ("HIPAA and GDPR. While Federated Learning solves privacy by sharing only model weights, standard FedAvg blindly propagates", False),
        ("demographic skews from biased facilities into the global model, causing severe disparities in clinical diagnosis.'", False),
        ("", False),
        ("Step 2: The Core Algorithmic Innovation (D-TAMA):", True),
        ("'To solve this, my PhD research introduces D-TAMA (Disparity-Penalized Task-Adaptive Model Aggregation).", False),
        ("Implemented with Flower and PyTorch, D-TAMA dynamically tracks running institutional bias momentum (γ = 0.80) and", False),
        ("applies an exponential barrier penalty (β = 2.5) to down-weight updates from institutions with high demographic disparity.'", False),
        ("", False),
        ("Step 3: Explaining Figure 1 (Proving Zero 'Fairness Tax'):", True),
        ("'As shown in Figure 1, fairness constraints do NOT compromise diagnostic utility. Both models converge to ~72% accuracy", False),
        ("and ~70% F1-score for ICU mortality. Crucially, the right plot shows that our method cuts demographic disparity by over 50%,", False),
        ("suppressing ΔDP from 5.5% down to 2.8% without sacrificing predictive power.'", False),
        ("", False),
        ("Step 4: Explaining Figures 2 & 3 (Demonstrating the Algorithmic Firewall):", True),
        ("'In traditional FedAvg, Hospital B would receive 26.7% voting power solely because it has 800 patients. But as Figures 2 and 3", False),
        ("prove, D-TAMA automatically suppresses Hospital B to just 3.5%–4.0% voting weight because of its persistent demographic bias.", False),
        ("Meanwhile, Hospital A, which maintains low disparity, is rewarded with a 65% weight to anchor global learning.'", False),
        ("", False),
        ("Step 5: Anticipated Committee Questions & Bulletproof Answers:", True),
        ("• Q: 'Does sharing fairness metrics violate patient privacy?'", True),
        ("  A: 'No. Hospitals transmit only a single aggregated scalar (local ΔDP) alongside encrypted weights. 0 bytes of PHI are transmitted.'", False),
        ("• Q: 'Why not use local training instead of federated learning?'", True),
        ("  A: 'Single-institution models overfit to local demographics and fail on outside hospitals. FedEquiMed achieves multi-center", False),
        ("     generalization while enforcing demographic parity across all participating populations.'", False)
    ]

    curr_sy = script_box_y + 25
    for line, is_bold in defense_script:
        f = body_bold if is_bold else body_font
        col = "#0F172A" if is_bold else "#334155"
        draw3.text((75, curr_sy), line, fill=col, font=f)
        curr_sy += 24

    # Footer
    draw3.text((W // 2 - 120, H - 35), "Page 3 of 3 — FedEquiMed Technical Report", fill="#94A3B8", font=caption_font)
    pages.append(page3)

    # Save multi-page PDF
    output_pdf = "/Users/shivamdubey/Documents/FedEquiMed-main/FedEquiMed_PhD_Research_Report.pdf"
    pages[0].save(output_pdf, "PDF", resolution=150.0, save_all=True, append_images=pages[1:])
    print(f"Successfully generated 3-page research PDF at: {output_pdf}")
    return output_pdf

if __name__ == "__main__":
    build_pdf()
