# CDG-DFD: a leakage-controlled evaluation of frequency-based deepfake detection

Code for the paper **"A Leakage Controlled Evaluation of Cross Domain Generalization in Frequency Based Deepfake Detection for Unseen GAN and Diffusion Models"** by Shaik Riyaz Ahamed and Monalisa Sahu (VIT-AP University). The detector studied is an EfficientNet-B4 on 2-D DCT input with radial-band spectral attention and domain-adversarial training.

Everything is here: the data rebuild, the leakage audit, every experiment reported in the paper, and every figure. Each number in the paper can be traced to a script and its saved output.

> Status: code accompanying a manuscript under review. The full citation will be added once the paper is published.

## Repository layout

| Path | What it is |
|---|---|
| `src/cdg_phase1.py` | Data pipeline: scan → MTCNN crop → exact/near-duplicate detection → group-disjoint splits → leakage audit (accounting, domain-label and file-format tables) |
| `src/cdg_train.py` | Model and trainer: DCT/RGB input, band spectral attention, optional SE block, DAT with gradient reversal, test-time corruptions, metrics with bootstrap CIs |
| `src/cdg_external.py` | External test on unseen generators (ArtiFact face subsets): sampling, cropping, scoring, report |
| `src/cdg_e3.py` | Multi-family training with leave-one-family-out test splits |
| `src/cdg_univfd.py` | UnivFD baseline (frozen CLIP ViT-L/14 + linear probe) under the same protocol |
| `notebooks/` | The Kaggle notebooks exactly as run (`phase1_*`, `phase2_E*`, `phase2_FIG4_roc`) |
| `notebooks/builders/` | Scripts that generate those notebooks from `src/` |
| `figures/` | Figure 1 (architecture), Figures 3–4 (`make_fig3_fig4.py`, from the E10 summary numbers) and the plotting cells used inside the experiment notebooks |
| `efficiency/` | Parameter / GMAC / latency measurement (E9) and its raw output |
| `tests/make_mock.py` | Tiny synthetic dataset for a fast end-to-end smoke test |

## Requirements

Python 3.10+, PyTorch 2.x with CUDA for training (2× Tesla T4 were used), and:

```
pip install timm facenet-pytorch --no-deps
pip install numpy pandas scikit-learn pillow opencv-python matplotlib open_clip_torch ftfy regex
```

## Data

The datasets are not redistributed. Download them from their sources:

- **DeepFakeFace (DFF)**: Song et al., 2023. Wiki reals, SD 1.5 text-to-image, SD inpainting, InsightFace.
- **Celeb-DF v2**: Li et al., CVPR 2020. Use the official test list.
- **HardFakeVsRealFaces**: Kaggle.
- **ArtiFact**: Rahman et al., ICIP 2023. Face subsets only, used as an external test.

`cdg_phase1.py scan` takes one root folder per dataset (`--hardfake-root` with `real/` and `fake/`; `--dff-root` with `wiki/`, `text2img/`, `inpainting/`, `insight/`; `--celebdf-root` with `Celeb-real/`, `Celeb-synthesis/`, `YouTube-real/`). See `--help`.

## Reproducing the paper

```bash
# 1. Data and protocol (CPU + GPU for MTCNN)
python src/cdg_phase1.py scan  --out DATA --hardfake-root <hardfake> --dff-root <dff> --dff-matched-real \
       --dff-insight-sample 8000 --celebdf-root <celebdf>
python src/cdg_phase1.py crop  --out DATA
python src/cdg_phase1.py dedup --out DATA
python src/cdg_phase1.py split --out DATA --ratios 0.7,0.1,0.2 --dff-real matched --split-name splits_matched
python src/cdg_phase1.py audit --out DATA --split-name splits_matched    # must report: leakage PASS

# 2. Main model (band SAM + DAT), fold A, seed 1
python src/cdg_train.py --data DATA --splits splits_matched --fold A --seed 1 --out runs/m_band_A_s1
```

The notebooks in `notebooks/` contain the exact command lines and settings for each experiment:

| Experiment | Notebook |
|---|---|
| Data rebuild and leakage audit | `phase1_data_rebuild`, `phase1_addon_matched` |
| E1/E1b: main model, 3 seeds × 2 folds, matched reals | `phase2_E1_train`, `phase2_E1b_matched` |
| E1c: 14 unseen ArtiFact generators | `phase2_E1c_external` |
| E2: controlled ablation (RGB, DCT, DCT+SE, DCT+band) | `phase2_E2_ablation` |
| E3: leave-one-family-out training | `phase2_E3_multifamily` |
| E5: robustness (JPEG, resize, blur, noise, chain) | `phase2_E5_robustness` |
| E6: RGB follow-up (ArtiFact, corruptions, throughput) | `phase2_E6_rgb_followup` |
| E9: parameters, GMACs, latency | `efficiency/e9_cost.py` |
| E10: UnivFD and CNNDetection baselines | `phase2_E10_baselines` |
| Figure 5: ROC curves and operating points | `phase2_FIG4_roc` |

Every training run writes the following, so tables and figures can be regenerated without retraining:
- `config.json`
- `summary.json` (including the validation threshold)
- `history.csv`
- `test_predictions.csv`
- `test_metrics.csv` (AUC with 95% bootstrap CI, EER, balanced accuracy)

## Protocol in one paragraph

**Splits.**
- Images are grouped by source video, source photo and near-duplicate links before splitting. No group crosses train, val and test.
- Fold A trains on SD inpainting and holds out SD 1.5 text-to-image. Fold B is the reverse.
- DFF reals are re-encoded to the fakes' storage format (512×512 JPEG, quality 75), so file history cannot separate the classes. A metadata-only classifier scores AUC 0.50.

**Selection and testing.** Checkpoints and thresholds are chosen on validation data only, and each test set is scored once.

**Known limitation.** Celeb-DF is split by video, not by identity, because its swap pairs link almost all 59 celebrities. The audit reports this overlap.

## Quick smoke test

```bash
pip install scikit-image
python tests/make_mock.py /tmp/mock     # small mock of the dataset layouts (real faces, so MTCNN runs)
# then run steps 1-2 above with the --*-root flags pointing into /tmp/mock and --epochs 1
```

## License

MIT License (see `LICENSE`). The datasets keep their own licences and are not included here.

## Contact

- Shaik Riyaz Ahamed, VIT-AP University (riyaz.25phd7140@vitap.ac.in)
- Dr. Monalisa Sahu, VIT-AP University, corresponding author (monalisa.sahu@vitap.ac.in)
