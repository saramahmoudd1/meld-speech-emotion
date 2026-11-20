# Multimodal Speech Emotion Recognition on MELD

Deep Learning (AI403) final project, Pharos University in Alexandria, Fall 2026.

Team: [SonicAI] | Members: [Sara Mahmoud  202301747
Aya Ibrahim  202301658
Hend Mohamed  202303479
Zenab Yasser  202301702]

Chosen idea: no. 7, multimodal speech emotion recognition (audio + transcript -> emotion).

## 1. Task and best result

Utterance-level 7-class emotion recognition (neutral, joy, sadness, anger, surprise, fear, disgust) on MELD. Input: a speech clip and its transcript. Output: one emotion.

Best model: an MLP on frozen WavLM-base-plus (audio) and RoBERTa-base (text) features, using both modalities. Test set, mean +- SD over 3 seeds:

| Model | Weighted-F1 | Macro-F1 | Accuracy |
|---|---|---|---|
| Always neutral | 0.313 | 0.093 | 0.481 |
| Logistic regression baseline (fusion) | 0.461 | 0.312 | 0.449 |
| BiGRU from scratch (fusion) | 0.559 +- 0.012 | 0.369 +- 0.009 | 0.573 +- 0.014 |
| MLP on frozen WavLM + RoBERTa (fusion) | 0.626 +- 0.006 | 0.450 +- 0.005 | 0.642 +- 0.012 |
| Cross-modal attention (fusion) | 0.622 +- 0.006 | 0.432 +- 0.013 | 0.640 +- 0.007 |

Weighted-F1 and macro-F1 are the primary metrics. Accuracy is secondary because the always-neutral predictor already reaches 0.481. Full tables, ablation and error analysis are in the report (report/) and in experiments/results/.

## 2. Repository layout

    README.md
    requirements.txt
    configs/             one JSON config per experiment (written by the training scripts)
    data/
      prepare_meld.py    audio extraction and manifest (no raw data in the repo)
    src/
      datasets/          loader for the extracted features
      models/            focal loss and MLP head (the BiGRU and cross-attention classes are in train_seq.py and train_attn.py)
      utils/             seeds and metrics
      eda.py, majority_baseline.py, baseline.py, extract_features.py, extract_logmel.py
      train.py           MLP on frozen features
      train_seq.py       BiGRU from scratch
      train_attn.py      cross-modal attention (and the variant without it)
      evaluate.py        shared evaluation and paired bootstrap tests
      hpo_summary.py, error_analysis.py, auto_categorize.py, cost_table.py
    experiments/         results (json, csv, predictions) and figures
    checkpoints/         small checkpoints (the full set is on Drive)
    demo/                Gradio app
    report/              report PDF and slides
    contributions/       individual logs

Checkpoints and the processed manifest: [[Drive link](https://drive.google.com/drive/folders/1TnPBWNvEBI-8WIu8LksQngRhTsS5Aqfm)]

## 3. Environment setup

Tested on Windows with Python 3.11 and an NVIDIA GPU. ffmpeg must be on the PATH.

    python -m venv .venv
    .venv\Scripts\activate
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
    pip install -r requirements.txt
    winget install Gyan.FFmpeg

## 4. Dataset download and preprocessing

MELD is not included (about 10.9 GB). Download MELD.Raw.tar.gz from https://huggingface.co/datasets/declare-lab/MELD and extract it, then extract train.tar.gz, dev.tar.gz and test.tar.gz inside MELD.Raw:

    tar -xzf MELD.Raw.tar.gz
    cd MELD.Raw
    tar -xzf train.tar.gz
    tar -xzf dev.tar.gz
    tar -xzf test.tar.gz

From the repository root:

    python data/prepare_meld.py --root C:\meld\MELD.Raw --out C:\meld\processed
    python src/eda.py --manifest C:\meld\processed\manifest.csv
    python -m src.extract_features --manifest C:\meld\processed\manifest.csv
    python -m src.extract_logmel --manifest C:\meld\processed\manifest.csv

The official train, dev and test splits are used unchanged. Two corrupted clips are skipped. Features are written to C:\meld\features (about 3 GB).

## 5. Training commands

Every run writes configs/NAME.json, experiments/results/NAME.json and checkpoints/NAME_seedK.pt.

    # Baselines
    python -m src.majority_baseline --manifest C:\meld\processed\manifest.csv
    python -m src.baseline --manifest C:\meld\processed\manifest.csv

    # Classic: BiGRU from scratch
    python -m src.train_seq --modality audio --loss ce
    python -m src.train_seq --modality text --loss ce
    python -m src.train_seq --modality fusion --loss ce

    # Modern: MLP on frozen WavLM and RoBERTa features (selected configuration)
    python -m src.train --modality audio --loss ce --dropout 0.5 --tag _tuned
    python -m src.train --modality text --loss ce --dropout 0.5 --tag _tuned
    python -m src.train --modality fusion --loss ce --dropout 0.5 --tag _tuned

    # Advanced: cross-modal attention, and the same network without cross-attention
    python -m src.train_attn --cross 1 --dropout 0.5 --tag _tuned
    python -m src.train_attn --cross 0 --dropout 0.5 --tag _tuned

    # Data-strategy study (default dropout 0.3)
    python -m src.train --modality fusion --loss ce
    python -m src.train --modality fusion --loss wce
    python -m src.train --modality fusion --loss focal
    python -m src.train --modality fusion --loss focal_w

    # Hyperparameter study (seed 42, one factor at a time), for example
    python -m src.train --modality fusion --loss ce --seeds 42 --lr 3e-4 --tag _lrlow
    python -m src.train --modality fusion --loss ce --seeds 42 --dropout 0.5 --tag _do05

The list of all hyperparameter runs is in the report (Section 12) and in configs/.

## 6. Evaluation

    python -m src.evaluate --models mlp_audio_ce_tuned mlp_text_ce_tuned mlp_fusion_ce_tuned attn_cross_ce_tuned attn_nocross_ce_tuned gru_audio_ce gru_text_ce gru_fusion_ce --compare mlp_fusion_ce_tuned:mlp_text_ce_tuned attn_cross_ce_tuned:attn_nocross_ce_tuned attn_cross_ce_tuned:mlp_fusion_ce_tuned mlp_fusion_ce_tuned:gru_fusion_ce
    python -m src.hpo_summary
    python -m src.error_analysis --model mlp_fusion_ce_tuned --text_model mlp_text_ce_tuned --audio_model mlp_audio_ce_tuned
    python -m src.auto_categorize --model mlp_fusion_ce_tuned --text_model mlp_text_ce_tuned --audio_model mlp_audio_ce_tuned
    python -m src.cost_table --models mlp_audio_ce_tuned mlp_text_ce_tuned mlp_fusion_ce_tuned attn_cross_ce_tuned attn_nocross_ce_tuned gru_audio_ce gru_text_ce gru_fusion_ce

## 7. Demo

    pip install gradio "huggingface_hub<1.0"
    python -m demo.app

Open http://127.0.0.1:7860, upload or record a clip and type its transcript. The demo needs checkpoints/mlp_fusion_ce_tuned_seed43.pt (included) and internet access on the first run to download WavLM and RoBERTa. To check that it reproduces the saved test predictions:

    python -m demo.pipeline --self_test 50

## 8. Hardware and expected runtimes

Laptop with an NVIDIA GeForce RTX 3050 (6 GB), Python 3.11.

| Step | Time |
|---|---|
| Audio extraction with ffmpeg (13,706 clips) | roughly 30 minutes |
| MFCC features for the baseline | about 30 s |
| WavLM + RoBERTa feature extraction (all splits) | 582 s |
| Log-mel features | about 30 s |
| MLP training | 13 to 23 s per seed |
| BiGRU training | 64 s (text), 357 s (fusion), 625 s (audio) per run |
| Cross-modal attention training | about 270 s (without cross-attention) to 430 s (with) per seed |

## 9. Credits

- Dataset: MELD (Poria et al., ACL 2019), via Hugging Face declare-lab/MELD. See the dataset page for its licence.
- Pretrained models, used frozen: microsoft/wavlm-base-plus (WavLM, Chen et al.) and roberta-base (Liu et al.), through Hugging Face Transformers.
- Libraries: PyTorch, scikit-learn, librosa, soundfile, pandas, matplotlib, seaborn, Gradio, ffmpeg.
- Written by us: the focal loss (src/models/losses.py), the multi-head cross-attention layer (src/train_attn.py), all training, evaluation and analysis code.
- No external AI API is used for predictions.
