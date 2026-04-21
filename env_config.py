# =============================================================================
# env_config.py — Runtime configuration for ETH-GBERT
# "Dynamic Feature Fusion: Combining Global Graph Structures and Local
#  Semantics for Blockchain Fraud Detection" (Zhang et al., arXiv 2501.02032)
#
# Only settings actually consumed by the clean ETH-GBERT codebase are kept:
#   - ETH_GBert.py        (BERT + VocabGCN + DynamicFusionLayer + Classifier)
#   - trainModel.py       (training loop, BertAdam, cross-entropy)
#
# All removed fields and why:
#   BERT_NAME             — not in paper; trainModel.py hardcodes bert-base-uncased
#   GCN_DISABLE_IN_EMB    — ablation toggle, not in paper
#   AUX_MLM / MLM_PROB / MLM_LAMBDA
#                         — A2 auxiliary MLM head, not in paper
#   A3_ENABLE / A3_TOPK_ADDR / A3_VAL_BINS / A3_DT_BINS / A3_UNUSED_BUDGET
#                         — A3 multi-modal numeric/hash encoding, not in paper
#
# All three active fields below can be overridden via a .env file or shell
# environment variables without changing this file.
#
# Usage:
#   from env_config import env_config
#   seed = env_config.GLOBAL_SEED
# =============================================================================

import os
from pathlib import Path

from dotenv import load_dotenv


class EnvConfig:
    # Load .env from project root (three levels up), if it exists.
    # Variables already set in the shell environment take precedence.
    _env_path = Path(__file__).parent.parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path)

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    # Integer seed passed to random, numpy, and torch before any training run.
    # Paper does not state a specific seed; default 44 matches the codebase.
    GLOBAL_SEED: int = int(os.environ.get("GLOBAL_SEED", 44))

    # -------------------------------------------------------------------------
    # HuggingFace / Transformers offline mode
    # -------------------------------------------------------------------------
    # Set TRANSFORMERS_OFFLINE=1 in the environment to load BERT weights from a
    # local directory instead of downloading from the HuggingFace Hub.
    # Paper uses bert-base-uncased (§III-B3); that name is hardcoded in
    # trainModel.py and only redirected to a local path when this flag is 1.
    TRANSFORMERS_OFFLINE: int = int(os.environ.get("TRANSFORMERS_OFFLINE", 0))

    # Absolute path to the local model directory, used only when
    # TRANSFORMERS_OFFLINE == 1.  Expected layout:
    #   <HUGGING_LOCAL_MODEL_FILES_PATH>/hf-maintainers_bert-base-uncased/
    HUGGING_LOCAL_MODEL_FILES_PATH: str = os.environ.get(
        "HUGGING_LOCAL_MODEL_FILES_PATH", ""
    )


env_config = EnvConfig()