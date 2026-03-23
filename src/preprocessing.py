import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from tensorflow.keras.preprocessing.sequence import pad_sequences  # noqa: E402

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 1. Data Loading & Cleaning
# ------------------------------------------------------------------

def augment_v_final_factory(df: pd.DataFrame, num_samples_per_class: int = 5000) -> pd.DataFrame:
    """
    V-Final Data Augmentation Factory:
    Injects high-variance synthetic edge cases to harden the model.
    """
    import random
    import string
    
    synthetic_domains = []
    
    # -------------------------------------------------------------
    # 1. Target 0 (Legit) - Safe but Entropic CDNs & Long names
    # -------------------------------------------------------------
    for _ in range(num_samples_per_class):
        # Pattern 1: [random_string].googlevideo.com / fbcdn / cloudfront
        rand_str_len = random.randint(15, 30)
        rand_str = "".join(random.choices(string.ascii_lowercase + string.digits + "-", k=rand_str_len))
        cdn_domain = rand_str + random.choice([".googlevideo.com", ".fbcdn.net", ".cloudfront.net"])
        synthetic_domains.append({"domain": cdn_domain, "label": 0})
        
        # Pattern 2: [ip-like-string].bc.googleusercontent.com
        ip_str = f"{random.randint(1,255)}-{random.randint(1,255)}-{random.randint(1,255)}-{random.randint(1,255)}"
        ip_domain = ip_str + ".bc.googleusercontent.com"
        synthetic_domains.append({"domain": ip_domain, "label": 0})
        
        # Pattern 3: Extremely long valid domains (like the Welsh town)
        long_words = ["llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch", "internationalsportsobserver", "thelongestdomainnameintheworldandthensomeandthensomemore", "reallylongdomainnamejustfortestingpurposes"]
        long_domain = random.choice(long_words) + random.choice([".co.uk", ".com", ".net", ".org", ".co"])
        synthetic_domains.append({"domain": long_domain, "label": 0})
        
    # -------------------------------------------------------------
    # 2. Target 1 (DGA) - Malicious but Clean Phishing/Rule-based
    # -------------------------------------------------------------
    tech_words = ["windows", "update", "telemetry", "sys", "adobe", "flash", "installer", "bin", "office", "auth", "verify", "security", "patch", "service", "java", "driver", "system", "host", "config"]
    dict_words = ["black", "water", "chair", "blue", "sky", "cloud", "server", "network", "system", "apple", "tree", "river", "stone", "flower", "mountain", "bird", "fish", "home", "paper", "glass", "board", "yellow", "bridge", "car", "camera", "market", "red", "green", "white", "house", "door", "window", "book", "pen", "table", "phone", "light", "star", "sun", "moon", "fire", "ice", "wind", "rain", "snow", "night", "day", "time", "food", "music", "art", "world", "life", "love", "peace", "war", "city", "street", "road", "train", "plane", "ship", "boat", "computer", "data", "code", "software", "hardware", "user", "admin", "login", "password", "email", "web", "site", "page", "link", "search", "game", "play", "stop", "start", "end", "top", "bottom", "left", "right", "high", "low", "fast", "slow", "good", "bad", "new", "old", "big", "small", "hot", "cold", "hard", "soft", "heavy", "light", "strong", "weak", "rich", "poor", "free", "cheap", "expensive", "buy", "sell", "pay", "money", "bank", "card", "shop", "store", "sale", "deal", "price", "cost", "value", "tax", "fee", "bill", "invoice", "receipt", "order", "track", "ship", "deliver"]
    tlds = [".com", ".net", ".org", ".info", ".biz", ".co"]
    
    # 2025/2026 High-Risk Scammer TLDs
    high_risk_tlds = [".top", ".xyz", ".monster", ".xin", ".bond", ".lol", ".cfd", ".vip", ".cc", ".ru"]
    
    # Combosquatting Brands & Keywords
    brands = ["paypal", "apple", "microsoft", "netflix", "amazon", "chase", "boa"]
    keywords = ["secure", "login", "support", "verify", "update", "auth", "billing"]
    
    for _ in range(num_samples_per_class):
        # Pattern 1: [tech-word]-[tech-word]-[random].net/com
        w1 = random.choice(tech_words)
        w2 = random.choice(tech_words)
        rand_ext = "".join(random.choices(string.ascii_lowercase + string.digits, k=random.randint(2, 5)))
        tld = random.choice([".com", ".net"])
        phish_domain = f"{w1}-{w2}-{rand_ext}{tld}"
        synthetic_domains.append({"domain": phish_domain, "label": 1})
        
        # Pattern 2: Dictionary DGAs ([word][word][word].com)
        n_words = random.randint(2, 4)
        chosen_words = random.sample(dict_words, n_words)
        dict_domain = "".join(chosen_words) + random.choice(tlds)
        synthetic_domains.append({"domain": dict_domain, "label": 1})

        # Pattern 3: Combosquatting (Brand Impersonation)
        b = random.choice(brands)
        k = random.choice(keywords)
        combo_tld = random.choice([".com", ".net"] + high_risk_tlds)
        combo = f"{b}-{k}{combo_tld}" if random.random() > 0.5 else f"{k}-{b}{combo_tld}"
        synthetic_domains.append({"domain": combo, "label": 1})

        # Pattern 4: High-Risk TLD Abuse (Dictionary or Random + high risk TLD)
        hr_tld = random.choice(high_risk_tlds)
        if random.random() > 0.5:
            hr_domain = random.choice(dict_words) + str(random.randint(1, 999)) + hr_tld
        else:
            hr_domain = "".join(random.choices(string.ascii_lowercase, k=random.randint(5, 12))) + hr_tld
        synthetic_domains.append({"domain": hr_domain, "label": 1})

        # Pattern 5: Typosquatting & Homoglyphs (Visual Deception)
        base_typos = random.sample(brands + ["google", "facebook", "yahoo", "amazon", "paypal"], 2)
        for base_typo in base_typos:
            if "l" in base_typo:
                typo = base_typo.replace("l", random.choice(["1", "I"]))
            elif "o" in base_typo:
                typo = base_typo.replace("o", "0")
            elif "m" in base_typo:
                typo = base_typo.replace("m", "rn")
            else:
                drop_idx = random.randint(1, len(base_typo)-2)
                typo = base_typo[:drop_idx] + base_typo[drop_idx+1:]
                
            typo_domain = f"{typo}{random.choice(tlds)}"
            synthetic_domains.append({"domain": typo_domain, "label": 1})
            
        # Explicit hardcore homoglyphs to override generic safety bias
        if random.random() < 0.2:
            synthetic_domains.extend([
                {"domain": f"g00gle{random.choice(tlds)}", "label": 1},
                {"domain": f"arnazon{random.choice(tlds)}", "label": 1},
                {"domain": f"paypa1{random.choice(tlds)}", "label": 1},
            ])
        
    logger.info(f"Generated {len(synthetic_domains)} synthetic edge cases across all V-Final Threat patterns.")
    synth_df = pd.DataFrame(synthetic_domains)
    return pd.concat([df, synth_df], ignore_index=True)


def load_data(filepath: str, domain_col: str, label_col: str,
              positive_label: str) -> pd.DataFrame:
    """Load the raw CSV, standardise column names, and augment data."""
    logger.info("Loading data from %s …", filepath)
    df = pd.read_csv(filepath)
    logger.info("Raw dataset shape: %s", df.shape)

    # CRITICAL TLD FIX: The CSV 'domain' column strips TLDs. We must use the 'host' column
    # so the model learns that TLDs are perfectly normal in legitimate traffic.
    if "host" in df.columns:
        df[domain_col] = df["host"]

    # Standardise column names ------------------------------------------------
    df = df.rename(columns={domain_col: "domain", label_col: "label"})
    df = df[["domain", "label"]].copy()

    # Drop missing values -----------------------------------------------------
    before = len(df)
    df.dropna(inplace=True)
    after = len(df)
    if before != after:
        logger.warning("Dropped %d rows with missing values.", before - after)

    # Encode label to binary (1 = DGA, 0 = Legit) ----------------------------
    df["label"] = (df["label"].str.strip().str.lower()
                   == positive_label.strip().lower()).astype(int)
                   
    # V-Final Upgrade: Augment Dataset with massive edge case factory
    df = augment_v_final_factory(df, num_samples_per_class=5000)
    
    logger.info("Label distribution after augmentation:\n%s", df["label"].value_counts().to_string())
    # Shuffle the dataset
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


# ------------------------------------------------------------------
# 2. Character-level Vectorisation (Option A)
# ------------------------------------------------------------------

def _build_char_vocab() -> dict:
    """Build a character-to-index mapping.

    Maps each printable ASCII character (codes 32-126) plus the
    dot/hyphen separators frequently found in domain names.
    Index 0 is reserved for padding, index 1 for out-of-vocabulary (OOV).

    Returns
    -------
    dict
        ``{character: integer_index}``
    """
    vocab: dict = {}
    # 0 = PAD, 1 = OOV
    for idx, ch in enumerate(
        [chr(c) for c in range(32, 127)], start=2
    ):
        vocab[ch] = idx
    return vocab


def vectorize_domains(domains: pd.Series,
                      max_len: int) -> np.ndarray:
    """Convert domain name strings into padded integer sequences.

    Each character is mapped to its vocabulary index. Sequences are
    post-padded / post-truncated to *max_len*.

    Parameters
    ----------
    domains : pd.Series
        Series of domain-name strings.
    max_len : int
        Fixed output length for every sequence.

    Returns
    -------
    np.ndarray
        2-D array of shape ``(n_samples, max_len)`` with dtype int32.
    """
    vocab = _build_char_vocab()
    oov_idx = 1

    encoded = []
    for domain in domains:
        seq = [vocab.get(ch, oov_idx) for ch in str(domain).lower()]
        encoded.append(seq)

    padded = pad_sequences(encoded, maxlen=max_len, padding="post",
                           truncating="post", value=0)
    logger.info("Vectorised %d domains → shape %s", len(domains), padded.shape)
    return padded


# ------------------------------------------------------------------
# 3. SMOTE Over-sampling
# ------------------------------------------------------------------

def apply_smote(X: np.ndarray, y: np.ndarray,
                random_state: int = 42) -> tuple:
    """Apply SMOTE to balance the training set.

    Because SMOTE produces continuous synthetic samples, the resulting
    feature vectors are rounded back to the nearest integer so they
    remain valid token indices for the embedding layer.

    Parameters
    ----------
    X : np.ndarray
        Encoded and padded domain sequences.
    y : np.ndarray
        Binary labels.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Balanced ``(X_resampled, y_resampled)``.
    """
    logger.info("Applying SMOTE (random_state=%d) …", random_state)
    logger.info("Before SMOTE — class distribution: %s",
                dict(zip(*np.unique(y, return_counts=True))))

    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X, y)

    # Round to nearest valid integer token and clip to valid range
    X_res = np.clip(np.round(X_res), 0, None).astype(np.int32)

    logger.info("After  SMOTE — class distribution: %s",
                dict(zip(*np.unique(y_res, return_counts=True))))
    return X_res, y_res


# ------------------------------------------------------------------
# 4. Data Splitting (7 : 2 : 1)
# ------------------------------------------------------------------

def split_data(X: np.ndarray, y: np.ndarray,
               train_ratio: float = 0.7,
               val_ratio: float = 0.2,
               test_ratio: float = 0.1,
               random_seed: int = 42) -> dict:
    """Split data into Train / Validation / Test sets.

    The splitting is performed in two stages:
    1. Separate *test_ratio* of the data as the held-out test set.
    2. Split the remainder into train and validation according to
       their relative proportions.

    Parameters
    ----------
    X, y : np.ndarray
        Feature matrix and label vector.
    train_ratio, val_ratio, test_ratio : float
        Must sum to 1.0.
    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    dict
        Keys: ``X_train, y_train, X_val, y_val, X_test, y_test``.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    # Stage 1: carve out the test set -----------------------------------------
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_seed, stratify=y
    )

    # Stage 2: split remaining into train / val --------------------------------
    relative_val = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val,
        random_state=random_seed, stratify=y_temp
    )

    logger.info("Split sizes → Train: %d | Val: %d | Test: %d",
                len(X_train), len(X_val), len(X_test))
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }
