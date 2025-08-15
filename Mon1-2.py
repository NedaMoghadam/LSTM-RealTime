import os
import re
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
from torch.optim.lr_scheduler import CosineAnnealingLR
import gc
import json
from multiprocessing import Pool
import hashlib
import geoip2.database
from transformers import AlbertTokenizer, AlbertModel

# Set up logging with timestamped file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"anomaly_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

# Directories and files
json_file_path = "/datadrive/P2Final/p2_anonymized_final.json"
data_directory = "/datadrive/P2Final/users_combined"
save_directory = "/datadrive/P2Final/ALBERT_LSTM_Features"
daily_activity_file = os.path.join(save_directory, "daily_user_activity.csv")
os.makedirs(save_directory, exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Random seed
torch.manual_seed(42)
np.random.seed(42)

# ALBERT setup
logging.info("✅ Starting ALBERT loading...")
tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
albert_model = AlbertModel.from_pretrained("albert-base-v2").to(device)
logging.info("✅ ALBERT loaded.")

# Regex for sequence parsing
SEQUENCE_PATTERN = re.compile(r"<([^:]+):([^,>]+)(?:,([^>]+))?>")

# Configuration
CONFIG = {
    "time_zone_offset": -5,
    "unusual_hours_start": 0,
    "unusual_hours_end": 4,
    "event_actions": {
        "login": ["Logon", "Login", "UserLoggedIn"],
        "download": ["FileDownloaded", "FileAccessed", "download", "access"],
        "upload": ["FileUploaded", "upload", "write"],
        "delete": ["FileDeleted", "delete"]
    },
    "system_accounts": {"system", "admin", "svc-"},
    "excluded_countries": {"US", "CA"}
}

# Anomaly thresholds & weights
ANOMALY_SETTINGS = {
    'exchange_dlp':     {'percentile': 99, 'min_diff': 5,   'weight': 3},
    'sharepoint_dlp':   {'percentile': 99, 'min_diff': 5,   'weight': 3},
    'unusual_hours':    {'percentile': 99, 'min_diff': 5,   'weight': 1},
    'deleted':          {'percentile': 99, 'min_diff': 100, 'weight': 1},
    'download':         {'percentile': 99, 'min_diff': 100, 'weight': 1},
    'upload':           {'percentile': 99, 'min_diff': 100, 'weight': 1},
    'unusual_location': {'percentile': 99, 'min_diff': 1,   'weight': 2},
}

# Feature keys for consistency across functions
feature_keys = ["unusual_hours", "deleted", "download", "upload", "unusual_location",
                "exchange_dlp", "sharepoint_dlp"]

# Track per-user login hour history
user_hour_history = {}  # maps user -> set of hours seen

# GeoIP2 Reader
try:
    geoip_reader = geoip2.database.Reader('/datadrive/P2Final/GeoLite2-Country.mmdb')
except FileNotFoundError:
    logging.error("GeoLite2-Country.mmdb not found. Geolocation features will be limited.")
    geoip_reader = None

# Global dictionary for routine patterns
routine_patterns = {}

# Helper Functions
def parse_sequence(sequence):
    if isinstance(sequence, list):
        sequence = " ".join(str(s) for s in sequence if s)
    matches = SEQUENCE_PATTERN.findall(sequence)
    return matches if matches else [("Default", "NoActivity", None)]

def sequence_to_text(parsed_sequence):
    return " ".join([f"{category}:{action}" for category, action, _ in parsed_sequence])

def is_internal_ip(ip):
    return ip.startswith("192.168.") or ip.startswith("10.") or ip.startswith("172.")

def is_valid_ip(ip):
    try:
        parts = ip.split('.')
        return len(parts) == 4 and all(0 <= int(part) <= 255 for part in parts)
    except (ValueError, AttributeError):
        return False

def extract_features(parsed_sequence, user, date, df_day):
    features = {
        "unusual_hours": 0,
        "deleted": 0,
        "download": 0,
        "upload": 0,
        "unusual_location": 0,
        "exchange_dlp": 0,
        "sharepoint_dlp": 0
    }

    login_hours = []
    observed_countries = []

    # Initialize hour history
    if user not in user_hour_history:
        user_hour_history[user] = set()

    if any(acc in user.lower() for acc in CONFIG["system_accounts"]) or "test" in user.lower():
        return features

    for i, (category, action, param) in enumerate(parsed_sequence):
        if category in CONFIG["event_actions"]["login"]:
            if "date" in df_day.columns and i < len(df_day):
                login_time = pd.to_datetime(df_day["date"].iloc[i], errors="coerce")
                if not pd.isna(login_time):
                    login_hour = (login_time + timedelta(hours=CONFIG["time_zone_offset"])).hour
                    login_hours.append(login_hour)
                    if 0 <= login_hour < 6 and login_hour not in user_hour_history[user]:
                        features["unusual_hours"] += 1
                    user_hour_history[user].add(login_hour)
            if param and is_valid_ip(param) and not is_internal_ip(param):
                if geoip_reader:
                    try:
                        response = geoip_reader.country(param)
                        country_code = response.country.iso_code
                        if country_code and country_code not in CONFIG["excluded_countries"]:
                            features["unusual_location"] += 1
                            observed_countries.append(country_code)
                    except geoip2.errors.AddressNotFoundError:
                        pass

        elif any(act in action.lower() for act in CONFIG["event_actions"]["download"]):
            features["download"] += 1
        elif any(act in action.lower() for act in CONFIG["event_actions"]["upload"]):
            features["upload"] += 1
        elif any(act in action.lower() for act in CONFIG["event_actions"]["delete"]):
            features["deleted"] += 1
        elif "exchange_dlp" in action.lower():
            features["exchange_dlp"] += 1
        elif "sharepoint_dlp" in action.lower():
            features["sharepoint_dlp"] += 1

    if len(set(observed_countries)) > 1:
        features["unusual_location"] += len(set(observed_countries)) - 1

    features["user"] = user
    features["date"] = date
    return features

def get_albert_embedding(text, chunk_size=512, stride=256):
    text_hash = hashlib.md5(text.encode()).hexdigest()
    cache_file = os.path.join(save_directory, f"embedding_{text_hash}.npy")
    if os.path.exists(cache_file):
        return np.load(cache_file)
    if not text.strip():
        embedding = np.zeros(768)
    else:
        tokens = tokenizer.tokenize(text)
        token_chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), stride)]
        chunked_embeddings = []
        for chunk in token_chunks:
            inputs = tokenizer(" ".join(chunk), return_tensors="pt", truncation=True, max_length=512)
            inputs = {key: val.to(device) for key, val in inputs.items()}
            with torch.no_grad():
                output = albert_model(**inputs)
            chunked_embeddings.append(output.last_hidden_state.mean(dim=1).cpu().numpy().flatten())
        embedding = np.mean(chunked_embeddings, axis=0) if chunked_embeddings else np.zeros(768)
    np.save(cache_file, embedding)
    return embedding

def extract_daily_user_activity(filename):
    file_path = os.path.join(data_directory, filename)
    try:
        df = pd.read_csv(file_path)
        if df.empty or "sequence" not in df.columns or "date" not in df.columns:
            logging.warning(f"File {filename} is empty or missing required columns.")
            return []
        user = df["user"].iloc[0] if "user" in df.columns else filename.split(".")[0]
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        daily_data = []
        for date, group in df.groupby("date"):
            if pd.isna(date):
                continue
            sequence = " ".join(group["sequence"].dropna().astype(str).tolist()[:1000])
            daily_data.append({
                "User": user,
                "Date": str(date),
                "Sequence": sequence
            })
        return daily_data
    except Exception as e:
        logging.error(f"Error processing file {filename}: {e}")
        return []

def save_daily_user_activity(batch_size=100, num_workers=4, limit=601, use_json=False):
    if os.path.exists(daily_activity_file):
        logging.info(f"Loading existing daily user activity from {daily_activity_file}")
        return pd.read_csv(daily_activity_file)

    if use_json or not os.path.exists(data_directory) or not os.listdir(data_directory):
        logging.info(f"Using JSON file: {json_file_path}")
        try:
            with open(json_file_path, "r") as f:
                json_data = json.load(f)
            df = pd.DataFrame(json_data)
            df.to_csv(daily_activity_file, index=False)
            logging.info(f"Daily user activity saved to {daily_activity_file}")
            return df
        except Exception as e:
            logging.error(f"Failed to load JSON file {json_file_path}: {e}")
            return pd.DataFrame()

    csv_files = [f for f in os.listdir(data_directory) if f.endswith('.csv')][:limit]
    logging.info(f"Total files to process (limited to {limit} users): {len(csv_files)}")

    all_daily_data = []
    with Pool(num_workers) as pool:
        for batch_start in range(0, len(csv_files), batch_size):
            batch_files = csv_files[batch_start:batch_start + batch_size]
            logging.info(f"Processing batch {batch_start // batch_size + 1}: {len(batch_files)} files")
            results = pool.map(extract_daily_user_activity, batch_files)
            for daily_entries in results:
                all_daily_data.extend(daily_entries)
            gc.collect()

    if all_daily_data:
        df = pd.DataFrame(all_daily_data)
        df.to_csv(daily_activity_file, index=False)
        logging.info(f"Daily user activity saved to {daily_activity_file}")
        return df
    return pd.DataFrame()

def process_sequences(daily_activity_df):
    if daily_activity_df.empty:
        logging.error("No daily activity data to process.")
        return pd.DataFrame(), {}

    all_embeddings = []
    all_features = []
    user_info = []
    user_historical_data = {}

    for idx, row in daily_activity_df.iterrows():
        user = row["User"]
        date = row["Date"]
        sequence = row["Sequence"]
        parsed = parse_sequence(sequence)[:1000]
        text = sequence_to_text(parsed)
        embedding = get_albert_embedding(text)

        df_day = pd.DataFrame({"date": [date], "sequence": [sequence]})
        features = extract_features(parsed, user, date, df_day)

        feature_vector = np.array([features.get(k, 0) for k in feature_keys], dtype=np.float32)
        combined_vector = np.concatenate([embedding, feature_vector])

        if user not in user_historical_data:
            user_historical_data[user] = {"embeddings": [], "features": [], "combined": []}
        user_historical_data[user]["embeddings"].append(embedding)
        user_historical_data[user]["features"].append(feature_vector)
        user_historical_data[user]["combined"].append(combined_vector)

        all_embeddings.append(embedding)
        all_features.append(feature_vector)
        user_info.append({"User": user, "Date": date})

    for user in user_historical_data:
        embeddings = np.array(user_historical_data[user]["embeddings"])
        features = np.array(user_historical_data[user]["features"])
        routine_patterns[user] = {
            "embedding_median": np.median(embeddings, axis=0),
            "embedding_iqr": np.percentile(embeddings, 75, axis=0) - np.percentile(embeddings, 25, axis=0),
            "feature_median": np.median(features, axis=0),
            "feature_iqr": np.percentile(features, 75, axis=0) - np.percentile(features, 25, axis=0),
            "total_days": len(embeddings)
        }

    emb_df = pd.DataFrame(all_embeddings, columns=[f"emb_{i}" for i in range(768)])
    feat_df = pd.DataFrame(all_features, columns=feature_keys)
    combined_df = pd.concat([emb_df, feat_df], axis=1)
    combined_df["User"] = [info["User"] for info in user_info]
    combined_df["Date"] = [info["Date"] for info in user_info]

    combined_df.to_csv(os.path.join(save_directory, "combined_vectors.csv"), index=False)
    logging.info(f"Combined vectors saved to {os.path.join(save_directory, 'combined_vectors.csv')}")

    return combined_df, user_historical_data

class LSTMAnomalyDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_layers=3, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

def train_model(combined_df, user_historical_data):
    if combined_df.empty:
        logging.error("No combined vectors to train on. Exiting.")
        return None, None, None, None

    X = combined_df.drop(columns=["User", "Date"] + feature_keys).values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    n_components = min(50, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    joblib.dump(pca, os.path.join(save_directory, "pca_model.pkl"))

    dataset = TensorDataset(torch.tensor(X_pca, dtype=torch.float32).unsqueeze(1).to(device))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    input_dim = X_pca.shape[1]
    global_model = LSTMAnomalyDetector(input_dim=input_dim).to(device)
    optimizer = optim.AdamW(global_model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    patience = 15
    epochs_without_improvement = 0

    for epoch in range(50):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            output = global_model(batch[0].to(device))
            loss = criterion(output, batch[0].squeeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(global_model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        logging.info(f"Epoch {epoch + 1}, Loss: {total_loss:.6f}")
        if total_loss < best_loss:
            best_loss = total_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            logging.info("Early stopping triggered.")
            break

    torch.save(global_model.state_dict(), os.path.join(save_directory, "lstm_anomaly_model_global.pth"))

    # Train per-user LSTM models on embeddings only (users with ≥ 10 days)
    user_models = {}
    for user, hist in user_historical_data.items():
        embs = np.array(hist["embeddings"])  # shape (N_days, 768)
        if embs.shape[0] < 10:
            continue

        # Scale & PCA exactly as global
        embs_scaled = scaler.transform(embs)  # OK: 768 dims
        embs_pca = pca.transform(embs_scaled)

        ds = TensorDataset(torch.tensor(embs_pca, dtype=torch.float32).unsqueeze(1))
        dl = DataLoader(ds, batch_size=16, shuffle=True)

        m = LSTMAnomalyDetector(input_dim=embs_pca.shape[1]).to(device)
        opt = optim.AdamW(m.parameters(), lr=1e-3)
        crit = nn.MSELoss()
        for epoch in range(10):
            for (batch,) in dl:
                batch = batch.to(device)
                opt.zero_grad()
                out = m(batch)
                loss = crit(out, batch.squeeze(1))
                loss.backward()
                opt.step()
        user_models[user] = m
        torch.save(m.state_dict(), os.path.join(save_directory, f"lstm_anomaly_model_user_{user}.pth"))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return global_model, user_models, scaler, pca, X_pca

def detect_anomalies(combined_df, model, scaler, pca, X_pca, feature_thresholds):
    if combined_df.empty or model is None:
        logging.error("Cannot detect anomalies: Empty data or untrained model.")
        return pd.DataFrame()

    global_model, user_models = model
    X = combined_df.drop(columns=["User", "Date"] + feature_keys).values
    X_scaled = scaler.transform(X)
    X_pca_test = pca.transform(X_scaled)
    X_tensor = torch.tensor(X_pca_test, dtype=torch.float32).unsqueeze(1).to(device)

    # Global reconstruction errors
    with torch.no_grad():
        reconstructed = global_model(X_tensor).detach().cpu().numpy()
    global_errors = np.mean(np.abs(X_pca_test - reconstructed), axis=1)

    # Per-user reconstruction errors
    user_errors = []
    for idx, row in combined_df.iterrows():
        u = row["User"]
        vec = torch.tensor(X_pca_test[idx], dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)
        if u in user_models:
            with torch.no_grad():
                out_u = user_models[u](vec).cpu().numpy().squeeze()
            err_u = np.mean(np.abs(X_pca_test[idx] - out_u))
        else:
            err_u = np.nan
        user_errors.append(err_u)

    df_anomalies = pd.DataFrame({
        "User": combined_df["User"],
        "Date": combined_df["Date"],
        "Recon_Error_Global": global_errors,
        "Recon_Error_User": user_errors
    })

    # Feature-anomaly score
    feat_scores = []
    for idx, row in combined_df.iterrows():
        s = 0
        for feat, cfg in ANOMALY_SETTINGS.items():
            val = row.get(feat, 0)
            if val > feature_thresholds.get(feat, 0):
                s += cfg["weight"]
        feat_scores.append(s)
    df_anomalies["Feature_Anom_Score"] = feat_scores

    # Per-user deviation scores
    deviation_scores = []
    for idx, row in df_anomalies.iterrows():
        user = row["User"]
        combined_vector = combined_df.iloc[idx].drop(["User", "Date"] + feature_keys).values
        if user in routine_patterns:
            embedding_median = routine_patterns[user]["embedding_median"]
            embedding_iqr = routine_patterns[user]["embedding_iqr"]
            feature_median = routine_patterns[user]["feature_median"]
            feature_iqr = routine_patterns[user]["feature_iqr"]

            embedding_dev = np.mean(np.abs(combined_vector - embedding_median) / (embedding_iqr + 1e-6))
            feature_dev = np.mean(np.abs(combined_df.iloc[idx][feature_keys].values - feature_median) / (feature_iqr + 1e-6))
            deviation_score = (embedding_dev + feature_dev) / 2
        else:
            deviation_score = 0
        deviation_scores.append(deviation_score)

    df_anomalies["Deviation_Score"] = deviation_scores

    # Compute Z-scores
    user_stats = {}
    for user in df_anomalies["User"].unique():
        user_data = df_anomalies[df_anomalies["User"] == user]
        user_errors_global = user_data["Recon_Error_Global"]
        user_errors_user = user_data["Recon_Error_User"].dropna()
        user_dev_scores = user_data["Deviation_Score"]
        user_feat_scores = user_data["Feature_Anom_Score"]
        user_stats[user] = {
            "recon_global_mean": np.mean(user_errors_global),
            "recon_global_std": np.std(user_errors_global),
            "recon_user_mean": np.mean(user_errors_user) if len(user_errors_user) > 0 else np.nan,
            "recon_user_std": np.std(user_errors_user) if len(user_errors_user) > 0 else np.nan,
            "dev_mean": np.mean(user_dev_scores),
            "dev_std": np.std(user_dev_scores),
            "feat_mean": np.mean(user_feat_scores),
            "feat_std": np.std(user_feat_scores),
            "total_days": routine_patterns[user]["total_days"] if user in routine_patterns else 0,
            "max_z_score": 0
        }

    df_anomalies["Recon_Global_Z_Score"] = [
        (row["Recon_Error_Global"] - user_stats[row["User"]]["recon_global_mean"]) /
        (user_stats[row["User"]]["recon_global_std"] + 1e-6)
        for idx, row in df_anomalies.iterrows()
    ]
    df_anomalies["Recon_User_Z_Score"] = [
        (row["Recon_Error_User"] - user_stats[row["User"]]["recon_user_mean"]) /
        (user_stats[row["User"]]["recon_user_std"] + 1e-6)
        if not np.isnan(row["Recon_Error_User"]) else np.nan
        for idx, row in df_anomalies.iterrows()
    ]
    df_anomalies["Dev_Z_Score"] = [
        (row["Deviation_Score"] - user_stats[row["User"]]["dev_mean"]) /
        (user_stats[row["User"]]["dev_std"] + 1e-6)
        for idx, row in df_anomalies.iterrows()
    ]
    df_anomalies["Feature_Z_Score"] = [
        (row["Feature_Anom_Score"] - user_stats[row["User"]]["feat_mean"]) /
        (user_stats[row["User"]]["feat_std"] + 1e-6)
        for idx, row in df_anomalies.iterrows()
    ]

    # Compute final combined score
    α, β, γ, δ = 0.3, 0.3, 0.2, 0.2
    df_anomalies["Final_Score"] = (
        α * (df_anomalies["Recon_Global_Z_Score"] - df_anomalies["Recon_Global_Z_Score"].mean()) /
            (df_anomalies["Recon_Global_Z_Score"].std() + 1e-6) +
        β * (df_anomalies["Recon_User_Z_Score"].fillna(0) - df_anomalies["Recon_User_Z_Score"].fillna(0).mean()) /
            (df_anomalies["Recon_User_Z_Score"].fillna(0).std() + 1e-6) +
        γ * df_anomalies["Dev_Z_Score"] +
        δ * (df_anomalies["Feature_Z_Score"] - df_anomalies["Feature_Z_Score"].mean()) /
            (df_anomalies["Feature_Z_Score"].std() + 1e-6)
    )

    # Re-rank by Final_Score descending
    df_anomalies = df_anomalies.sort_values("Final_Score", ascending=False)

    # Flag anomalies
    is_anomaly = []
    for idx, row in df_anomalies.iterrows():
        user = row["User"]
        recon_global_z = row["Recon_Global_Z_Score"]
        recon_user_z = row["Recon_User_Z_Score"]
        dev_z = row["Dev_Z_Score"]
        feat_z = row["Feature_Z_Score"]
        final_score = row["Final_Score"]

        user_stats_user = user_stats[user]
        is_global_anomaly = recon_global_z > 2.5 if user_stats_user["recon_global_std"] >= 0.005 else False
        is_user_anomaly = recon_user_z > 2.5 if not np.isnan(recon_user_z) and user_stats_user["recon_user_std"] >= 0.005 else False
        is_dev_anomaly = dev_z > 2.5 if user_stats_user["dev_std"] >= 0.005 else False
        is_feat_anomaly = feat_z > 2.5 if user_stats_user["feat_std"] >= 0.005 else False
        is_final_anomaly = final_score > df_anomalies["Final_Score"].quantile(0.99)

        is_anomaly.append(is_global_anomaly or is_user_anomaly or is_dev_anomaly or is_feat_anomaly or is_final_anomaly)
        if max(recon_global_z, dev_z, feat_z, recon_user_z if not np.isnan(recon_user_z) else 0) > user_stats_user["max_z_score"]:
            user_stats_user["max_z_score"] = max(recon_global_z, dev_z, feat_z, recon_user_z if not np.isnan(recon_user_z) else 0)

    df_anomalies["Is_Anomaly"] = is_anomaly

    # Apply Z-score cutoff and limit to top 50 anomalies
    significant_anomalies = df_anomalies[
        (df_anomalies["Is_Anomaly"]) &
        (
            (df_anomalies["Recon_Global_Z_Score"] > 4.0) |
            (df_anomalies["Recon_User_Z_Score"] > 4.0) |
            (df_anomalies["Dev_Z_Score"] > 4.0) |
            (df_anomalies["Feature_Z_Score"] > 4.0) |
            (df_anomalies["Final_Score"] > df_anomalies["Final_Score"].quantile(0.99))
        )
    ].sort_values(by="Final_Score", ascending=False).head(50)
    num_anomalies = len(significant_anomalies)
    logging.info(f"Total number of significant anomalies detected (Z-Score > 4.0 or Final_Score > 99th percentile, top 50): {num_anomalies}")

    # Log top 10 anomalies
    if num_anomalies > 0:
        logging.info("Top 10 Significant Anomalies Detected (by Final_Score):")
        top_to_log = significant_anomalies.head(10)
        for idx, row in top_to_log.iterrows():
            logging.info(
                f"User: {row['User']}, Date: {row['Date']}, "
                f"Recon_Error_Global: {row['Recon_Error_Global']:.4f}, "
                f"Recon_Error_User: {row['Recon_Error_User']:.4f}, "
                f"Deviation_Score: {row['Deviation_Score']:.4f}, "
                f"Feature_Anom_Score: {row['Feature_Anom_Score']:.4f}, "
                f"Final_Score: {row['Final_Score']:.4f}, "
                f"Recon_Global_Z_Score: {row['Recon_Global_Z_Score']:.4f}, "
                f"Recon_User_Z_Score: {row['Recon_User_Z_Score']:.4f}, "
                f"Dev_Z_Score: {row['Dev_Z_Score']:.4f}, "
                f"Feature_Z_Score: {row['Feature_Z_Score']:.4f}"
            )

    # Per-user summary
    logging.info("Per-User Anomaly Summary:")
    for user, stats in user_stats.items():
        anomalies_detected = len(df_anomalies[(df_anomalies["User"] == user) & (df_anomalies["Is_Anomaly"])])
        logging.info(
            f"User: {user} | Total Days: {stats['total_days']} | "
            f"Anomalies Detected: {anomalies_detected} | Max Z-Score: {stats['max_z_score']:.4f}"
        )

    # Save results
    output_file = os.path.join(save_directory, "daily_anomalies.csv")
    df_anomalies.to_csv(output_file, index=False)
    logging.info(f"All anomalies saved to {output_file}")

    significant_output_file = os.path.join(save_directory, "significant_anomalies.csv")
    significant_anomalies.to_csv(significant_output_file, index=False)
    logging.info(f"Top 50 significant anomalies saved to {significant_output_file}")

    return df_anomalies

# Main execution
if __name__ == "__main__":
    logging.info("Extracting and saving daily user activity...")
    daily_activity_df = save_daily_user_activity()

    if not daily_activity_df.empty:
        logging.info("Processing sequences and combining features...")
        combined_df, user_historical_data = process_sequences(daily_activity_df)

        # Compute feature thresholds
        feature_thresholds = {}
        for feat, cfg in ANOMALY_SETTINGS.items():
            if feat in combined_df.columns:
                col = combined_df[feat]
                p = np.percentile(col, cfg['percentile'])
                feature_thresholds[feat] = p + cfg['min_diff']
            else:
                logging.warning(f"Feature {feat} not found in combined_df. Setting threshold to 0.")
                feature_thresholds[feat] = 0
        logging.info(f"Feature thresholds: {feature_thresholds}")

        if not combined_df.empty:
            logging.info("Training LSTM model on combined vectors...")
            global_model, user_models, scaler, pca, X_pca = train_model(combined_df, user_historical_data)
            if global_model:
                logging.info("Detecting anomalies...")
                df_anomalies = detect_anomalies(combined_df, (global_model, user_models), scaler, pca, X_pca, feature_thresholds)
                logging.info("Process completed.")

                if not df_anomalies.empty:
                    significant_anomalies = df_anomalies[df_anomalies["Is_Anomaly"]]
                    num_anomalies = len(significant_anomalies)
                    logging.info(f"Total number of significant anomalies detected: {num_anomalies}")

                    if num_anomalies > 0:
                        logging.info("Top 10 anomalies by Final_Score:")
                        top_anomalies = significant_anomalies.sort_values(by="Final_Score", ascending=False).head(10)
                        for idx, row in top_anomalies.iterrows():
                            logging.info(
                                f"User: {row['User']}, Date: {row['Date']}, "
                                f"Recon_Error_Global: {row['Recon_Error_Global']:.4f}, "
                                f"Recon_Error_User: {row['Recon_Error_User']:.4f}, "
                                f"Deviation_Score: {row['Deviation_Score']:.4f}, "
                                f"Feature_Anom_Score: {row['Feature_Anom_Score']:.4f}, "
                                f"Final_Score: {row['Final_Score']:.4f}"
                            )
                    else:
                        logging.info("No significant anomalies detected.")
                else:
                    logging.info("No anomalies detected.")
        else:
            logging.error("No combined vectors processed.")
    else:
        logging.error("No daily user activity data available. Check input files.")