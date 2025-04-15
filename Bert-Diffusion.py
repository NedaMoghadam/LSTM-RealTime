import os
import time
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging

# Suppress joblib warning
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

# Set up logging
logging.basicConfig(
    filename="anomaly_alerts.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# Start overall timer
overall_start_time = time.time()

# -------------------------------
# 1. Data Loading and Preprocessing
# -------------------------------
file_path = r"D:\CV Neda\New folder\dataset\r4.2 (2)\r4.2\New folder\merged_data.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

start_time = time.time()
df = pd.read_csv(file_path, low_memory=False, nrows=10000)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date")
df["hour"] = df["date"].dt.hour
df["day_of_week"] = df["date"].dt.dayofweek
data_loading_time = time.time() - start_time

print("\nActivity Distribution in Input Data:")
print(df["activity"].value_counts())

# -------------------------------
# 2. LLM Setup
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased").to(device)


def get_llm_embeddings(texts, hours, days, batch_size=32):
    embeddings = []
    # Normalize temporal features
    hour_scaler = StandardScaler()
    day_scaler = StandardScaler()
    hours = hour_scaler.fit_transform(hours.reshape(-1, 1)).flatten()
    days = day_scaler.fit_transform(days.reshape(-1, 1)).flatten()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_hours = hours[i:i + batch_size]
        batch_days = days[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # Use [CLS] token
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        # Concatenate normalized temporal features
        temporal_features = np.vstack((batch_hours, batch_days)).T
        batch_embeddings = np.concatenate([batch_embeddings, temporal_features], axis=1)
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)


# -------------------------------
# 3. Anomaly Detection Setup
# -------------------------------
scaler = StandardScaler()


class DiffusionAnomalyDetector:
    def __init__(self, max_components=5):
        self.max_components = max_components
        self.gmm = None
        self.score_scaler = StandardScaler()

    def fit(self, X):
        n_samples = X.shape[0]
        # Dynamically adjust components
        n_components = min(self.max_components, n_samples)
        if n_components < 1:
            raise ValueError("Not enough samples to fit GMM")
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            reg_covar=1e-5,
            random_state=42
        )
        self.gmm.fit(X)
        scores = -self.gmm.score_samples(X).reshape(-1, 1)
        self.score_scaler.fit(scores)

    def score_samples(self, X):
        if self.gmm is None:
            raise ValueError("GMM not fitted")
        scores = -self.gmm.score_samples(X).reshape(-1, 1)
        return self.score_scaler.transform(scores).flatten()


# -------------------------------
# 4. Process Logs by Time Slots
# -------------------------------
def filter_by_time_range(df, start_date, minutes):
    end_date = start_date + pd.Timedelta(minutes=minutes)
    return df[(df['date'] >= start_date) & (df['date'] <= end_date)]


def filter_by_event_count(df, start_idx, event_count):
    end_idx = min(start_idx + event_count, len(df))
    return df.iloc[start_idx:end_idx], end_idx


def init_time_slot(df, start_date, minutes=None, event_count=None):
    if minutes is not None:
        res = filter_by_time_range(df, start_date, minutes)
        return [start_date, res, 0]
    elif event_count is not None:
        res, end_idx = filter_by_event_count(df, 0, event_count)
        return [df.iloc[0]["date"], res, end_idx]


def next_time_slot(df, start_date_or_idx, minutes=None, slide=None, event_count=None):
    if minutes is not None and slide is not None:
        new_start_date = start_date_or_idx + pd.Timedelta(minutes=slide)
        result_df = filter_by_time_range(df, new_start_date, minutes)
        return [new_start_date, result_df, 0]
    elif event_count is not None:
        new_start_idx = start_date_or_idx
        if new_start_idx >= len(df):
            return [None, pd.DataFrame(), None]
        result_df, end_idx = filter_by_event_count(df, new_start_idx, event_count)
        if result_df.empty:
            return [None, pd.DataFrame(), None]
        return [result_df.iloc[0]["date"], result_df, end_idx]


def data_stream(df, start_date, initial_minutes=10, initial_slide=5, event_count=None, challenge_interval=12):
    if event_count is not None:
        res = init_time_slot(df, start_date, event_count=event_count)
        total_slots = (len(df) // event_count) + 1
        mode = "event"
    else:
        minutes = initial_minutes
        slide = initial_slide
        res = init_time_slot(df, start_date, minutes=minutes)
        total_slots = ((df['date'].max() - start_date) // pd.Timedelta(minutes=slide)) + 1
        mode = "time"

    with tqdm(total=total_slots, desc=f"Processing {mode}-based slots") as pbar:
        slot_count = 0
        yield res[1], res[
            0], minutes if mode == "time" else event_count, slide if mode == "time" else event_count, slot_count, mode
        while res[1] is not None and not res[1].empty:
            slot_count += 1
            if mode == "time":
                if res[0] >= pd.Timestamp("2021-01-03"):
                    minutes = 60
                    slide = 15
                res = next_time_slot(df, res[0], minutes=minutes, slide=slide)
            else:
                res = next_time_slot(df, res[2], event_count=event_count)

            if slot_count % challenge_interval == 0:
                print(f"Challenging batch at slot {slot_count} ({mode}-based)...")
            yield res[1], res[
                0], minutes if mode == "time" else event_count, slide if mode == "time" else event_count, slot_count, mode
            time.sleep(1)
            pbar.update(1)


# Initialize variables
start_date = df["date"].min()
all_node_anomalies = []
window_metrics = []
processing_time_total = 0
first_slot = True
alert_counts = {}
MIN_SAMPLES = 5
diffusion_detector = None  # Initialize globally

# Process logs
for mode in ["time", "event"]:
    if mode == "time":
        stream = data_stream(df, start_date, initial_minutes=10, initial_slide=5)
    else:
        stream = data_stream(df, start_date, event_count=100)

    for time_slot_df, current_slot_time, chunk_size, slide_size, slot_count, chunk_mode in stream:
        if time_slot_df.empty or current_slot_time is None:
            continue

        start_time = time.time()

        # Prepare text data
        texts = time_slot_df.apply(
            lambda row: (
                f"{row['user']} ({row['employee_name']}, {row['role']}, {row['department']}) "
                f"on {row['pc']} at {row['date']} performed {row['activity']}"
            ),
            axis=1
        ).tolist()

        # Skip small batches
        if len(texts) < MIN_SAMPLES:
            logging.info(f"Skipping slot {slot_count} ({chunk_mode}): Only {len(texts)} samples")
            print(f"Skipping slot {slot_count} ({chunk_mode}): Only {len(texts)} samples")
            continue

        # Get embeddings with temporal features
        hours = time_slot_df["hour"].values
        days = time_slot_df["day_of_week"].values
        embeddings = get_llm_embeddings(texts, hours, days)

        # Scale embeddings
        if first_slot or (slot_count % 12 == 0):
            scaled_embeddings = scaler.fit_transform(embeddings)
            first_slot = False
        else:
            scaled_embeddings = scaler.transform(embeddings)

        # Anomaly detection
        if diffusion_detector is None or first_slot or (slot_count % 12 == 0):
            try:
                diffusion_detector = DiffusionAnomalyDetector(max_components=5)
                diffusion_detector.fit(scaled_embeddings)
            except ValueError as e:
                logging.info(f"Skipping slot {slot_count} ({chunk_mode}): {str(e)}")
                print(f"Skipping slot {slot_count} ({chunk_mode}): {str(e)}")
                continue

        iso_anomaly_scores = []
        num_anomalies = 0
        alert_counts[slot_count] = {}
        # Dynamic thresholds
        if all_node_anomalies:
            stats = pd.DataFrame(all_node_anomalies).groupby("activity")["score"].agg(["median", "std"])
            thresholds = {activity: stats.loc[activity, "median"] + 1.0 * stats.loc[activity, "std"]
                          for activity in stats.index}
        else:
            thresholds = {"default": 0.0}

        for i in range(len(scaled_embeddings)):
            embedding = scaled_embeddings[i].reshape(1, -1)
            score = diffusion_detector.score_samples(embedding)[0]
            iso_anomaly_scores.append(score)
            activity = time_slot_df.iloc[i]["activity"]

            all_node_anomalies.append({
                "timestamp": current_slot_time,
                "user": time_slot_df.iloc[i]["user"],
                "pc": time_slot_df.iloc[i]["pc"],
                "activity": activity,
                "score": score,
                "chunk_size": chunk_size,
                "slide_size": slide_size,
                "mode": chunk_mode
            })

            # Real-time alert
            threshold = thresholds.get(activity, thresholds.get("default", 0.0))
            alert_counts[slot_count].setdefault(activity, 0)
            if score > threshold and alert_counts[slot_count][activity] < 2:
                num_anomalies += 1
                alert_counts[slot_count][activity] += 1
                alert_msg = (f"REAL-TIME ALERT: User {time_slot_df.iloc[i]['user']} on PC {time_slot_df.iloc[i]['pc']} "
                             f"did '{activity}' with anomaly score {score:.2f} "
                             f"({chunk_mode}: {chunk_size}, Slide: {slide_size})")
                print(alert_msg)
                logging.info(alert_msg)

        # Print score statistics
        if iso_anomaly_scores:
            print(f"Slot {slot_count} ({chunk_mode}): Score Stats - Min: {min(iso_anomaly_scores):.2f}, "
                  f"Max: {max(iso_anomaly_scores):.2f}, Mean: {np.mean(iso_anomaly_scores):.2f}, "
                  f"Std: {np.std(iso_anomaly_scores):.2f}")

        window_metrics.append({
            "timestamp": current_slot_time,
            "chunk_size": chunk_size,
            "slide_size": slide_size,
            "num_events": len(time_slot_df),
            "num_anomalies": num_anomalies,
            "mode": chunk_mode
        })

        processing_time_total += time.time() - start_time

# -------------------------------
# 5. Save and Evaluate Metrics
# -------------------------------
anomaly_df = pd.DataFrame(all_node_anomalies)
if not anomaly_df.empty:
    threshold = anomaly_df["score"].quantile(0.90)
    anomaly_df["anomaly_label"] = (anomaly_df["score"] > threshold).astype(int)

    print("\nActivity Distribution in Anomalies:")
    print(anomaly_df["activity"].value_counts())

    print("\nAverage Anomaly Scores by Activity:")
    print(anomaly_df.groupby("activity")["score"].mean().sort_values(ascending=False))

    # Pseudo-ground truth
    pseudo_threshold = anomaly_df["score"].quantile(0.92)
    anomaly_df["pseudo_label"] = (anomaly_df["score"] > pseudo_threshold).astype(int)

    # Metrics
    precision = precision_score(anomaly_df["pseudo_label"], anomaly_df["anomaly_label"])
    recall = recall_score(anomaly_df["pseudo_label"], anomaly_df["anomaly_label"])
    accuracy = accuracy_score(anomaly_df["pseudo_label"], anomaly_df["anomaly_label"])
    conf_matrix = confusion_matrix(anomaly_df["pseudo_label"], anomaly_df["anomaly_label"])

    print(f"\nEvaluation Metrics (Pseudo-Ground Truth, contamination=0.08):")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    anomaly_df.to_csv("llm_anomalies_full.csv", index=False)
    metrics_df = pd.DataFrame(window_metrics)
    metrics_df.to_csv("window_metrics_full.csv", index=False)
else:
    print("No anomalies detected; skipping metrics and saving.")

# -------------------------------
# 6. Visualize
# -------------------------------
if not anomaly_df.empty:
    plt.figure(figsize=(18, 12))

    plt.subplot(2, 2, 1)
    for mode in metrics_df["mode"].unique():
        for chunk_size in metrics_df[metrics_df["mode"] == mode]["chunk_size"].unique():
            scores = anomaly_df[(anomaly_df["mode"] == mode) & (anomaly_df["chunk_size"] == chunk_size)]["score"]
            sns.kdeplot(scores, label=f"{mode}: {chunk_size}", alpha=0.5)
    plt.axvline(threshold, color="black", linestyle="--", label="Threshold (0.90)")
    plt.axvline(pseudo_threshold, color="red", linestyle="--", label="Pseudo-Threshold (0.92)")
    plt.title("Anomaly Score Distribution by Chunk Size and Mode")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Density")
    plt.legend()

    plt.subplot(2, 2, 2)
    sns.scatterplot(data=metrics_df, x="chunk_size", y="num_events", size="num_anomalies", hue="mode",
                    style="slide_size")
    plt.title("Events and Anomalies vs. Chunk Size")
    plt.xlabel("Chunk Size (minutes or events)")
    plt.ylabel("Number of Events")

    metrics_df["hour"] = metrics_df["timestamp"].dt.floor("h")
    hourly_anomalies = metrics_df.groupby(["hour", "mode", "chunk_size"])["num_anomalies"].sum().reset_index()
    plt.subplot(2, 2, 3)
    sns.lineplot(data=hourly_anomalies, x="hour", y="num_anomalies", hue="mode", style="chunk_size")
    plt.title("Anomalous Events per Hour by Mode and Chunk Size")
    plt.xlabel("Hour")
    plt.ylabel("Number of Anomalies")
    plt.xticks(rotation=45)

    metrics_df["anomaly_rate"] = metrics_df["num_anomalies"] / metrics_df["num_events"]
    plt.subplot(2, 2, 4)
    sns.lineplot(data=metrics_df, x="timestamp", y="anomaly_rate", hue="chunk_size", style="slide_size")
    plt.title("Anomaly Rate Over Time by Chunk Size and Slide")
    plt.xlabel("Timestamp")
    plt.ylabel("Anomaly Rate")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=anomaly_df, x="activity", y="score")
    plt.title("Anomaly Score Distribution by Activity Type")
    plt.xlabel("Activity")
    plt.ylabel("Anomaly Score")
    plt.xticks(rotation=45)
    plt.show()

# -------------------------------
# 7. Export Timing and Metrics Results
# -------------------------------
overall_time = time.time() - overall_start_time
log_file_path = os.path.join(os.path.dirname(file_path), os.path.basename(file_path).replace(".csv", "_log.txt"))
with open(log_file_path, "w") as log_file:
    log_file.write("Timing Results:\n")
    log_file.write(f"Overall Time: {overall_time:.2f} seconds\n")
    log_file.write(f"Data Loading Time: {data_loading_time:.2f} seconds\n")
    log_file.write(f"Processing Time (Total): {processing_time_total:.2f} seconds\n")
    if not anomaly_df.empty:
        log_file.write("\nEvaluation Metrics (Pseudo-Ground Truth):\n")
        log_file.write(f"Precision: {precision:.4f}\n")
        log_file.write(f"Recall: {recall:.4f}\n")
        log_file.write(f"Accuracy: {accuracy:.4f}\n")
        log_file.write("Confusion Matrix:\n")
        log_file.write(str(conf_matrix) + "\n")
    else:
        log_file.write("\nNo anomalies detected.\n")
print(f"Timing and metrics results saved to {log_file_path}")