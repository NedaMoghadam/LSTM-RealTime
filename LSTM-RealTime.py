import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Start overall timer
overall_start_time = time.time()

# -------------------------------
# 1. Data Loading and Preprocessing
# -------------------------------
file_path = r"D:\CV Neda\New folder\dataset\r4.2 (2)\r4.2\New folder\merged_data.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

start_time = time.time()
df = pd.read_csv(file_path, low_memory=False, nrows=200000)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date")
df["hour"] = df["date"].dt.hour
df["day_of_week"] = df["date"].dt.dayofweek
data_loading_time = time.time() - start_time

# -------------------------------
# 2. Feature Engineering for LSTM
# -------------------------------
# Encode categorical variables
le_user = LabelEncoder()
le_activity = LabelEncoder()
le_pc = LabelEncoder()
df["user_encoded"] = le_user.fit_transform(df["user"])
df["activity_encoded"] = le_activity.fit_transform(df["activity"])
df["pc_encoded"] = le_pc.fit_transform(df["pc"])

# Select features for LSTM (numerical and encoded categorical)
features = ["user_encoded", "activity_encoded", "pc_encoded", "hour", "day_of_week"]
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Define sequence length for LSTM
sequence_length = 10  # Number of events in each sequence

def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        seq = data[i:i + seq_length]
        sequences.append(seq)
    return np.array(sequences)

# -------------------------------
# 3. LSTM Autoencoder Model
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # Encoder
        _, (hidden, _) = self.encoder(x)
        # Repeat hidden state for decoder input
        hidden = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        # Decoder
        output, _ = self.decoder(hidden)
        # Fully connected layer to reconstruct input
        output = self.fc(output)
        return output

# Initialize model
input_dim = len(features)  # Number of features
hidden_dim = 64
num_layers = 2
lstm_autoencoder = LSTMAutoencoder(input_dim, hidden_dim, num_layers).to(device)
optimizer = torch.optim.Adam(lstm_autoencoder.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Function to train LSTM autoencoder
def train_autoencoder(model, data, epochs=5, batch_size=32):
    model.train()
    data_tensor = torch.FloatTensor(data).to(device)
    for epoch in range(epochs):
        for i in range(0, len(data), batch_size):
            batch = data_tensor[i:i + batch_size]
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
    return model

# Function to get reconstruction errors
def get_reconstruction_errors(model, data):
    model.eval()
    data_tensor = torch.FloatTensor(data).to(device)
    with torch.no_grad():
        output = model(data_tensor)
        mse = ((output - data_tensor) ** 2).mean(dim=(1, 2)).cpu().numpy()
    return mse

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

    # Suppress tqdm output by redirecting to /dev/null
    with tqdm(total=total_slots, desc=f"Processing {mode}-based slots", file=open(os.devnull, 'w')) as pbar:
        slot_count = 0
        yield res[1], res[0], minutes if mode == "time" else event_count, slide if mode == "time" else event_count, slot_count, mode
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
                # Log challenging batch message to anomaly log file
                with open(anomaly_log_path, "a", encoding="utf-8") as anomaly_log:
                    anomaly_log.write(f"Challenging batch at slot {slot_count} ({mode}-based)...\n")
            yield res[1], res[0], minutes if mode == "time" else event_count, slide if mode == "time" else event_count, slot_count, mode
            time.sleep(1)
            pbar.update(1)

# -------------------------------
# 5. Process Logs with LSTM Autoencoder
# -------------------------------
start_date = df["date"].min()
all_node_anomalies = []
window_metrics = []
processing_time_total = 0
first_slot = True

# Define anomaly log file path
anomaly_log_path = os.path.join(os.path.dirname(file_path), "anomaly_alerts.log")

# Clear anomaly log file at the start
with open(anomaly_log_path, "w", encoding="utf-8") as anomaly_log:
    anomaly_log.write("Anomaly Alerts:\n")

for mode in ["time", "event"]:
    if mode == "time":
        stream = data_stream(df, start_date, initial_minutes=10, initial_slide=5)
    else:
        stream = data_stream(df, start_date, event_count=100)

    for time_slot_df, current_slot_time, chunk_size, slide_size, slot_count, chunk_mode in stream:
        if time_slot_df.empty or current_slot_time is None:
            continue

        start_time = time.time()

        # Prepare data for LSTM
        data = time_slot_df[features].values
        if len(data) < sequence_length:
            continue  # Skip if not enough data for a sequence
        sequences = create_sequences(data, sequence_length)

        # Train or update LSTM autoencoder
        if first_slot or (slot_count % 12 == 0):
            lstm_autoencoder = train_autoencoder(lstm_autoencoder, sequences)
            first_slot = False

        # Get reconstruction errors
        anomaly_scores = get_reconstruction_errors(lstm_autoencoder, sequences)

        # Store anomaly scores
        for i, score in enumerate(anomaly_scores):
            # Adjust index to account for sequence length
            idx = i + sequence_length - 1
            if idx < len(time_slot_df):
                all_node_anomalies.append({
                    "timestamp": current_slot_time,
                    "user": time_slot_df.iloc[idx]["user"],
                    "pc": time_slot_df.iloc[idx]["pc"],
                    "activity": time_slot_df.iloc[idx]["activity"],
                    "score": score,
                    "chunk_size": chunk_size,
                    "slide_size": slide_size,
                    "mode": chunk_mode
                })

        # Real-time alerting (log to file instead of printing)
        threshold = pd.Series([entry["score"] for entry in all_node_anomalies]).quantile(0.95)
        num_anomalies = sum(score > threshold for score in anomaly_scores)
        for i, score in enumerate(anomaly_scores):
            idx = i + sequence_length - 1
            if idx < len(time_slot_df) and score > threshold:
                with open(anomaly_log_path, "a", encoding="utf-8") as anomaly_log:
                    anomaly_log.write(
                        f"ALERT: Anomalous behavior at {current_slot_time} for user {time_slot_df.iloc[idx]['user']} "
                        f"on pc {time_slot_df.iloc[idx]['pc']} with activity '{time_slot_df.iloc[idx]['activity']}' "
                        f"and score {score} ({chunk_mode}: {chunk_size}, Slide: {slide_size})\n"
                    )

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
# 6. Save and Evaluate Metrics
# -------------------------------
anomaly_df = pd.DataFrame(all_node_anomalies)
anomaly_df["anomaly_label"] = (anomaly_df["score"] > threshold).astype(int)

# Pseudo-ground truth: Top 6% of scores
pseudo_threshold = anomaly_df["score"].quantile(0.94)
anomaly_df["pseudo_label"] = (anomaly_df["score"] > pseudo_threshold).astype(int)

# Calculate metrics for original threshold (95th quantile)
precision = precision_score(anomaly_df["pseudo_label"], anomaly_df["anomaly_label"])
recall = recall_score(anomaly_df["pseudo_label"], anomaly_df["anomaly_label"])
accuracy = accuracy_score(anomaly_df["pseudo_label"], anomaly_df["anomaly_label"])
conf_matrix = confusion_matrix(anomaly_df["pseudo_label"], anomaly_df["anomaly_label"])

# Extract TN and FP from confusion matrix
tn, fp, fn, tp = conf_matrix.ravel()

# Calculate FPR and TNR
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

print(f"\nEvaluation Metrics (Pseudo-Ground Truth, 95th Quantile Threshold, contamination=0.06):")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"False Positive Rate (FPR): {fpr:.4f}")
print(f"True Negative Rate (TNR): {tnr:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# ROC Curve Analysis for ~5% FPR
y_true = anomaly_df["pseudo_label"].values
scores = anomaly_df["score"].values
fpr_curve, tpr_curve, thresh_curve = roc_curve(y_true, scores)

# Choose threshold for ~5% FPR
desired_fpr = 0.05
ix = np.argmin(np.abs(fpr_curve - desired_fpr))
threshold_at_5pct_fpr = thresh_curve[ix]
print(f"\nAt ~{fpr_curve[ix]*100:.1f}% FPR, threshold = {threshold_at_5pct_fpr:.3f}")

# Recompute metrics at ~5% FPR threshold
y_pred_5pct = (scores > threshold_at_5pct_fpr).astype(int)
conf_matrix_5pct = confusion_matrix(y_true, y_pred_5pct)
tn_5pct, fp_5pct, fn_5pct, tp_5pct = conf_matrix_5pct.ravel()

# Calculate metrics for ~5% FPR threshold
precision_5pct = tp_5pct / (tp_5pct + fp_5pct) if (tp_5pct + fp_5pct) > 0 else 0.0
recall_5pct = tp_5pct / (tp_5pct + fn_5pct) if (tp_5pct + fn_5pct) > 0 else 0.0
accuracy_5pct = (tp_5pct + tn_5pct) / (tp_5pct + tn_5pct + fp_5pct + fn_5pct) if (tp_5pct + tn_5pct + fp_5pct + fn_5pct) > 0 else 0.0
fpr_5pct = fp_5pct / (fp_5pct + tn_5pct) if (fp_5pct + tn_5pct) > 0 else 0.0
tnr_5pct = tn_5pct / (tn_5pct + fp_5pct) if (tn_5pct + fp_5pct) > 0 else 0.0

print(f"\nMetrics at ~5% FPR Threshold:")
print(f"Precision: {precision_5pct:.4f}")
print(f"Recall: {recall_5pct:.4f}")
print(f"Accuracy: {accuracy_5pct:.4f}")
print(f"False Positive Rate (FPR): {fpr_5pct:.4f}")
print(f"True Negative Rate (TNR): {tnr_5pct:.4f}")
print("Confusion Matrix:")
print(conf_matrix_5pct)

# ROC Curve Analysis for Max Precision with 0.005 < FPR ≤ 5%
desired_fpr = 0.05
min_fpr = 0.005  # Ensure small, non-zero FPR
best_prec = 0.0
best_thr = None

# Sweep thresholds to maximize precision with 0.005 < FPR ≤ 5%
for fpr_val, thr in zip(fpr_curve, thresh_curve):
    if min_fpr < fpr_val <= desired_fpr:
        y_pred = (scores > thr).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        if prec > best_prec:
            best_prec = prec
            best_thr = thr

# Fallback to ~5% FPR threshold if no suitable threshold found
if best_thr is None:
    best_thr = threshold_at_5pct_fpr
    y_pred = (scores > best_thr).astype(int)
    best_prec = precision_score(y_true, y_pred, zero_division=0)
    print(f"\nNo threshold found with {min_fpr*100:.1f}% < FPR ≤ {desired_fpr*100:.1f}%; using ~5% FPR threshold = {best_thr:.3f}")

print(f"\nBest precision {best_prec:.3f} at threshold {best_thr:.3f} with FPR ≤ {desired_fpr*100:.1f}%")

# Recompute metrics at best precision threshold
y_pred_best = (scores > best_thr).astype(int)
conf_matrix_best = confusion_matrix(y_true, y_pred_best)
tn_best, fp_best, fn_best, tp_best = conf_matrix_best.ravel()

# Calculate metrics for best precision threshold
precision_best = tp_best / (tp_best + fp_best) if (tp_best + fp_best) > 0 else 0.0
recall_best = tp_best / (tp_best + fn_best) if (tp_best + fn_best) > 0 else 0.0
accuracy_best = (tp_best + tn_best) / (tp_best + tn_best + fp_best + fn_best) if (tp_best + tn_best + fp_best + fn_best) > 0 else 0.0
fpr_best = fp_best / (fp_best + tn_best) if (fp_best + tn_best) > 0 else 0.0
tnr_best = tn_best / (tn_best + fp_best) if (tn_best + fp_best) > 0 else 0.0

print(f"Metrics at Best Precision Threshold (FPR ≤ 5%):")
print(f"Precision: {precision_best:.4f}")
print(f"Recall: {recall_best:.4f}")
print(f"Accuracy: {accuracy_best:.4f}")
print(f"False Positive Rate (FPR): {fpr_best:.4f}")
print(f"True Negative Rate (TNR): {tnr_best:.4f}")
print("Confusion Matrix:")
print(conf_matrix_best)

anomaly_df.to_csv("lstm_anomalies_full.csv", index=False)
metrics_df = pd.DataFrame(window_metrics)
metrics_df.to_csv("window_metrics_full.csv", index=False)

# -------------------------------
# 7. Visualize
# -------------------------------
plt.figure(figsize=(18, 12))

plt.subplot(2, 2, 1)
for mode in metrics_df["mode"].unique():
    for chunk_size in metrics_df[metrics_df["mode"] == mode]["chunk_size"].unique():
        scores = anomaly_df[(anomaly_df["mode"] == mode) & (anomaly_df["chunk_size"] == chunk_size)]["score"]
        sns.kdeplot(scores, label=f"{mode}: {chunk_size}", alpha=0.5)
plt.axvline(threshold, color="black", linestyle="--", label="Threshold (0.95)")
plt.axvline(pseudo_threshold, color="red", linestyle="--", label="Pseudo-Threshold (0.94)")
plt.axvline(threshold_at_5pct_fpr, color="green", linestyle="--", label="Threshold (~5% FPR)")
plt.axvline(best_thr, color="blue", linestyle="--", label=f"Threshold (Max Precision, FPR ≤ 5%)")
plt.title("Reconstruction Error Distribution by Chunk Size and Mode")
plt.xlabel("Reconstruction Error")
plt.ylabel("Density")
plt.legend()

plt.subplot(2, 2, 2)
sns.scatterplot(data=metrics_df, x="chunk_size", y="num_events", size="num_anomalies", hue="mode", style="slide_size")
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

# -------------------------------
# 8. Export Timing and Metrics Results
# -------------------------------
overall_time = time.time() - overall_start_time
log_file_path = os.path.join(os.path.dirname(file_path), os.path.basename(file_path).replace(".csv", "_log.txt"))
with open(log_file_path, "w", encoding="utf-8") as log_file:
    log_file.write("Timing Results:\n")
    log_file.write(f"Overall Time: {overall_time:.2f} seconds\n")
    log_file.write(f"Data Loading Time: {data_loading_time:.2f} seconds\n")
    log_file.write(f"Processing Time (Total): {processing_time_total:.2f} seconds\n")
    log_file.write("\nEvaluation Metrics (Pseudo-Ground Truth, 95th Quantile Threshold):\n")
    log_file.write(f"Precision: {precision:.4f}\n")
    log_file.write(f"Recall: {recall:.4f}\n")
    log_file.write(f"Accuracy: {accuracy:.4f}\n")
    log_file.write(f"False Positive Rate (FPR): {fpr:.4f}\n")
    log_file.write(f"True Negative Rate (TNR): {tnr:.4f}\n")
    log_file.write("Confusion Matrix:\n")
    log_file.write(str(conf_matrix) + "\n")
    log_file.write(f"\nEvaluation Metrics (Pseudo-Ground Truth, ~5% FPR Threshold = {threshold_at_5pct_fpr:.3f}):\n")
    log_file.write(f"Precision: {precision_5pct:.4f}\n")
    log_file.write(f"Recall: {recall_5pct:.4f}\n")
    log_file.write(f"Accuracy: {accuracy_5pct:.4f}\n")
    log_file.write(f"False Positive Rate (FPR): {fpr_5pct:.4f}\n")
    log_file.write(f"True Negative Rate (TNR): {tnr_5pct:.4f}\n")
    log_file.write("Confusion Matrix:\n")
    log_file.write(str(conf_matrix_5pct) + "\n")
    log_file.write(f"\nEvaluation Metrics (Pseudo-Ground Truth, Max Precision Threshold = {best_thr:.3f}, FPR ≤ 5%):\n")
    log_file.write(f"Precision: {precision_best:.4f}\n")
    log_file.write(f"Recall: {recall_best:.4f}\n")
    log_file.write(f"Accuracy: {accuracy_best:.4f}\n")
    log_file.write(f"False Positive Rate (FPR): {fpr_best:.4f}\n")
    log_file.write(f"True Negative Rate (TNR): {tnr_best:.4f}\n")
    log_file.write("Confusion Matrix:\n")
    log_file.write(str(conf_matrix_best) + "\n")
print(f"Timing and metrics results saved to {log_file_path}")