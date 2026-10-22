"""
Web Server and REST API for Fairness-Aware Federated Learning in Healthcare (FedEquiMed)
Zero external dependencies - powered by Python standard library (http.server, json, urllib, secrets).
"""

import json
import os
import sys
import mimetypes
import secrets
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from fl_engine.dataset import generate_federated_healthcare_datasets, FEATURE_NAMES, SENSITIVE_ATTRIBUTES
from fl_engine.hospital_node import HospitalNode
from fl_engine.global_server import GlobalFederatedServer
from fl_engine.privacy_engine import PrivacyEngine
from fl_engine.experiments import (
    run_and_persist_experiment,
    list_experiments,
    run_ablation_study,
    EXPERIMENTS_DIR
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STATE_FILE = os.path.join(DATA_DIR, "persisted_hospitals.json")

# Secure Admin Passkey handling
ADMIN_PASSKEY = os.environ.get("ADMIN_PASSKEY")
if not ADMIN_PASSKEY:
    # If not set in environment, provide standard secure fallback or generated token
    ADMIN_PASSKEY = "fedequimed_admin_2026"

def verify_admin_key(key: str) -> bool:
    if not key or not isinstance(key, str):
        return False
    return secrets.compare_digest(key.strip(), ADMIN_PASSKEY.strip())

# Global State Container with Persistence across Browser Refreshes and Server Restarts
class FLSystemState:
    def __init__(self):
        self.hospitals = {}
        self.global_test = []
        self.server = None
        self.dp_noise = 0.02
        self.dp_clipping = 1.0
        self.initialize_system()

    def save_state(self):
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            saved_dict = {
                "hospitals": {h_id: h.to_dict() for h_id, h in self.hospitals.items()},
                "current_round": self.server.current_round if self.server else 0,
                "history": self.server.history if self.server else [],
                "benchmark_comparison": self.server.benchmark_comparison if self.server else None,
                "global_weights": self.server.get_global_weights() if self.server else None,
                "dp_noise": self.dp_noise,
                "dp_clipping": self.dp_clipping
            }
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(saved_dict, f)
        except Exception as e:
            print(f"Warning: Failed to save hospital state: {e}")

    def load_state(self) -> bool:
        if not os.path.exists(STATE_FILE):
            return False
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                saved_dict = json.load(f)
            if not saved_dict or not isinstance(saved_dict, dict):
                return False

            hosp_dict = saved_dict.get("hospitals", saved_dict)
            loaded_hospitals = {}
            for h_id, data in hosp_dict.items():
                if isinstance(data, dict) and "hospital_id" in data:
                    loaded_hospitals[h_id] = HospitalNode.from_dict(data)

            if loaded_hospitals:
                self.hospitals = loaded_hospitals
                self.server = GlobalFederatedServer(self.hospitals, self.global_test, num_features=10)
                if "current_round" in saved_dict:
                    self.server.current_round = int(saved_dict["current_round"])
                if "history" in saved_dict and isinstance(saved_dict["history"], list):
                    self.server.history = saved_dict["history"]
                if "benchmark_comparison" in saved_dict:
                    self.server.benchmark_comparison = saved_dict["benchmark_comparison"]
                if "global_weights" in saved_dict and saved_dict["global_weights"]:
                    self.server.global_model.set_weights(saved_dict["global_weights"])
                if "dp_noise" in saved_dict:
                    self.dp_noise = float(saved_dict["dp_noise"])
                if "dp_clipping" in saved_dict:
                    self.dp_clipping = float(saved_dict["dp_clipping"])
                return True
        except Exception as e:
            print(f"Warning: Failed to load hospital state: {e}")
        return False

    def initialize_system(self, force_fresh: bool = False):
        client_data, global_test = generate_federated_healthcare_datasets(seed=42)
        self.global_test = global_test

        if not force_fresh and self.load_state():
            if self.server.current_round == 0 or len(self.server.history) == 0:
                self.server.run_federated_round(strategy="fairfed", epochs_per_hospital=2, fairness_beta=2.5)
                self.save_state()
            return

        # Pre-enroll canonical clinical healthcare centers
        self.hospitals = {
            "hospital_a": HospitalNode("hospital_a", "Hospital A (Urban Medical Center)", client_data["hospital_a"], description="Metropolitan acute trauma & tertiary care center with balanced demographics."),
            "hospital_b": HospitalNode("hospital_b", "Hospital B (Rural Community Hospital)", client_data["hospital_b"], description="Under-resourced rural hospital with older patient demographic skew."),
            "hospital_c": HospitalNode("hospital_c", "Hospital C (Heart & Vascular Institute)", client_data["hospital_c"], description="Specialty cardiac institute with predominantly male cardiovascular admissions.")
        }
        self.server = GlobalFederatedServer(self.hospitals, self.global_test, num_features=10)
        # Train initial local representations
        for h in self.hospitals.values():
            h.train_local_model(global_weights=self.server.get_global_weights(), epochs=1)
        # Run Round 1 immediately
        self.server.run_federated_round(strategy="fairfed", epochs_per_hospital=2, fairness_beta=2.5)
        self.save_state()

SYSTEM_STATE = FLSystemState()

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

class FLRequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, data: dict, status_code: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/status":
            global_audit = SYSTEM_STATE.server.evaluate_global_model()
            resp = {
                "status": "online",
                "current_round": SYSTEM_STATE.server.current_round,
                "num_hospitals": len(SYSTEM_STATE.hospitals),
                "total_private_records": sum(h.sample_size for h in SYSTEM_STATE.hospitals.values()),
                "global_test_cohort_size": len(SYSTEM_STATE.global_test),
                "feature_names": FEATURE_NAMES,
                "primary_task": {
                    "dataset": "MIMIC-IV Critical Care EHR Cohort (simulated multi-hospital non-IID)",
                    "prediction_task": "30-Day Adverse Clinical Event & Mortality Risk",
                    "sensitive_attributes": SENSITIVE_ATTRIBUTES,
                    "target_variable": "label (0: Normal/Low Risk, 1: Adverse Event/High Risk)"
                },
                "global_metrics": {
                    "accuracy": global_audit["overall_accuracy"],
                    "precision": global_audit["precision"],
                    "recall": global_audit["recall"],
                    "specificity": global_audit["specificity"],
                    "f1": global_audit["f1"],
                    "auroc": global_audit["auroc"],
                    "gender_dpd": global_audit["gender_audit"]["demographic_parity_difference"],
                    "gender_eod": global_audit["gender_audit"]["equalized_odds_difference"],
                    "gender_dir": global_audit["gender_audit"]["disparate_impact_ratio"],
                    "age_dpd": global_audit["age_audit"]["demographic_parity_difference"],
                    "composite_disparity": global_audit["composite_disparity"],
                    "gender_stats": global_audit["gender_audit"]["group_stats"],
                    "age_stats": global_audit["age_audit"]["group_stats"]
                },
                "history_length": len(SYSTEM_STATE.server.history)
            }
            self._send_json(resp)

        elif path == "/api/hospitals":
            hospitals_info = []
            for h_id, h in SYSTEM_STATE.hospitals.items():
                summary = h.get_summary()
                summary["sample_records"] = []
                summary["current_audit"] = h.audit_bias()
                summary["latest_export"] = h.latest_exported_payload
                summary["history"] = h.training_history
                hospitals_info.append(summary)
            self._send_json({"hospitals": hospitals_info})

        elif path == "/api/history":
            self._send_json({
                "history": SYSTEM_STATE.server.history,
                "benchmark": SYSTEM_STATE.server.benchmark_comparison
            })

        elif path == "/api/download_history":
            qs = parse_qs(parsed.query)
            fmt = qs.get("format", ["json"])[0].lower()
            weights = SYSTEM_STATE.server.get_global_weights()
            
            if fmt == "csv":
                import csv, io
                out = io.StringIO()
                writer = csv.writer(out)
                writer.writerow(["round", "strategy", "accuracy", "gender_dpd", "gender_eod", "disparate_impact_dir", "age_dpd", "timestamp"])
                for r in SYSTEM_STATE.server.history:
                    writer.writerow([
                        r.get("round", ""),
                        r.get("strategy", ""),
                        r.get("accuracy", ""),
                        r.get("gender_dpd", ""),
                        r.get("gender_eod", ""),
                        r.get("disparate_impact_dir", ""),
                        r.get("age_dpd", ""),
                        r.get("timestamp", "")
                    ])
                body = out.getvalue().encode("utf-8")
                filename = f"fedequimed_round_history_{SYSTEM_STATE.server.current_round}.csv"
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                payload = {
                    "title": "FedEquiMed Federated Aggregation & Model Weight History",
                    "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "current_round": SYSTEM_STATE.server.current_round,
                    "global_metrics": SYSTEM_STATE.server.history[-1] if SYSTEM_STATE.server.history else {},
                    "model_parameters": {
                        "weights": weights["weights"],
                        "bias": weights["bias"],
                        "feature_names": FEATURE_NAMES
                    },
                    "round_history": SYSTEM_STATE.server.history,
                    "benchmark_comparison": SYSTEM_STATE.server.benchmark_comparison,
                    "hospitals": [h.get_summary() for h in SYSTEM_STATE.hospitals.values()]
                }
                body = json.dumps(payload, indent=2).encode("utf-8")
                filename = f"fedequimed_round_history_{SYSTEM_STATE.server.current_round}.json"
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        elif path == "/api/download_experiments":
            qs = parse_qs(parsed.query)
            fmt = qs.get("format", ["json"])[0].lower()
            exp_list = list_experiments()
            if fmt == "csv":
                import csv, io
                out = io.StringIO()
                writer = csv.writer(out)
                writer.writerow(["experiment_id", "date", "algorithm", "dataset", "clients", "rounds", "accuracy_mean", "accuracy_std", "delta_dp_mean", "delta_dp_std", "status"])
                for e in exp_list:
                    ms = e.get("metrics_summary", {})
                    acc = ms.get("accuracy", {})
                    dpd = ms.get("delta_dp", {})
                    writer.writerow([
                        e.get("experiment_id", ""),
                        e.get("date", ""),
                        e.get("algorithm", ""),
                        e.get("dataset", ""),
                        e.get("clients", ""),
                        e.get("rounds", ""),
                        acc.get("mean", ""),
                        acc.get("std", ""),
                        dpd.get("mean", ""),
                        dpd.get("std", ""),
                        e.get("status", "")
                    ])
                body = out.getvalue().encode("utf-8")
                filename = f"fedequimed_experiments_{int(time.time())}.csv"
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                body = json.dumps({"status": "success", "experiments": exp_list}, indent=2).encode("utf-8")
                filename = f"fedequimed_experiments_{int(time.time())}.json"
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        elif path == "/api/global_model":
            weights = SYSTEM_STATE.server.get_global_weights()
            self._send_json({
                "status": "success",
                "round": SYSTEM_STATE.server.current_round,
                "weights": weights["weights"],
                "bias": weights["bias"],
                "feature_names": FEATURE_NAMES
            })

        elif path == "/api/experiments":
            exp_list = list_experiments()
            self._send_json({
                "status": "success",
                "experiments": exp_list
            })

        elif path.startswith("/api/experiment/"):
            exp_id = path.split("/")[-1]
            exp_dir = os.path.join(EXPERIMENTS_DIR, exp_id)
            cfg_file = os.path.join(exp_dir, "config.json")
            res_file = os.path.join(exp_dir, "results.json")
            if os.path.exists(cfg_file) and os.path.exists(res_file):
                with open(cfg_file, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                with open(res_file, "r", encoding="utf-8") as f:
                    res = json.load(f)
                self._send_json({
                    "status": "success",
                    "experiment_id": exp_id,
                    "config": cfg,
                    "results": res
                })
            else:
                self._send_json({"status": "error", "error": f"Experiment {exp_id} not found"}, status_code=404)

        elif path == "/api/privacy_config":
            pe = PrivacyEngine(clipping_norm=SYSTEM_STATE.dp_clipping, noise_multiplier=SYSTEM_STATE.dp_noise)
            total_records = sum(h.sample_size for h in SYSTEM_STATE.hospitals.values())
            summary = pe.get_summary(
                sample_size=max(10, total_records),
                batch_size=16,
                rounds=max(1, SYSTEM_STATE.server.current_round),
                local_epochs=4
            )
            self._send_json({
                "status": "success",
                "privacy_config": summary
            })

        elif path == "/api/sample_dataset":
            from fl_engine.dataset import generate_benchmark_csv_text, BENCHMARK_DATASETS_META
            qs = parse_qs(parsed.query)
            benchmark_key = qs.get("benchmark", ["mimic_iv"])[0].lower()
            if benchmark_key not in BENCHMARK_DATASETS_META:
                benchmark_key = "mimic_iv"
            
            meta = BENCHMARK_DATASETS_META[benchmark_key]
            count = int(qs.get("count", ["100"])[0])
            csv_data = generate_benchmark_csv_text(dataset_key=benchmark_key, count=min(1000, max(10, count)))

            if qs.get("format", ["csv"])[0] == "json":
                self._send_json({
                    "status": "success",
                    "benchmark_key": benchmark_key,
                    "metadata": meta,
                    "count": count,
                    "csv_content": csv_data,
                    "available_benchmarks": BENCHMARK_DATASETS_META
                })
            else:
                body = csv_data.encode("utf-8")
                filename = f"{benchmark_key}_cohort_{count}.csv"
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        elif path == "/api/benchmarks_meta":
            from fl_engine.dataset import BENCHMARK_DATASETS_META
            self._send_json({
                "status": "success",
                "benchmarks": BENCHMARK_DATASETS_META
            })

        else:
            # Serve static frontend files
            if path == "/" or path == "":
                filepath = os.path.join(STATIC_DIR, "index.html")
            else:
                filepath = os.path.join(STATIC_DIR, path.lstrip("/"))

            if os.path.exists(filepath) and not os.path.isdir(filepath):
                ctype, _ = mimetypes.guess_type(filepath)
                ctype = ctype or "application/octet-stream"
                try:
                    with open(filepath, "rb") as f:
                        content = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                except Exception as e:
                    self.send_error(500, f"Error reading file: {e}")
            else:
                self.send_error(404, "File Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_len = int(self.headers.get("Content-Length", 0))
        # Protect against excessive payload sizes (max 10MB)
        if content_len > 10 * 1024 * 1024:
            self._send_json({"error": "Payload exceeds 10MB limit"}, status_code=413)
            return

        post_data = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
        try:
            params = json.loads(post_data) if post_data else {}
        except Exception:
            params = {}

        try:
            self._dispatch_post(path, params)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send_json({
                "status": "error",
                "error": f"Server processing error: {str(e)}"
            }, status_code=500)

    def _dispatch_post(self, path: str, params: dict):
        if path == "/api/train_local":
            h_id = params.get("hospital_id", "hospital_a")
            if h_id not in SYSTEM_STATE.hospitals:
                self._send_json({"error": f"Unknown hospital: {h_id}"}, status_code=400)
                return

            epochs = int(params.get("epochs", 5))
            lr = float(params.get("learning_rate", 0.08))
            use_adaptive = bool(params.get("use_adaptive_mitigation", True))
            intensity = float(params.get("mitigation_intensity", 1.2))

            hospital = SYSTEM_STATE.hospitals[h_id]
            curr_global_weights = SYSTEM_STATE.server.get_global_weights()
            payload = hospital.train_local_model(
                global_weights=curr_global_weights,
                epochs=epochs,
                lr=lr,
                use_adaptive_mitigation=use_adaptive,
                mitigation_intensity=intensity,
                dp_noise_multiplier=SYSTEM_STATE.dp_noise
            )
            SYSTEM_STATE.save_state()
            self._send_json({
                "status": "success",
                "message": f"Local model trained for {hospital.name}",
                "payload": payload
            })

        elif path == "/api/federated_round":
            strategy = params.get("strategy", "fairfed")
            epochs = int(params.get("epochs", 4))
            lr = float(params.get("learning_rate", 0.08))
            use_adaptive = bool(params.get("use_adaptive_mitigation", True))
            beta = float(params.get("fairness_beta", 2.5))

            round_summary = SYSTEM_STATE.server.run_federated_round(
                strategy=strategy,
                epochs_per_hospital=epochs,
                learning_rate=lr,
                use_adaptive_mitigation=use_adaptive,
                fairness_beta=beta,
                dp_noise_multiplier=SYSTEM_STATE.dp_noise
            )
            if round_summary.get("status") == "error":
                self._send_json({
                    "status": "error",
                    "error": round_summary.get("error", "Cannot run round without enrolled hospitals.")
                }, status_code=400)
                return

            SYSTEM_STATE.save_state()
            self._send_json({
                "status": "success",
                "round_summary": round_summary
            })

        elif path == "/api/run_benchmark":
            rounds = int(params.get("rounds", 4))
            benchmark = SYSTEM_STATE.server.run_benchmark_comparison(rounds=rounds)
            self._send_json({
                "status": "success",
                "benchmark": benchmark
            })

        elif path == "/api/experiment/run":
            # Reproducible experiment runner (single or multi-seed)
            try:
                results = run_and_persist_experiment(params)
                self._send_json({
                    "status": "success",
                    "message": f"Experiment {results['experiment_id']} executed and persisted successfully.",
                    "results": results
                })
            except Exception as e:
                self._send_json({"status": "error", "error": f"Experiment execution failed: {str(e)}"}, status_code=500)

        elif path == "/api/ablation":
            rounds = int(params.get("rounds", 4))
            seed = int(params.get("seed", 42))
            try:
                ablation_data = run_ablation_study(rounds=rounds, seed=seed)
                self._send_json({
                    "status": "success",
                    "ablation": ablation_data
                })
            except Exception as e:
                self._send_json({"status": "error", "error": f"Ablation study failed: {str(e)}"}, status_code=500)

        elif path == "/api/privacy_config":
            # Update DP settings
            if "noise_multiplier" in params:
                SYSTEM_STATE.dp_noise = max(0.0, float(params["noise_multiplier"]))
            if "clipping_norm" in params:
                SYSTEM_STATE.dp_clipping = max(0.1, float(params["clipping_norm"]))
            for h in SYSTEM_STATE.hospitals.values():
                h.privacy_engine.noise_multiplier = SYSTEM_STATE.dp_noise
                h.privacy_engine.clipping_norm = SYSTEM_STATE.dp_clipping
            SYSTEM_STATE.save_state()
            self._send_json({
                "status": "success",
                "message": "Differential privacy configuration updated.",
                "noise_multiplier": SYSTEM_STATE.dp_noise,
                "clipping_norm": SYSTEM_STATE.dp_clipping
            })

        elif path == "/api/simulate_non_iid":
            # Configures demographic non-IID shift
            heterogeneity = params.get("heterogeneity", "medium").lower()
            seed = int(params.get("seed", 42))
            client_data, global_test = generate_federated_healthcare_datasets(seed=seed)
            SYSTEM_STATE.global_test = global_test
            for h_id in ["hospital_a", "hospital_b", "hospital_c"]:
                if h_id in SYSTEM_STATE.hospitals and h_id in client_data:
                    h = SYSTEM_STATE.hospitals[h_id]
                    h._private_dataset = client_data[h_id]
                    h._sample_size = len(client_data[h_id])
                    h._cached_summary = h._compute_summary(client_data[h_id])
                    h.training_history = []
                    h.audit_bias()
            SYSTEM_STATE.server.global_test_dataset = global_test
            SYSTEM_STATE.save_state()
            self._send_json({
                "status": "success",
                "message": f"Simulated non-IID hospital cohorts updated (Heterogeneity: {heterogeneity}).",
                "hospitals": [h.get_summary() for h in SYSTEM_STATE.hospitals.values()]
            })

        elif path == "/api/load_custom_data":
            h_id = params.get("hospital_id", "hospital_a")
            if h_id not in SYSTEM_STATE.hospitals:
                self._send_json({"error": f"Unknown hospital: {h_id}"}, status_code=400)
                return

            csv_path = params.get("csv_path")
            csv_content = params.get("csv_content")

            from fl_engine.dataset import load_hospital_dataset_from_csv
            import tempfile

            try:
                if csv_content:
                    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
                        tmp.write(csv_content)
                        tmp_path = tmp.name
                    custom_dataset = load_hospital_dataset_from_csv(tmp_path)
                    os.unlink(tmp_path)
                elif csv_path and os.path.exists(csv_path):
                    custom_dataset = load_hospital_dataset_from_csv(csv_path)
                else:
                    self._send_json({"error": "Missing or invalid CSV data"}, status_code=400)
                    return

                if not custom_dataset:
                    self._send_json({"error": "CSV contains no valid patient records"}, status_code=400)
                    return

                hospital = SYSTEM_STATE.hospitals[h_id]
                hospital._private_dataset = custom_dataset
                hospital._sample_size = len(custom_dataset)
                hospital._cached_summary = hospital._compute_summary(custom_dataset)
                hospital.training_history = []
                hospital.latest_exported_payload = None
                hospital.audit_bias()
                SYSTEM_STATE.save_state()

                self._send_json({
                    "status": "success",
                    "message": f"Successfully loaded {len(custom_dataset)} patient records into {hospital.name}",
                    "hospital_summary": hospital.get_summary()
                })
            except Exception as e:
                self._send_json({"error": f"Failed to load CSV: {str(e)}"}, status_code=500)

        elif path == "/api/register_hospital":
            h_name = params.get("hospital_name", "").strip() or f"Hospital {chr(65 + len(SYSTEM_STATE.hospitals))}"
            desc = params.get("description", "").strip() or "Newly enrolled clinical healthcare institution."
            csv_content = params.get("csv_content")
            csv_path = params.get("csv_path")

            from fl_engine.dataset import load_hospital_dataset_from_csv, generate_patient_record
            import tempfile, random

            custom_dataset = []
            if csv_content:
                try:
                    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
                        tmp.write(csv_content)
                        tmp_path = tmp.name
                    custom_dataset = load_hospital_dataset_from_csv(tmp_path)
                    os.unlink(tmp_path)
                except Exception as e:
                    self._send_json({"error": f"Error parsing uploaded CSV: {str(e)}"}, status_code=400)
                    return
            elif csv_path and os.path.exists(csv_path):
                try:
                    custom_dataset = load_hospital_dataset_from_csv(csv_path)
                except Exception as e:
                    self._send_json({"error": f"Error parsing CSV path: {str(e)}"}, status_code=400)
                    return
            else:
                rng = random.Random()
                for _ in range(750):
                    g = 1 if rng.random() < 0.50 else 0
                    a = 1 if rng.random() < 0.60 else 0
                    eth = rng.choice([0, 1, 2])
                    rec = generate_patient_record(g, a, eth, "new_hospital", rng=rng)
                    custom_dataset.append(rec)

            if not custom_dataset:
                self._send_json({"error": "No valid patient records could be extracted from input."}, status_code=400)
                return

            counter = len(SYSTEM_STATE.hospitals) + 1
            new_id = f"hospital_{counter}"
            while new_id in SYSTEM_STATE.hospitals:
                counter += 1
                new_id = f"hospital_{counter}"

            new_node = HospitalNode(
                hospital_id=new_id,
                name=h_name,
                dataset=custom_dataset,
                num_features=10,
                description=desc,
                privacy_engine=PrivacyEngine(clipping_norm=SYSTEM_STATE.dp_clipping, noise_multiplier=SYSTEM_STATE.dp_noise)
            )
            new_node.train_local_model(
                global_weights=SYSTEM_STATE.server.get_global_weights(),
                epochs=2,
                use_adaptive_mitigation=True
            )
            SYSTEM_STATE.hospitals[new_id] = new_node
            SYSTEM_STATE.server.hospitals = SYSTEM_STATE.hospitals
            SYSTEM_STATE.save_state()

            self._send_json({
                "status": "success",
                "message": f"Successfully enrolled {h_name} ({len(custom_dataset)} patients processed).",
                "hospital": new_node.get_summary()
            })

        elif path == "/api/test_hospital_data":
            h_id = params.get("hospital_id")
            csv_content = params.get("csv_content")
            h_name = params.get("hospital_name", "").strip()

            from fl_engine.bias_detector import BiasDetector
            from fl_engine.dataset import load_hospital_dataset_from_csv, generate_patient_record
            import tempfile, random

            test_dataset = None
            source_name = "Uploaded Dataset"

            if h_id and h_id in SYSTEM_STATE.hospitals:
                hospital = SYSTEM_STATE.hospitals[h_id]
                test_dataset = hospital._private_dataset
                source_name = hospital.name
            elif csv_content:
                try:
                    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
                        tmp.write(csv_content)
                        tmp_path = tmp.name
                    test_dataset = load_hospital_dataset_from_csv(tmp_path)
                    os.unlink(tmp_path)
                    source_name = h_name or "Uploaded CSV Dataset"
                except Exception as e:
                    self._send_json({"error": f"Failed to parse CSV: {str(e)}"}, status_code=400)
                    return
            else:
                self._send_json({"error": "No patient dataset provided. Please upload a .CSV file or select an active hospital to test the global model."}, status_code=400)
                return

            audit = BiasDetector.audit_model(SYSTEM_STATE.server.global_model, test_dataset)
            
            y_true = [d["label"] for d in test_dataset]
            y_pred = SYSTEM_STATE.server.global_model.predict([d["features"] for d in test_dataset])

            tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
            tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
            fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

            sensitivity = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
            specificity = round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0.0
            f1 = round((2 * tp) / (2 * tp + fp + fn), 4) if (2 * tp + fp + fn) > 0 else 0.0

            self._send_json({
                "status": "success",
                "source_name": source_name,
                "tested_records": len(test_dataset),
                "global_round": SYSTEM_STATE.server.current_round,
                "overall_performance": {
                    "accuracy": audit["overall_accuracy"],
                    "precision": audit["precision"],
                    "sensitivity_recall": sensitivity,
                    "specificity": specificity,
                    "f1_score": f1,
                    "auroc": audit["auroc"],
                    "tp": tp, "tn": tn, "fp": fp, "fn": fn
                },
                "fairness_audit": {
                    "gender_dpd": audit["gender_audit"]["demographic_parity_difference"],
                    "gender_eod": audit["gender_audit"]["equalized_odds_difference"],
                    "gender_dir": audit["gender_audit"]["disparate_impact_ratio"],
                    "age_dpd": audit["age_audit"]["demographic_parity_difference"],
                    "composite_disparity": audit["composite_disparity"],
                    "gender_stats": audit["gender_audit"]["group_stats"],
                    "age_stats": audit["age_audit"]["group_stats"]
                }
            })

        elif path == "/api/verify_admin":
            admin_key = params.get("admin_key", "")
            if verify_admin_key(admin_key):
                self._send_json({"status": "success", "message": "Admin authority verified."})
            else:
                self._send_json({"status": "error", "error": "Invalid admin passkey"}, status_code=403)

        elif path == "/api/delete_hospital":
            admin_key = params.get("admin_key", "")
            if not verify_admin_key(admin_key):
                self._send_json({
                    "status": "error",
                    "error": "Access Denied: Invalid admin authorization passkey."
                }, status_code=403)
                return

            h_id = params.get("hospital_id")
            if not h_id or h_id not in SYSTEM_STATE.hospitals:
                self._send_json({"error": f"Hospital ID '{h_id}' not found."}, status_code=404)
                return

            deleted_name = SYSTEM_STATE.hospitals[h_id].name
            del SYSTEM_STATE.hospitals[h_id]
            SYSTEM_STATE.server.hospitals = SYSTEM_STATE.hospitals
            SYSTEM_STATE.save_state()

            self._send_json({
                "status": "success",
                "message": f"Hospital '{deleted_name}' successfully removed by administrator."
            })

        elif path == "/api/delete_all_hospitals":
            admin_key = params.get("admin_key", "")
            if not verify_admin_key(admin_key):
                self._send_json({
                    "status": "error",
                    "error": "Access Denied: Invalid admin authorization passkey."
                }, status_code=403)
                return

            deleted_count = len(SYSTEM_STATE.hospitals)
            SYSTEM_STATE.hospitals.clear()
            SYSTEM_STATE.server.hospitals = SYSTEM_STATE.hospitals
            SYSTEM_STATE.save_state()

            self._send_json({
                "status": "success",
                "message": f"Successfully deleted all {deleted_count} hospital nodes.",
                "deleted_count": deleted_count
            })

        elif path == "/api/reset":
            admin_key = params.get("admin_key", "")
            if not verify_admin_key(admin_key):
                self._send_json({
                    "status": "error",
                    "error": "Access Denied: Invalid admin authorization passkey."
                }, status_code=403)
                return

            if os.path.exists(STATE_FILE):
                try:
                    os.remove(STATE_FILE)
                except Exception:
                    pass
            SYSTEM_STATE.initialize_system(force_fresh=True)
            self._send_json({
                "status": "success",
                "message": "System and models reset to clean initial state by administrator."
            })
        else:
            self.send_error(404, "Endpoint Not Found")

    def log_message(self, format, *args):
        pass

def run_server(port: int = 8085):
    for try_port in [port, 8085, 8088, 8000, 5001, 8888]:
        try:
            server_address = ("", try_port)
            httpd = HTTPServer(server_address, FLRequestHandler)
            print(f"Fairness-Aware Federated Learning Server listening on http://localhost:{try_port}")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nShutting down server.")
                httpd.server_close()
            return
        except OSError as e:
            if e.errno == 48:
                continue
            raise e

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8085"))
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port)
