
import html
import time
import numpy as np
from typing import Dict, Any, Iterable, List

from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QWidget, QGroupBox, QComboBox, QPushButton, QTextEdit,
    QPlainTextEdit, QCheckBox, QToolTip, QRubberBand
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QEvent, QRect, QSize

import pyqtgraph as pg

from scipy.spatial import cKDTree

from vmeans.ai_client import MODEL_PRESETS, ask_ai
from vmeans.colors import get_colors_for_centers

class AIChatWorker(QThread):
    response_ready = pyqtSignal(str)
    error_ready = pyqtSignal(str)

    def __init__(self, provider: str, model: str, system_prompt: str,
                 user_prompt: str, parent=None):
        super().__init__(parent)
        self.provider = provider
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

    def run(self):
        try:
            response = ask_ai(
                self.provider,
                self.model,
                self.system_prompt,
                self.user_prompt,
            )
            self.response_ready.emit(response)
        except Exception as exc:
            self.error_ready.emit(str(exc))


class HoverAIMixin:
    def _build_ai_panel(self):
        panel = QGroupBox("AI Feedback")
        panel.setMinimumWidth(360)
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)

        provider_row = QHBoxLayout()
        provider_row.addWidget(QLabel("Provider"))
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(MODEL_PRESETS.keys())
        provider_row.addWidget(self.provider_combo, 1)
        layout.addLayout(provider_row)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model"))
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        model_row.addWidget(self.model_combo, 1)
        layout.addLayout(model_row)

        self.provider_combo.currentTextChanged.connect(self._on_provider_changed)
        self._on_provider_changed(self.provider_combo.currentText())

        self.selected_count_label = QLabel("Selected points: 0")
        self.selected_count_label.setStyleSheet("font-weight: bold; color: #444;")
        layout.addWidget(self.selected_count_label)

        self.include_region_summary_cb = QCheckBox("Use cluster/region summary")
        self.include_region_summary_cb.setChecked(True)
        layout.addWidget(self.include_region_summary_cb)

        self.include_selected_cb = QCheckBox("Use selected point rows")
        self.include_selected_cb.setChecked(True)
        layout.addWidget(self.include_selected_cb)

        self.include_full_rows_cb = QCheckBox("Use raw table rows")
        self.include_full_rows_cb.setChecked(True)
        self.include_full_rows_cb.setToolTip(
            "For cloud providers this may send CSV rows to an external service."
        )
        layout.addWidget(self.include_full_rows_cb)

        quick_row = QHBoxLayout()
        summarize_btn = QPushButton("Ask Summary")
        summarize_btn.setToolTip("Ask AI to summarize all visible clusters.")
        summarize_btn.clicked.connect(
            lambda: self._ask_prefilled("Summarize the visible clusters and any obvious outliers.")
        )
        selected_btn = QPushButton("Ask Selection")
        selected_btn.setToolTip("Ask AI to explain the currently selected points.")
        selected_btn.clicked.connect(
            lambda: self._ask_prefilled("Explain the selected points compared with the whole dataset.")
        )
        anomaly_btn = QPushButton("Find Anomalies")
        anomaly_btn.setToolTip("Ask AI to look for unusual clusters or outlier points.")
        anomaly_btn.clicked.connect(
            lambda: self._ask_prefilled("Which regions or points look unusual, and why?")
        )
        self.quick_buttons = [summarize_btn, selected_btn, anomaly_btn]
        quick_row.addWidget(summarize_btn)
        quick_row.addWidget(selected_btn)
        quick_row.addWidget(anomaly_btn)
        layout.addLayout(quick_row)

        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setMinimumHeight(260)
        self.chat_history.setPlaceholderText("Ask about this clustering result...")
        layout.addWidget(self.chat_history, 1)

        self.question_input = QPlainTextEdit()
        self.question_input.setPlaceholderText(
            "Ask AI about this result, selected points, or the source CSV..."
        )
        self.question_input.setMaximumHeight(90)
        layout.addWidget(self.question_input)

        button_row = QHBoxLayout()
        self.send_btn = QPushButton("Ask AI")
        self.send_btn.clicked.connect(self._send_ai_question)
        clear_selection_btn = QPushButton("Clear Selection")
        clear_selection_btn.clicked.connect(self._clear_selection)
        clear_chat_btn = QPushButton("Clear Chat")
        clear_chat_btn.clicked.connect(self._clear_chat)
        button_row.addWidget(self.send_btn)
        button_row.addWidget(clear_selection_btn)
        button_row.addWidget(clear_chat_btn)
        layout.addLayout(button_row)

        self.ai_status_label = QLabel(
            "Ready. Tip: llama3.2 is faster; qwen2.5:14b-instruct is stronger but slower."
        )
        self.ai_status_label.setStyleSheet("color: #666; font-size: 11px;")
        self.ai_status_label.setWordWrap(True)
        layout.addWidget(self.ai_status_label)
        return panel


    def _on_provider_changed(self, provider: str):
        models = MODEL_PRESETS.get(provider, [])
        self.model_combo.clear()
        self.model_combo.addItems(models)
        if models:
            self.model_combo.setCurrentText(models[0])
        self.model_combo.setEditable(True)

        if hasattr(self, "include_full_rows_cb"):
            self.include_full_rows_cb.setChecked(provider == "Ollama")
        if hasattr(self, "ai_status_label"):
            if provider == "Ollama":
                self.ai_status_label.setText("Ready. Ollama context stays local.")
            else:
                self.ai_status_label.setText(
                    "Cloud provider selected. Raw table rows are off by default unless you enable them."
                )


    def _ask_prefilled(self, question: str):
        self.question_input.setPlainText(question)
        self._send_ai_question()


    def _append_chat(self, role: str, text: str, record: bool = True):
        if record and role in {"You", "AI"}:
            self.chat_turns.append((role, text))
            max_turns = self.MAX_CHAT_CONTEXT_TURNS * 2
            if len(self.chat_turns) > max_turns:
                self.chat_turns = self.chat_turns[-max_turns:]

        color = "#0b6bcb" if role == "You" else "#176b2c"
        safe = html.escape(text).replace("\n", "<br>")
        self.chat_history.append(
            f'<p style="margin: 6px 0;"><b style="color:{color};">{role}:</b><br>{safe}</p>'
        )


    def _clear_chat(self):
        self.chat_history.clear()
        self.chat_turns.clear()


    def _send_ai_question(self):
        if self.ai_worker is not None and self.ai_worker.isRunning():
            self.ai_status_label.setText("AI is still answering. Please wait for the current response.")
            return

        question = self.question_input.toPlainText().strip()
        if not question:
            return

        provider = self.provider_combo.currentText().strip()
        model = self.model_combo.currentText().strip()
        if not model:
            self._append_chat("AI", "Please choose or type a model name first.")
            return

        context_text = self._build_ai_context()
        system_prompt = (
            "You are an AI feedback assistant embedded in the Visible Silhouettes "
            "Algorithm visualization. Use only the supplied data context. The user "
            "may ask about the whole visible clustering result or about selected "
            "points. If no points are selected, answer from the whole plotted "
            "dataset. If points are selected, prioritize those points and compare "
            "them with their own region and the full dataset. Selected points are "
            "user focus points, not automatically outliers. Use recent chat turns "
            "only to resolve follow-up references; the current data and current "
            "selection are authoritative. Trust the explicit diagnostic_conclusion "
            "fields; do not invert boolean diagnostic flags or recalculate thresholds "
            "from rounded values. Be concise, explain uncertainty, and distinguish "
            "observations from speculation."
        )
        user_prompt = (
            f"{context_text}\n\n"
            "User question:\n"
            f"{question}"
        )

        self._append_chat("You", question)
        self.question_input.clear()
        self._set_ai_controls_enabled(False)
        self.ai_status_label.setText(f"Asking {provider} / {model}...")

        self.ai_worker = AIChatWorker(provider, model, system_prompt, user_prompt, self)
        self.ai_worker.response_ready.connect(self._on_ai_response)
        self.ai_worker.error_ready.connect(self._on_ai_error)
        self.ai_worker.finished.connect(self._on_ai_finished)
        self.ai_worker.start()


    def _on_ai_response(self, response: str):
        self._append_chat("AI", response)
        self.ai_status_label.setText("Ready.")


    def _on_ai_error(self, message: str):
        self._append_chat("AI", f"Error: {message}")
        self.ai_status_label.setText("AI request failed.")


    def _on_ai_finished(self):
        self._set_ai_controls_enabled(True)
        self.ai_worker = None


    def _set_ai_controls_enabled(self, enabled: bool):
        self.send_btn.setEnabled(enabled)
        self.provider_combo.setEnabled(enabled)
        self.model_combo.setEnabled(enabled)
        for button in getattr(self, "quick_buttons", []):
            button.setEnabled(enabled)


    def _build_ai_context(self) -> str:
        lines: List[str] = []
        lines.append("VISIBLE SILHOUETTES CLUSTERING CONTEXT")
        lines.append(
            "Project: Visible Silhouettes Algorithm. This view shows the final or "
            "nearest final 2D clustering result produced by the step-animation UI."
        )
        lines.append(
            "Task semantics: analyze the supplied plotted coordinates, cluster/region "
            "assignments, optional source rows, and optional user-selected points. "
            "Do not rerun the clustering algorithm."
        )
        lines.append(
            "Selection semantics: selected points are user-highlighted focus points. "
            "They are not automatically anomalies unless the statistics support that."
        )
        lines.append(f"Frame: {self.frame_name}")
        lines.append(f"Dataset/source: {self.dataset_label}")
        lines.append(f"Number of plotted points: {len(self.points)}")
        lines.append(f"Number of regions/clusters shown: {self.data.get('n_regions', 0)}")
        lines.append(f"X axis: {self.x_label}")
        lines.append(f"Y axis: {self.y_label}")

        if self.analysis_settings:
            lines.append("")
            lines.append("ALGORITHM / UI SETTINGS")
            for key, value in self.analysis_settings.items():
                lines.append(f"- {key}: {value}")

        lines.append("")
        lines.append("PLOT GEOMETRY")
        lines.extend(self._plot_geometry_lines())
        lines.append("")
        lines.append("DISTANCE DIAGNOSTIC GUIDE")
        lines.extend(self._distance_diagnostic_guide_lines())

        if self.include_selected_cb.isChecked():
            lines.append("")
            lines.append("CURRENT USER SELECTION - HIGH PRIORITY")
            lines.extend(self._selected_points_lines())

        if self.include_region_summary_cb.isChecked():
            lines.append("")
            lines.append("REGION SUMMARY")
            lines.extend(self._region_summary_lines())
            lines.append("")
            lines.append("DISTANCE-BASED CANDIDATE OUTLIERS")
            lines.extend(self._candidate_outlier_lines())

        lines.append("")
        lines.append("SOURCE TABLE SUMMARY")
        lines.extend(self._source_table_summary_lines())

        lines.append("")
        lines.append("RECENT CHATBOX CONTEXT")
        lines.extend(self._recent_chat_lines())

        if self.include_full_rows_cb.isChecked():
            lines.append("")
            lines.append("RAW ROWS / POINT TABLE")
            lines.append(self._rows_to_text(range(len(self.points)), self.MAX_FULL_ROWS))

        return self._clip_text("\n".join(lines), self.MAX_PROMPT_CHARS)


    def _region_indices(self, rid: int) -> np.ndarray:
        return np.where(self.region_ids == rid)[0]


    def _region_label(self, rid: int) -> str:
        if 0 <= rid < len(self.region_labels):
            return self.region_labels[rid]
        return "Unassigned"


    def _region_geometry(self, rid: int) -> Dict[str, Any] | None:
        indices = self._region_indices(rid)
        if len(indices) == 0:
            return None

        xs = self.plot_x[indices]
        ys = self.plot_y[indices]
        center = np.array([float(np.mean(xs)), float(np.mean(ys))])
        coords = np.column_stack([xs, ys])
        distances = np.linalg.norm(coords - center, axis=1)
        return {
            "indices": indices,
            "label": self._region_label(rid),
            "center": center,
            "x_min": float(np.min(xs)),
            "x_max": float(np.max(xs)),
            "y_min": float(np.min(ys)),
            "y_max": float(np.max(ys)),
            "x_std": float(np.std(xs)),
            "y_std": float(np.std(ys)),
            "radial_mean": float(np.mean(distances)) if len(distances) else 0.0,
            "radial_p95": float(np.percentile(distances, 95)) if len(distances) else 0.0,
            "radial_max": float(np.max(distances)) if len(distances) else 0.0,
            "distances": distances,
        }


    def _all_region_geometries(self) -> Dict[int, Dict[str, Any]]:
        geometries: Dict[int, Dict[str, Any]] = {}
        for rid in sorted(set(int(v) for v in self.region_ids)):
            geometry = self._region_geometry(rid)
            if geometry is not None:
                geometries[rid] = geometry
        return geometries


    def _distance_percentile(self, distance: float, distances: np.ndarray) -> float:
        if len(distances) == 0:
            return 0.0
        return float(np.mean(distances <= distance) * 100.0)


    def _plot_geometry_lines(self) -> List[str]:
        center = np.array([float(np.mean(self.plot_x)), float(np.mean(self.plot_y))])
        coords = np.column_stack([self.plot_x, self.plot_y])
        global_distances = np.linalg.norm(coords - center, axis=1)
        return [
            f"Global centroid in plot coordinates: ({center[0]:.3f}, {center[1]:.3f})",
            (
                f"Global x range: [{float(np.min(self.plot_x)):.3f}, "
                f"{float(np.max(self.plot_x)):.3f}]"
            ),
            (
                f"Global y range: [{float(np.min(self.plot_y)):.3f}, "
                f"{float(np.max(self.plot_y)):.3f}]"
            ),
            (
                f"Distance from global centroid: mean={float(np.mean(global_distances)):.3f}, "
                f"p95={float(np.percentile(global_distances, 95)):.3f}, "
                f"max={float(np.max(global_distances)):.3f}"
            ),
        ]


    def _distance_diagnostic_guide_lines(self) -> List[str]:
        return [
            "within-region_distance_percentile means the percentage of points in the same region "
            "that are no farther from that region center. Higher means farther from its own region center; "
            "100 means this point is the farthest or tied for farthest in its region.",
            "distance_over_region_p95 compares a point's center distance with the 95th-percentile radius "
            "of its own region. Values above 1.0 mean the point is beyond the region p95 radius and should "
            "be treated as a distance-based candidate outlier, not as automatically normal.",
            "Use diagnostic_conclusion and plain_language_summary first. The numeric values are supporting evidence. "
            "Do not say a point is below p95 when beyond_region_p95=True, and do not say it is normal when "
            "distance_candidate_outlier=True.",
            "These distance diagnostics are descriptive heuristics; combine them with the plot and source data."
        ]


    def _region_summary_lines(self) -> List[str]:
        lines: List[str] = []
        geometries = self._all_region_geometries()
        centers = {rid: geometry["center"] for rid, geometry in geometries.items()}
        for rid, geometry in geometries.items():
            indices = geometry["indices"]
            label = geometry["label"]
            selected = sum(1 for i in indices if int(i) in self.selected_indices)
            nearest = None
            if len(centers) > 1:
                other_distances = [
                    float(np.linalg.norm(geometry["center"] - other_center))
                    for other_rid, other_center in centers.items()
                    if other_rid != rid
                ]
                nearest = min(other_distances) if other_distances else None
            lines.append(
                f"- {label}: n={len(indices)}, selected={selected}, "
                f"center=({geometry['center'][0]:.2f}, {geometry['center'][1]:.2f}), "
                f"{self.x_label} range=[{geometry['x_min']:.2f}, {geometry['x_max']:.2f}], "
                f"{self.y_label} range=[{geometry['y_min']:.2f}, {geometry['y_max']:.2f}], "
                f"std=({geometry['x_std']:.2f}, {geometry['y_std']:.2f}), "
                f"radial mean={geometry['radial_mean']:.2f}, "
                f"radial p95={geometry['radial_p95']:.2f}, "
                f"radial max={geometry['radial_max']:.2f}"
                + (f", nearest region-center distance={nearest:.2f}" if nearest is not None else "")
            )
        return lines or ["- No region summary available."]


    def _point_diagnostic(self, idx: int, geometries: Dict[int, Dict[str, Any]] | None = None) -> str:
        geometries = geometries or self._all_region_geometries()
        rid = int(self.region_ids[idx]) if idx < len(self.region_ids) else -1
        geometry = geometries.get(rid)
        if geometry is None:
            return (
                f"index={idx}, region={self._region_name(idx)}, "
                f"{self.x_label}={self.plot_x[idx]:.3f}, {self.y_label}={self.plot_y[idx]:.3f}"
            )

        point = np.array([float(self.plot_x[idx]), float(self.plot_y[idx])])
        distance = float(np.linalg.norm(point - geometry["center"]))
        percentile = self._distance_percentile(distance, geometry["distances"])
        ratio = distance / geometry["radial_p95"] if geometry["radial_p95"] > 1e-12 else 0.0
        beyond_p95 = ratio > 1.0
        farthest_or_tied = percentile >= 99.5
        distance_candidate = beyond_p95 or percentile >= 95.0
        if distance_candidate:
            interpretation = "distance-based candidate outlier within its assigned region"
            plain_language_summary = (
                "This point is unusually far from its assigned region center by the distance heuristic."
            )
            diagnostic_conclusion = "YES_DISTANCE_CANDIDATE_OUTLIER"
        else:
            interpretation = "not a distance-based outlier candidate within its assigned region"
            plain_language_summary = (
                "This point is not unusually far from its assigned region center by the distance heuristic."
            )
            diagnostic_conclusion = "NO_DISTANCE_CANDIDATE_OUTLIER"
        return (
            f"index={idx}, region={geometry['label']}, "
            f"diagnostic_conclusion={diagnostic_conclusion}, "
            f"plain_language_summary='{plain_language_summary}', "
            f"{self.x_label}={point[0]:.3f}, {self.y_label}={point[1]:.3f}, "
            f"distance_to_region_center={distance:.3f}, "
            f"within-region_distance_percentile={percentile:.1f}, "
            f"distance_over_region_p95={ratio:.2f}, "
            f"beyond_region_p95={beyond_p95}, "
            f"farthest_or_tied_in_region={farthest_or_tied}, "
            f"distance_candidate_outlier={distance_candidate}, "
            f"interpretation='{interpretation}'"
        )


    def _candidate_outlier_lines(self, limit: int = 10) -> List[str]:
        geometries = self._all_region_geometries()
        candidates = []
        for rid, geometry in geometries.items():
            p95 = geometry["radial_p95"] if geometry["radial_p95"] > 1e-12 else 1.0
            for local_pos, idx in enumerate(geometry["indices"]):
                distance = float(geometry["distances"][local_pos])
                candidates.append((distance / p95, distance, int(idx)))

        if not candidates:
            return ["No distance-based outlier diagnostics available."]

        candidates.sort(reverse=True)
        lines = [
            "Heuristic only: points far from their own region centroid. "
            "These are candidates for discussion, not guaranteed anomalies. "
            "Percentiles near 100 or distance_over_region_p95 above 1.0 mean the point is farther "
            "from its own region center than most same-region points."
        ]
        for _, _, idx in candidates[:limit]:
            lines.append(f"- {self._point_diagnostic(idx, geometries)}")
        return lines


    def _source_table_summary_lines(self) -> List[str]:
        if self.original_df is None:
            return ["No original CSV table was attached; using plotted x/y coordinates only."]

        lines = [
            f"Rows: {len(self.original_df)}",
            f"Columns: {', '.join(map(str, self.original_df.columns))}",
        ]
        try:
            numeric_cols = list(self.original_df.select_dtypes(include=[np.number]).columns)
        except Exception:
            numeric_cols = []

        if numeric_cols:
            lines.append("Numeric column summaries:")
            for col in numeric_cols[:12]:
                series = self.original_df[col].dropna()
                if len(series) == 0:
                    continue
                lines.append(
                    f"- {col}: mean={series.mean():.2f}, "
                    f"min={series.min():.2f}, max={series.max():.2f}"
                )
            if len(numeric_cols) > 12:
                lines.append(f"- {len(numeric_cols) - 12} more numeric columns omitted.")
        return lines


    def _selected_points_lines(self) -> List[str]:
        if not self.selected_indices:
            return [
                "No points selected. The answer should analyze the full plotted dataset, "
                "region summaries, source table summary, and any distance-based outlier candidates."
            ]

        geometries = self._all_region_geometries()
        lines = [
            "Answer rule for selected-point questions: if diagnostic_conclusion is "
            "YES_DISTANCE_CANDIDATE_OUTLIER, answer that the selected point is unusual "
            "by the distance heuristic; if it is NO_DISTANCE_CANDIDATE_OUTLIER, answer "
            "that it is not unusual by the distance heuristic.",
            f"Selected count: {len(self.selected_indices)}",
            f"Selected region counts: {self._selection_region_counts()}",
            "Interpret selected-point distance diagnostics using the DISTANCE DIAGNOSTIC GUIDE above.",
            "Selected point diagnostics relative to their assigned regions:",
        ]
        lines.extend(
            f"- {self._point_diagnostic(idx, geometries)}"
            for idx in sorted(self.selected_indices)[:self.MAX_SELECTED_ROWS]
        )
        if len(self.selected_indices) > self.MAX_SELECTED_ROWS:
            lines.append(f"- Diagnostics truncated after {self.MAX_SELECTED_ROWS} selected points.")
        lines.extend([
            "",
            "Selected point rows:",
            self._rows_to_text(sorted(self.selected_indices), self.MAX_SELECTED_ROWS),
        ])
        return lines


    def _recent_chat_lines(self) -> List[str]:
        if not self.chat_turns:
            return ["No previous chat turns in this Hover Details session."]

        lines = [
            "Recent chat turns from this Hover Details session. Use these for follow-up "
            "references, but prefer the current dataset, current selection, and current question."
        ]
        for role, text in self.chat_turns[-self.MAX_CHAT_CONTEXT_TURNS * 2:]:
            compact = " ".join(str(text).split())
            lines.append(f"{role}: {self._clip_text(compact, 1200)}")
        return lines


    def _rows_to_text(self, indices: Iterable[int], max_rows: int) -> str:
        indices = [int(i) for i in indices]
        truncated = len(indices) > max_rows
        indices = indices[:max_rows]

        if self.original_df is not None and len(self.original_df) == len(self.points):
            table = self.original_df.iloc[indices].copy()
            region_col = "_cluster_region"
            while region_col in table.columns:
                region_col = f"_{region_col}"
            table.insert(0, region_col, [self._region_name(i) for i in indices])
            text = table.to_csv(index=True)
        else:
            rows = ["index,region,x,y"]
            for i in indices:
                rows.append(f"{i},{self._region_name(i)},{self.plot_x[i]:.6g},{self.plot_y[i]:.6g}")
            text = "\n".join(rows)

        if truncated:
            text += f"\n... truncated after {max_rows} rows."
        return self._clip_text(text, 18000)


    def _clip_text(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n... [context truncated]"
