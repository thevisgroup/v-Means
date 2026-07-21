"""Finviz-style cluster table support for the hover scatter dialog."""

from __future__ import annotations

from typing import Any

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QBrush, QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class HoverClusterTableMixin:
    """Adds selectable compact-card and persistent cluster-table modes."""

    HOVER_MODE_CARD = "card"
    HOVER_MODE_TABLE = "table"

    def _init_hover_detail_state(self) -> None:
        self.hover_detail_mode = self.HOVER_MODE_CARD
        self._cluster_table_anchor_idx = None
        self._detail_column_cache = None
        self._cluster_table_rendered_region_id = None
        self._cluster_table_members = []
        self._cluster_table_row_by_point = {}
        self._cluster_table_active_idx = None
        self._cluster_table_selected_indices = frozenset()
        self._cluster_table_summary_base = []
        self._legend_hover_region_id = None

    @staticmethod
    def _normalize_detail_column(value: object) -> str:
        return " ".join(str(value).replace("\n", " ").split()).lower()

    def _detail_columns(self) -> dict[str, Any]:
        if self._detail_column_cache is not None:
            return self._detail_column_cache

        result = {
            "code": None,
            "description": None,
            "admissions": None,
            "bed_days": None,
        }
        frame = getattr(self, "original_df", None)
        if frame is None:
            self._detail_column_cache = result
            return result

        columns = list(frame.columns)
        normalized = {self._normalize_detail_column(column): column for column in columns}
        result["code"] = normalized.get("primary diagnosis: summary code and description")
        if result["code"] is not None:
            code_position = columns.index(result["code"])
            if code_position + 1 < len(columns):
                result["description"] = columns[code_position + 1]
        if result["description"] is None:
            result["description"] = normalized.get("primary diagnosis description")
        result["admissions"] = normalized.get("finished admission episodes")
        result["bed_days"] = normalized.get("fce bed days")
        self._detail_column_cache = result
        return result

    def _row_value(self, idx: int, column: Any, default: Any = None) -> Any:
        frame = getattr(self, "original_df", None)
        if frame is None or column is None or not (0 <= idx < len(frame)):
            return default
        value = frame.iloc[idx][column]
        try:
            if bool(np.isnan(value)):
                return default
        except (TypeError, ValueError):
            pass
        return value

    @staticmethod
    def _numeric_value(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if np.isfinite(number) else None

    @staticmethod
    def _table_value(value: Any, decimals: int = 1) -> str:
        number = HoverClusterTableMixin._numeric_value(value)
        if number is None:
            return "—" if value is None else str(value)
        if abs(number - round(number)) < 1e-9:
            return f"{int(round(number)):,}"
        return f"{number:,.{decimals}f}"

    def _diagnosis_parts(self, idx: int) -> tuple[str, str]:
        columns = self._detail_columns()
        code = self._row_value(idx, columns["code"], "")
        description = self._row_value(idx, columns["description"], "")
        return str(code).strip(), str(description).strip()

    def _diagnosis_label(self, idx: int) -> str | None:
        code, description = self._diagnosis_parts(idx)
        if code and description:
            return f"{code} — {description}"
        return code or description or None

    def _cluster_preview_index(self, region_id: int) -> int | None:
        """Choose a stable representative row when a cluster label is hovered."""
        members = [int(value) for value in np.where(self.region_ids == region_id)[0]]
        if not members:
            return None

        def admissions_key(member: int) -> tuple[bool, float]:
            admissions = self._numeric_value(self._cluster_record(member)["admissions"])
            return admissions is not None, admissions or 0.0

        return max(members, key=admissions_key)

    def _cluster_record(self, idx: int) -> dict[str, Any]:
        code, description = self._diagnosis_parts(idx)
        columns = self._detail_columns()
        if getattr(self, "has_original_coords", False):
            x_column, y_column = self.original_xy_cols
            x_value = self._row_value(idx, x_column)
            y_value = self._row_value(idx, y_column)
        else:
            x_column, y_column = self.x_label, self.y_label
            x_value = self.plot_x[idx]
            y_value = self.plot_y[idx]
        return {
            "idx": idx,
            "code": code or f"#{idx}",
            "description": description or f"Point {idx}",
            "x_header": str(x_column),
            "y_header": str(y_column),
            "x": x_value,
            "y": y_value,
            "admissions": self._row_value(idx, columns["admissions"]),
            "bed_days": self._row_value(idx, columns["bed_days"]),
        }

    @staticmethod
    def _short_metric_header(header: str) -> str:
        normalized = HoverClusterTableMixin._normalize_detail_column(header)
        if "mean time waited" in normalized:
            return "Wait (days)"
        if "mean age" in normalized:
            return "Age (years)"
        return " ".join(str(header).replace("\n", " ").split())

    def _build_cluster_table_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("clusterTablePanel")
        panel.setMinimumWidth(500)
        panel.setStyleSheet(
            """
            QWidget#clusterTablePanel {
                background-color: white;
            }
            QTableWidget {
                color: #333333;
                background-color: white;
                alternate-background-color: #f5f5f5;
                border: 1px solid #cccccc;
                border-radius: 4px;
                gridline-color: #dddddd;
                selection-background-color: #fff3e0;
                selection-color: #333333;
            }
            QHeaderView::section {
                color: #333333;
                background-color: #e0e0e0;
                border: none;
                border-right: 1px solid #cccccc;
                border-bottom: 1px solid #bdbdbd;
                padding: 7px 5px;
                font-weight: bold;
            }
            QTableCornerButton::section {
                background-color: #e0e0e0;
                border: 1px solid #cccccc;
            }
            """
        )
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.cluster_table_title = QLabel("Cluster table")
        self.cluster_table_title.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #333333; padding: 3px 2px;"
        )
        layout.addWidget(self.cluster_table_title)

        self.cluster_table_active = QLabel(
            "Choose ‘Cluster Table’, then hover or select a point."
        )
        self.cluster_table_active.setWordWrap(True)
        self.cluster_table_active.setTextFormat(Qt.TextFormat.RichText)
        self.cluster_table_active.setStyleSheet(
            "background-color: #f5f5f5; border: 1px solid #cccccc; "
            "border-left: 5px solid #FF9800; border-radius: 4px; "
            "padding: 8px; color: #333333;"
        )
        layout.addWidget(self.cluster_table_active)

        self.cluster_table_summary = QLabel("No cluster is currently shown.")
        self.cluster_table_summary.setWordWrap(True)
        self.cluster_table_summary.setStyleSheet(
            "color: #666666; font-style: italic; padding: 3px 2px;"
        )
        layout.addWidget(self.cluster_table_summary)

        self.cluster_table = QTableWidget(0, 6)
        self.cluster_table.setHorizontalHeaderLabels(
            ["Code", "Primary diagnosis", "X", "Y", "Admissions", "Bed days"]
        )
        self.cluster_table.verticalHeader().setVisible(False)
        self.cluster_table.setAlternatingRowColors(True)
        self.cluster_table.setWordWrap(False)
        self.cluster_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.cluster_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.cluster_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.cluster_table.verticalHeader().setDefaultSectionSize(27)
        header = self.cluster_table.horizontalHeader()
        # ResizeToContents recalculates size hints for every populated cell.
        # That is prohibitively expensive while the active row changes during
        # mouse movement, so keep predictable widths and stretch diagnosis.
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.cluster_table.setColumnWidth(0, 78)
        self.cluster_table.setColumnWidth(2, 76)
        self.cluster_table.setColumnWidth(3, 76)
        self.cluster_table.setColumnWidth(4, 90)
        self.cluster_table.setColumnWidth(5, 90)
        layout.addWidget(self.cluster_table, 1)
        return panel

    def _hover_mode_is_table(self) -> bool:
        return self.hover_detail_mode == self.HOVER_MODE_TABLE

    def _on_hover_detail_mode_changed(self, combo_index: int) -> None:
        self.hover_detail_mode = self.hover_mode_combo.itemData(combo_index)
        self._hide_hover_card()
        self._last_hover_idx = None
        tabs = getattr(self, "detail_tabs", None)
        if self._hover_mode_is_table():
            if tabs is not None:
                tabs.setCurrentWidget(self.cluster_table_panel)
            self._sync_cluster_table_to_selection()
        elif tabs is not None:
            tabs.setCurrentWidget(self.ai_panel)

    def _set_cluster_table_anchor(self, indices) -> None:
        normalized = [int(index) for index in indices]
        if normalized:
            candidate = normalized[-1]
            self._cluster_table_anchor_idx = (
                candidate if candidate in self.selected_indices else None
            )
        if self._cluster_table_anchor_idx not in self.selected_indices:
            self._cluster_table_anchor_idx = (
                min(self.selected_indices) if self.selected_indices else None
            )

    def _clear_cluster_table(self) -> None:
        table = getattr(self, "cluster_table", None)
        if table is None:
            return
        table.setRowCount(0)
        self._cluster_table_rendered_region_id = None
        self._cluster_table_members = []
        self._cluster_table_row_by_point = {}
        self._cluster_table_active_idx = None
        self._cluster_table_selected_indices = frozenset()
        self._cluster_table_summary_base = []
        self.cluster_table_title.setText("Cluster table")
        self.cluster_table_active.setText(
            "Hover a point to preview its cluster, or click a point to keep the table visible."
        )
        self.cluster_table_summary.setText("No cluster is currently shown.")

    def _show_cluster_table(self, idx: int) -> None:
        table = getattr(self, "cluster_table", None)
        if table is None or not (0 <= idx < len(self.points)):
            return

        region_id = int(self.region_ids[idx])
        rebuild_table = region_id != self._cluster_table_rendered_region_id
        if rebuild_table:
            members = [int(value) for value in np.where(self.region_ids == region_id)[0]]
            records = [self._cluster_record(member) for member in members]
            records.sort(
                key=lambda record: (
                    self._numeric_value(record["admissions"]) is not None,
                    self._numeric_value(record["admissions"]) or 0.0,
                ),
                reverse=True,
            )
            self._cluster_table_members = members
        else:
            members = self._cluster_table_members
            records = []
        active = self._cluster_record(idx)
        region_name = self._region_name(idx)
        selected_in_cluster = sum(member in self.selected_indices for member in members)
        self.cluster_table_title.setText(
            f"Cluster {region_name} — {len(members)} point{'s' if len(members) != 1 else ''}"
        )
        diagnosis = self._diagnosis_label(idx) or f"Point #{idx}"
        self.cluster_table_active.setText(
            f"<b>{diagnosis}</b><br>"
            f"{self._short_metric_header(active['x_header'])}: "
            f"{self._table_value(active['x'])} &nbsp;&nbsp; "
            f"{self._short_metric_header(active['y_header'])}: "
            f"{self._table_value(active['y'])}"
        )

        if rebuild_table:
            x_numbers = [self._numeric_value(record["x"]) for record in records]
            y_numbers = [self._numeric_value(record["y"]) for record in records]
            admissions = [self._numeric_value(record["admissions"]) for record in records]
            bed_days = [self._numeric_value(record["bed_days"]) for record in records]
            x_numbers = [value for value in x_numbers if value is not None]
            y_numbers = [value for value in y_numbers if value is not None]
            admissions = [value for value in admissions if value is not None]
            bed_days = [value for value in bed_days if value is not None]
            self._cluster_table_summary_base = [
                f"Cluster mean X: {self._table_value(np.mean(x_numbers) if x_numbers else None)}",
                f"mean Y: {self._table_value(np.mean(y_numbers) if y_numbers else None)}",
            ]
            if admissions:
                self._cluster_table_summary_base.append(
                    f"total admissions: {self._table_value(sum(admissions), 0)}"
                )
            if bed_days:
                self._cluster_table_summary_base.append(
                    f"total bed days: {self._table_value(sum(bed_days), 0)}"
                )
        summary_parts = list(self._cluster_table_summary_base)
        if selected_in_cluster:
            summary_parts.append(f"selected here: {selected_in_cluster}")
        self.cluster_table_summary.setText("  |  ".join(summary_parts))

        cluster_color = (
            self.colors_hex[region_id]
            if 0 <= region_id < len(self.colors_hex)
            else "#dbeafe"
        )
        self.cluster_table_active.setStyleSheet(
            "background-color: #f5f5f5; border: 1px solid #cccccc; "
            f"border-left: 5px solid {cluster_color}; border-radius: 4px; "
            "padding: 8px; color: #333333;"
        )
        anchor_brush = QBrush(QColor(cluster_color).lighter(178))
        selected_brush = QBrush(QColor("#fff3e0"))

        if rebuild_table:
            table.setUpdatesEnabled(False)
            try:
                table.setSortingEnabled(False)
                table.setHorizontalHeaderLabels(
                    [
                        "Code",
                        "Primary diagnosis",
                        self._short_metric_header(active["x_header"]),
                        self._short_metric_header(active["y_header"]),
                        "Admissions",
                        "Bed days",
                    ]
                )
                table.setRowCount(len(records))
                self._cluster_table_row_by_point = {}
                for row_number, record in enumerate(records):
                    self._cluster_table_row_by_point[record["idx"]] = row_number
                    values = [
                        record["code"],
                        record["description"],
                        self._table_value(record["x"]),
                        self._table_value(record["y"]),
                        self._table_value(record["admissions"], 0),
                        self._table_value(record["bed_days"], 0),
                    ]
                    for column_number, value in enumerate(values):
                        item = QTableWidgetItem(str(value))
                        item.setData(Qt.ItemDataRole.UserRole, record["idx"])
                        if column_number == 1:
                            item.setToolTip(record["description"])
                        table.setItem(row_number, column_number, item)
                self._cluster_table_rendered_region_id = region_id
            finally:
                table.setUpdatesEnabled(True)

        current_selected = frozenset(
            member for member in members if member in self.selected_indices
        )
        if rebuild_table:
            changed_points = set(self._cluster_table_row_by_point)
        else:
            changed_points = set(current_selected ^ self._cluster_table_selected_indices)
            if self._cluster_table_active_idx is not None:
                changed_points.add(self._cluster_table_active_idx)
            changed_points.add(idx)

        for point_idx in changed_points:
            row_number = self._cluster_table_row_by_point.get(point_idx)
            if row_number is None:
                continue
            is_selected = point_idx in current_selected
            is_active = point_idx == idx
            for column_number in range(table.columnCount()):
                item = table.item(row_number, column_number)
                if item is None:
                    continue
                item.setBackground(
                    anchor_brush if is_active
                    else selected_brush if is_selected
                    else QBrush()
                )
                font = QFont(item.font())
                font.setBold(is_selected)
                item.setFont(font)

        self._cluster_table_active_idx = idx
        self._cluster_table_selected_indices = current_selected
        active_row = self._cluster_table_row_by_point.get(idx)
        if active_row is not None:
            active_item = table.item(active_row, 0)
            if active_item is not None:
                table.scrollToItem(
                    active_item, QAbstractItemView.ScrollHint.PositionAtCenter
                )

    def _show_hover_detail(self, idx: int) -> None:
        if self._hover_mode_is_table():
            self._hide_hover_card()
            self._show_cluster_table(idx)
        else:
            self._show_hover_card(idx)

    def _hide_hover_detail(self) -> None:
        if self._hover_mode_is_table():
            if self.selected_indices:
                self._sync_cluster_table_to_selection()
            else:
                self._clear_cluster_table()
        else:
            self._hide_hover_card()

    def _sync_cluster_table_to_selection(self) -> None:
        if not self._hover_mode_is_table():
            return
        if self.selected_indices:
            anchor = self._cluster_table_anchor_idx
            if anchor not in self.selected_indices:
                anchor = min(self.selected_indices)
                self._cluster_table_anchor_idx = anchor
            self._show_cluster_table(anchor)
        elif self._last_hover_idx is not None:
            self._show_cluster_table(self._last_hover_idx)
        else:
            self._clear_cluster_table()
