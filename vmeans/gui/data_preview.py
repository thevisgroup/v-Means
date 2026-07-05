
import sys
import os
import tempfile
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

_MPL_CACHE_DIR = os.path.join(tempfile.gettempdir(), "viskmean_matplotlib_cache")
os.makedirs(_MPL_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _MPL_CACHE_DIR)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QComboBox, QSlider, QCheckBox,
    QSpinBox, QDoubleSpinBox, QGroupBox, QScrollArea, QSplitter,
    QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem,
    QProgressBar, QFrame, QSizePolicy, QGridLayout, QStackedWidget,
    QHeaderView, QSpacerItem, QDialog, QDialogButtonBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize
from PyQt6.QtGui import QFont, QPalette, QColor

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from vmeans.core_analysis import (
    compute_better_center, detect_gradient_boundaries,
    detect_dynamic_centers, segment_points_by_region
)
from vmeans.data import generate_structured_points, convert_to_polar
from vmeans.segment import segment_points_by_theta
from vmeans.interface import PlotOptions
from vmeans.animation import build_enhanced_visible_frames, StepFrame

class DataPreviewDialog(QDialog):
    """Dialog for previewing uploaded data with statistics and analysis"""

    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.df = df
        self.setWindowTitle("📊 Data Preview & Analysis")
        self.setMinimumSize(1000, 700)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Create tab widget for organizing content
        tab_widget = QTabWidget()
        tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #ccc;
                border-radius: 4px;
                background: white;
            }
            QTabBar::tab {
                padding: 8px 20px;
                font-size: 13px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #2196F3;
                color: white;
            }
            QTabBar::tab:!selected {
                background-color: #e0e0e0;
            }
        """)

        # Tab 1: Data Preview
        preview_tab = self.create_preview_tab()
        tab_widget.addTab(preview_tab, "🔍 Data Preview")

        # Tab 2: Statistical Analysis
        stats_tab = self.create_statistics_tab()
        tab_widget.addTab(stats_tab, "📈 Statistical Analysis")

        # Tab 3: Distribution Plots
        dist_tab = self.create_distribution_tab()
        tab_widget.addTab(dist_tab, "📊 Distribution")

        layout.addWidget(tab_widget)

        # Dialog Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)

    def create_preview_tab(self) -> QWidget:
        """Create the data preview tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        # --- Dataset Info ---
        info_group = QGroupBox("📋 Dataset Information")
        info_layout = QHBoxLayout(info_group)

        rows, cols = self.df.shape
        info_layout.addWidget(QLabel(f"<b>Shape:</b> {rows} rows × {cols} columns"))

        memory_mb = self.df.memory_usage(deep=True).sum() / 1024 / 1024
        info_layout.addWidget(QLabel(f"<b>Memory:</b> {memory_mb:.2f} MB"))

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        info_layout.addWidget(QLabel(f"<b>Numeric Columns:</b> {len(numeric_cols)}"))

        info_layout.addStretch()
        layout.addWidget(info_group)

        # --- Data Preview Table ---
        preview_group = QGroupBox("🔍 Data Preview (First 10 Rows)")
        preview_layout = QVBoxLayout(preview_group)

        preview_table = QTableWidget()
        preview_df = self.df.head(10)
        preview_table.setRowCount(len(preview_df))
        preview_table.setColumnCount(len(preview_df.columns))
        preview_table.setHorizontalHeaderLabels([str(col) for col in preview_df.columns])

        for i in range(len(preview_df)):
            for j, col in enumerate(preview_df.columns):
                value = preview_df.iloc[i, j]
                if pd.isna(value):
                    item = QTableWidgetItem("NaN")
                    item.setForeground(QColor(200, 0, 0))
                elif isinstance(value, float):
                    item = QTableWidgetItem(f"{value:.4f}")
                else:
                    item = QTableWidgetItem(str(value))
                preview_table.setItem(i, j, item)

        preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        preview_table.setAlternatingRowColors(True)
        preview_table.setStyleSheet("""
            QTableWidget {
                alternate-background-color: #f5f5f5;
                gridline-color: #ddd;
            }
        """)
        preview_layout.addWidget(preview_table)
        layout.addWidget(preview_group)

        # --- Column Info Table ---
        col_group = QGroupBox("📋 Column Information")
        col_layout = QVBoxLayout(col_group)

        col_table = QTableWidget()
        all_cols = self.df.columns.tolist()

        col_table.setColumnCount(6)
        col_table.setHorizontalHeaderLabels([
            "Column", "Type", "Non-Null", "Null", "Unique", "Numeric"
        ])
        col_table.setRowCount(len(all_cols))

        for i, col in enumerate(all_cols):
            col_table.setItem(i, 0, QTableWidgetItem(str(col)))
            col_table.setItem(i, 1, QTableWidgetItem(str(self.df[col].dtype)))
            col_table.setItem(i, 2, QTableWidgetItem(str(self.df[col].count())))

            null_count = self.df[col].isnull().sum()
            null_item = QTableWidgetItem(str(null_count))
            if null_count > 0:
                null_item.setForeground(QColor(200, 0, 0))
            col_table.setItem(i, 3, null_item)

            col_table.setItem(i, 4, QTableWidgetItem(str(self.df[col].nunique())))

            is_numeric = "✓" if col in numeric_cols else "✗"
            numeric_item = QTableWidgetItem(is_numeric)
            numeric_item.setForeground(QColor(0, 150, 0) if col in numeric_cols else QColor(150, 150, 150))
            col_table.setItem(i, 5, numeric_item)

        col_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        col_table.setAlternatingRowColors(True)
        col_table.setMaximumHeight(150)
        col_layout.addWidget(col_table)
        layout.addWidget(col_group)

        return tab

    def create_statistics_tab(self) -> QWidget:
        """Create the statistical analysis tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            no_data_label = QLabel("⚠️ No numeric columns found for statistical analysis")
            no_data_label.setStyleSheet("color: orange; font-size: 14px; padding: 20px;")
            no_data_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(no_data_label)
            return tab

        # --- Descriptive Statistics ---
        desc_group = QGroupBox("📊 Descriptive Statistics (Numeric Columns)")
        desc_layout = QVBoxLayout(desc_group)

        # Calculate statistics
        stats_df = self.df[numeric_cols].describe()
        stats_df.loc['missing'] = self.df[numeric_cols].isnull().sum()
        stats_df.loc['skewness'] = self.df[numeric_cols].skew()
        stats_df.loc['kurtosis'] = self.df[numeric_cols].kurtosis()

        # Create table
        stats_table = QTableWidget()
        stats_table.setRowCount(len(stats_df.index))
        stats_table.setColumnCount(len(stats_df.columns))
        stats_table.setHorizontalHeaderLabels([str(col) for col in stats_df.columns])
        stats_table.setVerticalHeaderLabels([str(idx) for idx in stats_df.index])

        for i, idx in enumerate(stats_df.index):
            for j, col in enumerate(stats_df.columns):
                value = stats_df.loc[idx, col]
                if pd.isna(value):
                    item = QTableWidgetItem("N/A")
                    item.setForeground(QColor(150, 150, 150))
                else:
                    # Format based on row type
                    if idx == 'count' or idx == 'missing':
                        item = QTableWidgetItem(f"{int(value)}")
                    else:
                        item = QTableWidgetItem(f"{value:.4f}")

                    # Highlight certain values
                    if idx == 'missing' and value > 0:
                        item.setForeground(QColor(200, 0, 0))
                    elif idx == 'skewness' and abs(value) > 1:
                        item.setForeground(QColor(200, 100, 0))  # Orange for high skewness

                stats_table.setItem(i, j, item)

        stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        stats_table.setAlternatingRowColors(True)
        desc_layout.addWidget(stats_table)

        # Legend for statistics
        legend_label = QLabel(
            "<b>Legend:</b> count=sample size, mean=average, std=standard deviation, min/max=minimum/maximum, "
            "25%/50%/75%=quartiles, missing=null values, skewness=asymmetry, kurtosis=tailedness"
        )
        legend_label.setStyleSheet("color: #666; font-size: 11px; padding: 5px;")
        legend_label.setWordWrap(True)
        desc_layout.addWidget(legend_label)

        layout.addWidget(desc_group)

        # --- Correlation Matrix ---
        if len(numeric_cols) >= 2:
            corr_group = QGroupBox("🔗 Correlation Matrix")
            corr_layout = QVBoxLayout(corr_group)

            corr_df = self.df[numeric_cols].corr()

            corr_table = QTableWidget()
            corr_table.setRowCount(len(corr_df.index))
            corr_table.setColumnCount(len(corr_df.columns))
            corr_table.setHorizontalHeaderLabels([str(col) for col in corr_df.columns])
            corr_table.setVerticalHeaderLabels([str(idx) for idx in corr_df.index])

            for i, idx in enumerate(corr_df.index):
                for j, col in enumerate(corr_df.columns):
                    value = corr_df.loc[idx, col]
                    item = QTableWidgetItem(f"{value:.3f}")

                    # Color code correlation values
                    if i == j:
                        item.setBackground(QColor(200, 200, 200))
                    elif abs(value) >= 0.7:
                        if value > 0:
                            item.setBackground(QColor(144, 238, 144))  # Light green
                        else:
                            item.setBackground(QColor(255, 182, 193))  # Light red
                    elif abs(value) >= 0.4:
                        if value > 0:
                            item.setBackground(QColor(200, 255, 200))
                        else:
                            item.setBackground(QColor(255, 220, 220))

                    corr_table.setItem(i, j, item)

            corr_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            corr_table.setMaximumHeight(200)
            corr_layout.addWidget(corr_table)

            # Correlation legend
            corr_legend = QLabel(
                "<b>Color Legend:</b> "
                "<span style='background-color: #90EE90; padding: 2px;'>Strong Positive (≥0.7)</span> | "
                "<span style='background-color: #C8FFC8; padding: 2px;'>Moderate Positive</span> | "
                "<span style='background-color: #FFB6C1; padding: 2px;'>Strong Negative (≤-0.7)</span> | "
                "<span style='background-color: #FFDCDC; padding: 2px;'>Moderate Negative</span>"
            )
            corr_legend.setStyleSheet("font-size: 11px; padding: 5px;")
            corr_layout.addWidget(corr_legend)

            layout.addWidget(corr_group)

        # --- Data Quality Summary ---
        quality_group = QGroupBox("✅ Data Quality Summary")
        quality_layout = QGridLayout(quality_group)

        total_cells = self.df.shape[0] * self.df.shape[1]
        missing_cells = self.df.isnull().sum().sum()
        completeness = (1 - missing_cells / total_cells) * 100 if total_cells > 0 else 100

        quality_layout.addWidget(QLabel(f"<b>Total Cells:</b> {total_cells:,}"), 0, 0)
        quality_layout.addWidget(QLabel(f"<b>Missing Cells:</b> {missing_cells:,}"), 0, 1)

        completeness_label = QLabel(f"<b>Completeness:</b> {completeness:.1f}%")
        if completeness >= 95:
            completeness_label.setStyleSheet("color: green;")
        elif completeness >= 80:
            completeness_label.setStyleSheet("color: orange;")
        else:
            completeness_label.setStyleSheet("color: red;")
        quality_layout.addWidget(completeness_label, 0, 2)

        # Duplicate rows
        duplicates = self.df.duplicated().sum()
        dup_label = QLabel(f"<b>Duplicate Rows:</b> {duplicates}")
        if duplicates > 0:
            dup_label.setStyleSheet("color: orange;")
        quality_layout.addWidget(dup_label, 1, 0)

        layout.addWidget(quality_group)
        layout.addStretch()

        return tab

    def create_distribution_tab(self) -> QWidget:
        """Create the distribution visualization tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            no_data_label = QLabel("⚠️ No numeric columns found for distribution plots")
            no_data_label.setStyleSheet("color: orange; font-size: 14px; padding: 20px;")
            no_data_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(no_data_label)
            return tab

        # Column selector
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Select columns to plot:"))

        self.x_dist_combo = QComboBox()
        self.x_dist_combo.addItems(numeric_cols)
        self.x_dist_combo.setMinimumWidth(120)
        selector_layout.addWidget(QLabel("X:"))
        selector_layout.addWidget(self.x_dist_combo)

        self.y_dist_combo = QComboBox()
        self.y_dist_combo.addItems(numeric_cols)
        if len(numeric_cols) >= 2:
            self.y_dist_combo.setCurrentIndex(1)
        self.y_dist_combo.setMinimumWidth(120)
        selector_layout.addWidget(QLabel("Y:"))
        selector_layout.addWidget(self.y_dist_combo)

        refresh_btn = QPushButton("🔄 Refresh Plots")
        refresh_btn.clicked.connect(self.refresh_distribution_plots)
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        selector_layout.addWidget(refresh_btn)
        selector_layout.addStretch()

        layout.addLayout(selector_layout)

        # Create matplotlib figure for distributions
        self.dist_figure = Figure(figsize=(10, 6), dpi=100)
        self.dist_canvas = FigureCanvas(self.dist_figure)
        self.dist_canvas.setMinimumHeight(400)
        layout.addWidget(self.dist_canvas)

        # Initial plot
        self.refresh_distribution_plots()

        return tab

    def refresh_distribution_plots(self):
        """Refresh the distribution plots"""
        self.dist_figure.clear()

        x_col = self.x_dist_combo.currentText()
        y_col = self.y_dist_combo.currentText()

        if not x_col or not y_col:
            return

        x_data = self.df[x_col].dropna()
        y_data = self.df[y_col].dropna()

        # Create 2x2 subplot layout
        # [Histogram X] [Histogram Y]
        # [Scatter X-Y] [Box plots  ]

        ax1 = self.dist_figure.add_subplot(2, 2, 1)
        ax2 = self.dist_figure.add_subplot(2, 2, 2)
        ax3 = self.dist_figure.add_subplot(2, 2, 3)
        ax4 = self.dist_figure.add_subplot(2, 2, 4)

        # Histogram for X
        ax1.hist(x_data, bins=30, color='#2196F3', alpha=0.7, edgecolor='white')
        ax1.axvline(x_data.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean: {x_data.mean():.2f}')
        ax1.axvline(x_data.median(), color='green', linestyle=':', linewidth=1.5, label=f'Median: {x_data.median():.2f}')
        ax1.set_title(f'Distribution: {x_col}', fontsize=11, fontweight='bold')
        ax1.set_xlabel(x_col)
        ax1.set_ylabel('Frequency')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Histogram for Y
        ax2.hist(y_data, bins=30, color='#4CAF50', alpha=0.7, edgecolor='white')
        ax2.axvline(y_data.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean: {y_data.mean():.2f}')
        ax2.axvline(y_data.median(), color='green', linestyle=':', linewidth=1.5, label=f'Median: {y_data.median():.2f}')
        ax2.set_title(f'Distribution: {y_col}', fontsize=11, fontweight='bold')
        ax2.set_xlabel(y_col)
        ax2.set_ylabel('Frequency')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Scatter plot
        # Use common indices
        common_df = self.df[[x_col, y_col]].dropna()
        if len(common_df) > 0:
            ax3.scatter(common_df[x_col], common_df[y_col], alpha=0.5, s=20, c='#9C27B0')

            # Add trend line
            if len(common_df) > 2:
                z = np.polyfit(common_df[x_col], common_df[y_col], 1)
                p = np.poly1d(z)
                x_line = np.linspace(common_df[x_col].min(), common_df[x_col].max(), 100)
                ax3.plot(x_line, p(x_line), 'r--', linewidth=1.5, label='Trend line')

            # Calculate and show correlation
            corr = common_df[x_col].corr(common_df[y_col])
            ax3.set_title(f'Scatter: {x_col} vs {y_col} (r={corr:.3f})', fontsize=11, fontweight='bold')
        else:
            ax3.set_title(f'Scatter: {x_col} vs {y_col}', fontsize=11, fontweight='bold')

        ax3.set_xlabel(x_col)
        ax3.set_ylabel(y_col)
        ax3.grid(True, alpha=0.3)

        # Box plots
        box_data = [x_data.values, y_data.values]
        bp = ax4.boxplot(box_data, tick_labels=[x_col, y_col], patch_artist=True)
        bp['boxes'][0].set_facecolor('#2196F3')
        bp['boxes'][1].set_facecolor('#4CAF50')
        for box in bp['boxes']:
            box.set_alpha(0.7)
        ax4.set_title('Box Plots', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Value')
        ax4.grid(True, alpha=0.3, axis='y')

        self.dist_figure.tight_layout()
        self.dist_canvas.draw()


def clean_data(df: pd.DataFrame, x_col: str, y_col: str,
               remove_outliers: bool = False, outlier_std: float = 3.0,
               standardize: bool = False) -> tuple:
    """
    Clean data: remove NA values, optionally remove outliers, optionally standardize

    Returns:
        (points, cleaned_df, info_messages)
    """
    messages = []

    # Create copy
    cleaned_df = df[[x_col, y_col]].copy()
    original_rows = len(cleaned_df)

    # Remove NA values
    cleaned_df = cleaned_df.dropna()
    na_removed = original_rows - len(cleaned_df)
    if na_removed > 0:
        messages.append(f"Removed {na_removed} rows with null values")

    # Remove outliers
    if remove_outliers and len(cleaned_df) > 0:
        x_mean = cleaned_df[x_col].mean()
        x_std = cleaned_df[x_col].std()
        y_mean = cleaned_df[y_col].mean()
        y_std = cleaned_df[y_col].std()

        x_outliers = (np.abs(cleaned_df[x_col] - x_mean) > outlier_std * x_std)
        y_outliers = (np.abs(cleaned_df[y_col] - y_mean) > outlier_std * y_std)

        before_outlier = len(cleaned_df)
        cleaned_df = cleaned_df[~(x_outliers | y_outliers)]
        outliers_removed = before_outlier - len(cleaned_df)

        if outliers_removed > 0:
            messages.append(f"Removed {outliers_removed} outliers (>{outlier_std}σ)")

    # Standardize
    if standardize and len(cleaned_df) > 0:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(cleaned_df.values)
        cleaned_df[x_col] = scaled_values[:, 0]
        cleaned_df[y_col] = scaled_values[:, 1]
        messages.append("Data standardized (mean=0, std=1)")

    if len(cleaned_df) > 0:
        points = cleaned_df.values
        messages.append(f"✓ Final: {len(points)} valid points")
        return points, cleaned_df, messages
    else:
        return None, None, ["Error: No data remaining after cleaning"]


def normalize_column_name(name: str) -> str:
    """Remove newlines from column names for proper display in ComboBox."""
    return " ".join(name.splitlines())


def adjust_combo_popup_width(combo: QComboBox):
    """
    Auto-adjust the width of the QComboBox and its popup to fit the longest item.
    Adjust both the ComboBox widget and its dropdown popup width.
    """
    view = combo.view()
    font_metrics = combo.fontMetrics()
    max_width = 0

    # Find the widest item text
    for i in range(combo.count()):
        text = combo.itemText(i)
        width = font_metrics.horizontalAdvance(text)
        max_width = max(max_width, width)

    # Add padding and arrow button width
    required_width = max_width + 50

    # Cap at 400px to keep things reasonable
    combo.setMinimumWidth(min(required_width, 400))

    # Match the dropdown popup width
    view.setMinimumWidth(max(combo.width(), required_width))
