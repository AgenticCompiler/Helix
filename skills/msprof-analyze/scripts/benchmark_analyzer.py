#!/usr/bin/env python3
"""
Benchmark Result Analyzer

A flexible tool to analyze and visualize benchmark results from profiler outputs.
Supports comparing different versions of tools with statistical analysis and plotting.

Author: Claude Code
"""

import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import json
import argparse
from datetime import datetime

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark analysis"""
    base_dir: str = "./profile_dir"
    versions: List[str] = None
    rounds: int = 5
    target_ops: List[str] = None
    output_dir: str = "./analysis_output"
    plot_format: str = "png"
    dpi: int = 300
    metric: str = "Avg Time(us)"  # Column to analyze

    def __post_init__(self):
        if self.versions is None:
            self.versions = ["new", "old"]
        if self.target_ops is None:
            self.target_ops = ["count_nonzero_combin_kernel", "count_nonzero_combin_kernel_1"]


class BenchmarkDataCollector:
    """Collect and parse benchmark data from CSV files"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.data = {}

    def find_csv_files(self) -> Dict[str, List[str]]:
        """Find all relevant CSV files in the directory structure"""
        csv_files = {}

        for version in self.config.versions:
            version_files = []
            for round_num in range(1, self.config.rounds + 1):
                pattern = f"{self.config.base_dir}/{version}-{round_num}/**/op_statistic_*.csv"
                files = glob.glob(pattern)
                if files:
                    version_files.extend(files)
                else:
                    print(f"⚠️  Warning: No CSV files found for {version}-{round_num}")

            csv_files[version] = version_files

        return csv_files

    def parse_csv_file(self, file_path: str) -> pd.DataFrame:
        """Parse a single CSV file and return DataFrame"""
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            print(f"❌ Error parsing {file_path}: {e}")
            return pd.DataFrame()

    def extract_round_data(self, files: List[str]) -> Dict[int, pd.DataFrame]:
        """Extract data organized by round number"""
        round_data = {}

        for file_path in files:
            # Extract round number from file path
            match = re.search(rf'({"|".join(self.config.versions)})-(\d+)', file_path)
            if match:
                version, round_num = match.groups()
                round_num = int(round_num)

                if round_num not in round_data:
                    round_data[round_num] = []

                df = self.parse_csv_file(file_path)
                if not df.empty:
                    round_data[round_num].append(df)

        # Merge multiple DataFrames for the same round
        for round_num in round_data:
            if len(round_data[round_num]) > 1:
                round_data[round_num] = pd.concat(round_data[round_num], ignore_index=True)
            elif len(round_data[round_num]) == 1:
                round_data[round_num] = round_data[round_num][0]

        return round_data

    def collect_data(self) -> Dict[str, Dict[int, pd.DataFrame]]:
        """Main method to collect all benchmark data"""
        print(f"🔍 Collecting benchmark data from {self.config.base_dir}")

        csv_files = self.find_csv_files()

        for version in self.config.versions:
            self.data[version] = self.extract_round_data(csv_files[version])

            print(f"📊 {version.title()} version: Found {len(self.data[version])} rounds")

        return self.data


class BenchmarkAnalyzer:
    """Analyze benchmark data and generate statistics"""

    def __init__(self, data: Dict[str, Dict[int, pd.DataFrame]], config: BenchmarkConfig):
        self.data = data
        self.config = config
        self.analysis_results = {}

    def extract_op_metrics(self, df: pd.DataFrame, op_type: str) -> List[float]:
        """Extract metrics for specific OP Type"""
        if df.empty or 'OP Type' not in df.columns:
            return []

        # Filter for specific OP Type
        op_data = df[df['OP Type'] == op_type]

        if op_data.empty:
            return []

        if self.config.metric not in op_data.columns:
            return []

        return op_data[self.config.metric].tolist()

    def calculate_statistics(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistical metrics"""
        if not values:
            return {}

        values = np.array(values)
        return {
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'count': len(values)
        }

    def analyze_by_round(self) -> Dict[str, Dict[int, Dict[str, Dict[str, float]]]]:
        """Analyze metrics organized by version, round, and operation"""
        results = {}

        for version in self.config.versions:
            results[version] = {}
            version_data = self.data[version]

            for round_num in sorted(version_data.keys()):
                results[version][round_num] = {}
                df = version_data[round_num]

                for op_type in self.config.target_ops:
                    metrics = self.extract_op_metrics(df, op_type)
                    stats = self.calculate_statistics(metrics)

                    if stats:  # Only include if we found data
                        results[version][round_num][op_type] = stats

        self.analysis_results = results
        return results

    def generate_comparison_table(self) -> pd.DataFrame:
        """Generate comparison table between versions"""
        comparison_data = []

        for op_type in self.config.target_ops:
            row = {'OP Type': op_type}

            for version in self.config.versions:
                version_rounds = self.analysis_results.get(version, {})

                # Collect all metrics across rounds
                all_values = []
                for round_data in version_rounds.values():
                    if op_type in round_data:
                        all_values.append(round_data[op_type]['mean'])

                if all_values:
                    stats = self.calculate_statistics(all_values)
                    row[f"{version.title()}_mean"] = stats['mean']
                    row[f"{version.title()}_std"] = stats['std']
                    row[f"{version.title()}_median"] = stats['median']
                else:
                    row[f"{version.title()}_mean"] = np.nan
                    row[f"{version.title()}_std"] = np.nan
                    row[f"{version.title()}_median"] = np.nan

            # Calculate improvement
            if len(self.config.versions) == 2 and f"{self.config.versions[0].title()}_mean" in row and f"{self.config.versions[1].title()}_mean" in row:
                old_mean = row.get(f"{self.config.versions[1].title()}_mean", np.nan)
                new_mean = row.get(f"{self.config.versions[0].title()}_mean", np.nan)
                if not np.isnan(old_mean) and not np.isnan(new_mean) and old_mean > 0:
                    improvement = ((old_mean - new_mean) / old_mean) * 100
                    row['improvement_%'] = improvement

            comparison_data.append(row)

        return pd.DataFrame(comparison_data)


class BenchmarkVisualizer:
    """Create visualizations for benchmark results"""

    def __init__(self, data: Dict[str, Dict[int, pd.DataFrame]],
                 analysis_results: Dict, config: BenchmarkConfig):
        self.data = data
        self.analysis_results = analysis_results
        self.config = config

    def create_line_plots(self) -> Dict[str, str]:
        """Create line plots comparing versions across rounds"""
        plot_paths = {}

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        for op_type in self.config.target_ops:
            plt.figure(figsize=(12, 8))

            # Prepare data for plotting
            for version in self.config.versions:
                rounds = []
                values = []

                version_rounds = sorted(self.analysis_results[version].keys())

                for round_num in version_rounds:
                    if op_type in self.analysis_results[version][round_num]:
                        rounds.append(round_num)
                        values.append(self.analysis_results[version][round_num][op_type]['mean'])

                if values:
                    plt.plot(rounds, values, marker='o', label=f'{version.title()} Version',
                            linewidth=2, markersize=6)

            plt.xlabel('Round Number')
            plt.ylabel(f'{self.config.metric} (μs)')
            plt.title(f'Performance Comparison: {op_type}')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Add annotations for each point
            for version in self.config.versions:
                version_rounds = sorted(self.analysis_results[version].keys())
                for round_num in version_rounds:
                    if op_type in self.analysis_results[version][round_num]:
                        value = self.analysis_results[version][round_num][op_type]['mean']
                        plt.annotate(f'{value:.1f}',
                                   (round_num, value),
                                   textcoords="offset points",
                                   xytext=(0,10),
                                   ha='center',
                                   fontsize=8)

            # Save plot
            filename = f"{op_type}_comparison.{self.config.plot_format}"
            plot_path = os.path.join(self.config.output_dir, filename)
            plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()

            plot_paths[op_type] = plot_path
            print(f"📈 Saved plot: {plot_path}")

        return plot_paths

    def create_summary_heatmap(self, comparison_df: pd.DataFrame) -> str:
        """Create a heatmap summary of all comparisons"""
        plt.figure(figsize=(14, 10))

        # Prepare data for heatmap
        numeric_columns = [col for col in comparison_df.columns if col not in ['OP Type']]
        if numeric_columns:
            heatmap_data = comparison_df.set_index('OP Type')[numeric_columns]

            # Create heatmap
            sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlBu_r',
                       center=0, square=True, linewidths=0.5)

            plt.title('Benchmark Results Heatmap', fontsize=16, fontweight='bold')
            plt.tight_layout()

            # Save heatmap
            heatmap_path = os.path.join(self.config.output_dir, f"summary_heatmap.{self.config.plot_format}")
            plt.savefig(heatmap_path, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()

            print(f"🔥 Saved heatmap: {heatmap_path}")
            return heatmap_path

        return ""


class BenchmarkReporter:
    """Generate comprehensive reports"""

    def __init__(self, analyzer: BenchmarkAnalyzer, visualizer: BenchmarkVisualizer,
                 config: BenchmarkConfig):
        self.analyzer = analyzer
        self.visualizer = visualizer
        self.config = config
        self.comparison_df = analyzer.generate_comparison_table()

    def save_results(self) -> Dict[str, str]:
        """Save all analysis results to files"""
        os.makedirs(self.config.output_dir, exist_ok=True)

        saved_files = {}

        # Save comparison table
        csv_path = os.path.join(self.config.output_dir, "comparison_table.csv")
        self.comparison_df.to_csv(csv_path, index=False)
        saved_files['comparison_csv'] = csv_path
        print(f"📋 Saved comparison table: {csv_path}")

        # Save detailed analysis results as JSON
        json_path = os.path.join(self.config.output_dir, "detailed_analysis.json")
        with open(json_path, 'w') as f:
            json.dump(self.analyzer.analysis_results, f, indent=2, default=str)
        saved_files['analysis_json'] = json_path
        print(f"📄 Saved detailed analysis: {json_path}")

        # Generate plots
        plot_paths = self.visualizer.create_line_plots()
        saved_files['plots'] = plot_paths

        # Generate heatmap
        heatmap_path = self.visualizer.create_summary_heatmap(self.comparison_df)
        if heatmap_path:
            saved_files['heatmap'] = heatmap_path

        return saved_files

    def print_summary(self):
        """Print a summary of the analysis results"""
        print("\n" + "="*80)
        print("📊 BENCHMARK ANALYSIS SUMMARY")
        print("="*80)

        print(f"\n📁 Data source: {self.config.base_dir}")
        print(f"🔨 Versions compared: {', '.join(self.config.versions)}")
        print(f"🎯 Target operations: {', '.join(self.config.target_ops)}")
        print(f"📈 Metric analyzed: {self.config.metric}")

        if not self.comparison_df.empty:
            print(f"\n📋 Comparison Results:")
            print(self.comparison_df.to_string(index=False, float_format='%.2f'))

            if 'improvement_%' in self.comparison_df.columns:
                improvements = self.comparison_df['improvement_%'].dropna()
                if not improvements.empty:
                    avg_improvement = improvements.mean()
                    print(f"\n🚀 Average improvement: {avg_improvement:.2f}%")

        print(f"\n📂 Output directory: {self.config.output_dir}")
        print("="*80)


def load_config_from_file(config_path: str) -> BenchmarkConfig:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return BenchmarkConfig(**config_dict)
    except Exception as e:
        print(f"❌ Error loading config file {config_path}: {e}")
        print("Using default configuration")
        return BenchmarkConfig()


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Benchmark Results Analyzer")
    parser.add_argument("--config", type=str, help="Path to configuration JSON file")
    parser.add_argument("--base-dir", type=str, default="./profile_dir",
                       help="Base directory containing benchmark results")
    parser.add_argument("--versions", nargs="+", default=["new", "old"],
                       help="Tool versions to compare")
    parser.add_argument("--rounds", type=int, default=5,
                       help="Number of rounds per version")
    parser.add_argument("--target-ops", nargs="+",
                       help="Specific OP types to analyze")
    parser.add_argument("--output-dir", type=str, default="./analysis_output",
                       help="Output directory for results")
    parser.add_argument("--metric", type=str, default="Avg Time(us)",
                       help="Metric to analyze from CSV")

    args = parser.parse_args()

    # Load configuration
    if args.config and os.path.exists(args.config):
        config = load_config_from_file(args.config)
    else:
        config = BenchmarkConfig(
            base_dir=args.base_dir,
            versions=args.versions,
            rounds=args.rounds,
            target_ops=args.target_ops,
            output_dir=args.output_dir,
            metric=args.metric
        )

    print("🚀 Starting Benchmark Analysis")
    print(f"⚙️  Configuration: {config}")

    # Execute analysis pipeline
    collector = BenchmarkDataCollector(config)
    data = collector.collect_data()

    if not data:
        print("❌ No data found. Please check your configuration and directory structure.")
        return

    analyzer = BenchmarkAnalyzer(data, config)
    analysis_results = analyzer.analyze_by_round()

    visualizer = BenchmarkVisualizer(data, analysis_results, config)
    reporter = BenchmarkReporter(analyzer, visualizer, config)

    # Save results and print summary
    saved_files = reporter.save_results()
    reporter.print_summary()

    print(f"\n✅ Analysis complete! Check {config.output_dir} for detailed results.")


if __name__ == "__main__":
    main()
