
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, probplot
from typing import Optional, List, Tuple, Dict
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9



class DistributionVisualizer:
    
    def __init__(self, output_dir: str = "data/output/stage14_visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.colors = {
            'histogram': '#3498db',
            'kde': '#e74c3c',
            'normal': '#2ecc71',
            'box': '#9b59b6',
            'violin': '#f39c12'
        }
    
    def plot_histogram_with_kde(
        self,
        data: np.ndarray,
        feature_name: str,
        bins: int = 30,
        show_normal: bool = True,
        title: Optional[str] = None,
        save: bool = True
    ) -> plt.Figure:
        data_clean = data[~np.isnan(data)]
        
        if len(data_clean) < 2:
            print(f" Недостаточно данных для {feature_name}")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        n, bins_edges, patches = ax.hist(
            data_clean,
            bins=bins,
            density=True,
            alpha=0.6,
            color=self.colors['histogram'],
            edgecolor='black',
            label='Histogram'
        )
        
        if len(data_clean) >= 3 and np.std(data_clean) > 1e-10:
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(data_clean)
                x_range = np.linspace(data_clean.min(), data_clean.max(), 200)
                kde_values = kde(x_range)
                ax.plot(
                    x_range,
                    kde_values,
                    color=self.colors['kde'],
                    linewidth=2,
                    label='KDE',
                    linestyle='--'
                )
            except np.linalg.LinAlgError:
                pass
        
        if show_normal:
            mu, sigma = data_clean.mean(), data_clean.std()
            x_range = np.linspace(data_clean.min(), data_clean.max(), 200)
            normal_values = norm.pdf(x_range, mu, sigma)
            ax.plot(
                x_range,
                normal_values,
                color=self.colors['normal'],
                linewidth=2,
                label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})',
                linestyle='-.'
            )
        
        if title is None:
            title = f'Distribution of {feature_name}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Value', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        stats_text = f'n={len(data_clean)}\nMean={data_clean.mean():.2f}\nStd={data_clean.std():.2f}\nSkew={stats.skew(data_clean):.2f}\nKurt={stats.kurtosis(data_clean):.2f}'
        ax.text(
            0.98, 0.97,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        plt.tight_layout()
        
        if save:
            filename = self.output_dir / f"{feature_name}_histogram_kde.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f" Saved: {filename}")
        
        return fig
    
    def plot_qq_plot(
        self,
        data: np.ndarray,
        feature_name: str,
        dist: str = 'norm',
        title: Optional[str] = None,
        save: bool = True
    ) -> plt.Figure:
        data_clean = data[~np.isnan(data)]
        
        if len(data_clean) < 3:
            print(f" Недостаточно данных для {feature_name}")
            return None
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        probplot(data_clean, dist=dist, plot=ax)
        
        ax.get_lines()[0].set_markerfacecolor(self.colors['histogram'])
        ax.get_lines()[0].set_markeredgecolor('black')
        ax.get_lines()[0].set_markersize(5)
        ax.get_lines()[0].set_alpha(0.6)
        
        ax.get_lines()[1].set_color(self.colors['kde'])
        ax.get_lines()[1].set_linewidth(2)
        ax.get_lines()[1].set_linestyle('--')
        
        if title is None:
            title = f'Q-Q Plot: {feature_name} vs {dist.capitalize()}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Theoretical Quantiles', fontsize=12)
        ax.set_ylabel('Sample Quantiles', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        theoretical_quantiles = ax.get_lines()[0].get_xdata()
        sample_quantiles = ax.get_lines()[0].get_ydata()
        r2 = np.corrcoef(theoretical_quantiles, sample_quantiles)[0, 1] ** 2
        
        ax.text(
            0.05, 0.95,
            f'R² = {r2:.4f}',
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        plt.tight_layout()
        
        if save:
            filename = self.output_dir / f"{feature_name}_qq_plot.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f" Saved: {filename}")
        
        return fig
    
    def plot_pp_plot(
        self,
        data: np.ndarray,
        feature_name: str,
        title: Optional[str] = None,
        save: bool = True
    ) -> plt.Figure:
        data_clean = data[~np.isnan(data)]
        
        if len(data_clean) < 3:
            print(f" Недостаточно данных для {feature_name}")
            return None
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        data_standardized = (data_clean - data_clean.mean()) / data_clean.std()
        
        data_sorted = np.sort(data_standardized)
        
        n = len(data_sorted)
        empirical_probs = np.arange(1, n + 1) / (n + 1)
        
        theoretical_probs = norm.cdf(data_sorted)
        
        ax.scatter(
            theoretical_probs,
            empirical_probs,
            alpha=0.6,
            s=30,
            color=self.colors['histogram'],
            edgecolors='black',
            linewidth=0.5,
            label='Data'
        )
        
        ax.plot(
            [0, 1],
            [0, 1],
            color=self.colors['kde'],
            linewidth=2,
            linestyle='--',
            label='Perfect Fit'
        )
        
        if title is None:
            title = f'P-P Plot: {feature_name} vs Normal'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Theoretical Cumulative Probability', fontsize=12)
        ax.set_ylabel('Empirical Cumulative Probability', fontsize=12)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        max_deviation = np.max(np.abs(theoretical_probs - empirical_probs))
        ax.text(
            0.05, 0.95,
            f'Max Deviation = {max_deviation:.4f}',
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        plt.tight_layout()
        
        if save:
            filename = self.output_dir / f"{feature_name}_pp_plot.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f" Saved: {filename}")
        
        return fig
    
    def plot_boxplot_violin(
        self,
        data: np.ndarray,
        feature_name: str,
        title: Optional[str] = None,
        save: bool = True
    ) -> plt.Figure:
        data_clean = data[~np.isnan(data)]
        
        if len(data_clean) < 3:
            print(f" Недостаточно данных для {feature_name}")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        bp = axes[0].boxplot(
            data_clean,
            vert=True,
            patch_artist=True,
            widths=0.5,
            showmeans=True,
            meanline=True
        )
        
        bp['boxes'][0].set_facecolor(self.colors['box'])
        bp['boxes'][0].set_alpha(0.6)
        bp['medians'][0].set_color('red')
        bp['medians'][0].set_linewidth(2)
        bp['means'][0].set_color('green')
        bp['means'][0].set_linewidth(2)
        
        axes[0].set_ylabel('Value', fontsize=12)
        axes[0].set_title('Box Plot', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        q1, median, q3 = np.percentile(data_clean, [25, 50, 75])
        iqr = q3 - q1
        stats_text = f'Median={median:.2f}\nQ1={q1:.2f}\nQ3={q3:.2f}\nIQR={iqr:.2f}'
        axes[0].text(
            0.98, 0.97,
            stats_text,
            transform=axes[0].transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        parts = axes[1].violinplot(
            [data_clean],
            positions=[1],
            widths=0.7,
            showmeans=True,
            showmedians=True,
            showextrema=True
        )
        
        for pc in parts['bodies']:
            pc.set_facecolor(self.colors['violin'])
            pc.set_alpha(0.6)
            pc.set_edgecolor('black')
        
        parts['cmeans'].set_color('green')
        parts['cmeans'].set_linewidth(2)
        parts['cmedians'].set_color('red')
        parts['cmedians'].set_linewidth(2)
        
        axes[1].set_ylabel('Value', fontsize=12)
        axes[1].set_title('Violin Plot', fontsize=12, fontweight='bold')
        axes[1].set_xticks([1])
        axes[1].set_xticklabels([feature_name])
        axes[1].grid(True, alpha=0.3, axis='y')
        
        if title is None:
            title = f'Box & Violin Plot: {feature_name}'
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if save:
            filename = self.output_dir / f"{feature_name}_box_violin.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f" Saved: {filename}")
        
        return fig
    
    def plot_full_diagnostic(
        self,
        data: np.ndarray,
        feature_name: str,
        bins: int = 30,
        save: bool = True
    ) -> plt.Figure:
        data_clean = data[~np.isnan(data)]
        
        if len(data_clean) < 3:
            print(f" Недостаточно данных для {feature_name}")
            return None
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        n, bins_edges, patches = ax1.hist(
            data_clean,
            bins=bins,
            density=True,
            alpha=0.6,
            color=self.colors['histogram'],
            edgecolor='black'
        )
        
        if len(data_clean) >= 3 and np.std(data_clean) > 1e-10:
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(data_clean)
                x_range = np.linspace(data_clean.min(), data_clean.max(), 200)
                kde_values = kde(x_range)
                ax1.plot(x_range, kde_values, color=self.colors['kde'], linewidth=2, label='KDE')
            except np.linalg.LinAlgError:
                pass
        
        mu, sigma = data_clean.mean(), data_clean.std()
        x_range = np.linspace(data_clean.min(), data_clean.max(), 200)
        ax1.plot(x_range, norm.pdf(x_range, mu, sigma), color=self.colors['normal'], 
                linewidth=2, linestyle='--', label='Normal Fit')
        
        ax1.set_title('Histogram + KDE', fontweight='bold')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[0, 1])
        probplot(data_clean, dist='norm', plot=ax2)
        ax2.get_lines()[0].set_markerfacecolor(self.colors['histogram'])
        ax2.get_lines()[0].set_alpha(0.6)
        ax2.get_lines()[1].set_color(self.colors['kde'])
        ax2.set_title('Q-Q Plot', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(gs[1, 0])
        bp = ax3.boxplot(
            data_clean,
            vert=False,
            patch_artist=True,
            widths=0.5,
            showmeans=True
        )
        bp['boxes'][0].set_facecolor(self.colors['box'])
        bp['boxes'][0].set_alpha(0.6)
        bp['medians'][0].set_color('red')
        bp['medians'][0].set_linewidth(2)
        ax3.set_title('Box Plot', fontweight='bold')
        ax3.set_xlabel('Value')
        ax3.grid(True, alpha=0.3, axis='x')
        
        ax4 = fig.add_subplot(gs[1, 1])
        data_sorted = np.sort(data_clean)
        empirical_cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
        ax4.plot(data_sorted, empirical_cdf, color=self.colors['histogram'], 
                linewidth=2, label='Empirical CDF')
        
        theoretical_cdf = norm.cdf(data_sorted, mu, sigma)
        ax4.plot(data_sorted, theoretical_cdf, color=self.colors['normal'], 
                linewidth=2, linestyle='--', label='Normal CDF')
        
        ax4.set_title('Cumulative Distribution', fontweight='bold')
        ax4.set_xlabel('Value')
        ax4.set_ylabel('Cumulative Probability')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        fig.suptitle(f'Distribution Diagnostic: {feature_name}', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        if save:
            filename = self.output_dir / f"{feature_name}_full_diagnostic.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f" Saved: {filename}")
        
        return fig
    
    def plot_categorical_distribution(
        self,
        data: np.ndarray,
        feature_name: str,
        categories: Optional[List] = None,
        title: Optional[str] = None,
        save: bool = True
    ) -> plt.Figure:
        data_clean = data[~pd.isna(data)]
        
        if len(data_clean) == 0:
            print(f" Нет данных для {feature_name}")
            return None
        
        unique, counts = np.unique(data_clean, return_counts=True)
        frequencies = dict(zip(unique, counts))
        
        if categories is not None:
            for cat in categories:
                if cat not in frequencies:
                    frequencies[cat] = 0
        
        sorted_items = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
        labels = [str(item[0]) for item in sorted_items]
        values = [item[1] for item in sorted_items]
        percentages = [v / sum(values) * 100 for v in values]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        colors_palette = sns.color_palette("husl", len(labels))
        bars = ax1.bar(range(len(labels)), values, color=colors_palette, 
                      edgecolor='black', linewidth=1.5, alpha=0.8)
        
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Frequency Distribution', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, val, pct) in enumerate(zip(bars, values, percentages)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=9)
        
        wedges, texts, autotexts = ax2.pie(
            values,
            labels=labels,
            autopct='%1.1f%%',
            colors=colors_palette,
            startangle=90,
            wedgeprops={'edgecolor': 'black', 'linewidth': 1.5}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
        
        ax2.set_title('Proportion Distribution', fontsize=12, fontweight='bold')
        
        if title is None:
            title = f'Categorical Distribution: {feature_name}'
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
        
        stats_text = f'Total: {len(data_clean)}\nCategories: {len(labels)}\nMode: {labels[0]} ({values[0]})'
        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        if save:
            filename = self.output_dir / f"{feature_name}_categorical.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f" Saved: {filename}")
        
        return fig



if __name__ == "__main__":
    print("Testing Distribution Visualizer")
    
    np.random.seed(42)
    
    normal_data = np.random.normal(50, 10, 500)
    
    lognormal_data = np.random.lognormal(2, 0.5, 500)
    
    bimodal_data = np.concatenate([
        np.random.normal(30, 5, 250),
        np.random.normal(70, 5, 250)
    ])
    
    categorical_data = np.random.choice(['A', 'B', 'C', 'D'], size=500, p=[0.4, 0.3, 0.2, 0.1])
    
    visualizer = DistributionVisualizer(output_dir="test_visualizations")
    
    print("\n" + "="*80)
    print("1. HISTOGRAM + KDE")
    print("="*80)
    visualizer.plot_histogram_with_kde(normal_data, "normal_test", save=True)
    visualizer.plot_histogram_with_kde(lognormal_data, "lognormal_test", save=True)
    
    print("\n" + "="*80)
    print("2. Q-Q PLOT")
    print("="*80)
    visualizer.plot_qq_plot(normal_data, "normal_test", save=True)
    visualizer.plot_qq_plot(lognormal_data, "lognormal_test", save=True)
    
    print("\n" + "="*80)
    print("3. P-P PLOT")
    print("="*80)
    visualizer.plot_pp_plot(normal_data, "normal_test", save=True)
    
    print("\n" + "="*80)
    print("4. BOX & Violin plot")
    print("="*80)
    visualizer.plot_boxplot_violin(bimodal_data, "bimodal_test", save=True)
    
    print("\n" + "="*80)
    print("5. Full diagnostic")
    print("="*80)
    visualizer.plot_full_diagnostic(normal_data, "normal_test", save=True)
    
    print("\n" + "="*80)
    print("6. Categorical distribution")
    print("="*80)
    visualizer.plot_categorical_distribution(categorical_data, "categorical_test", save=True)
    
    print("\n" + "="*80)
    print(" Distribution visualizer TEST Готово")
    print(f" Visualizations saved to: test_visualizations/")
    print("="*80)
    
    plt.close('all')
