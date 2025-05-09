import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import argparse
import os
import sys
import scipy
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path

### example usage
#
# python class_analyzer.py /home/jasonkhtam7/Documents/DSGT/scratch/fungiclef/dataset/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Train.csv --output /home/jasonkhtam7/Documents/DSGT/clef/fungiclef-2025/user/tamncheese/eda/


def analyze_class_imbalance(csv_path, target_col="category_id", output_dir=None):
    """
    A simple function to analyze class imbalance in a dataset

    Parameters:
    -----------
    csv_path : str
        Path to the CSV file
    target_col : str
        Column name containing the target classes
    output_dir : str
        Directory to save outputs (if None, just display)
    """
    # Load the dataset
    print(f"Loading dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded with {len(df)} entries and {len(df.columns)} columns")

    # Verify target column exists
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. Available columns: {', '.join(df.columns)}"
        )

    # Count classes
    class_counts = Counter(df[target_col])
    total_samples = len(df)
    num_classes = len(class_counts)

    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Basic statistics
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    imbalance_ratio = max_count / min_count

    # Calculate some metrics about the distribution
    counts = np.array(list(class_counts.values()))
    counts_sorted = np.sort(counts)[::-1]  # Sort in descending order

    # Calculate entropy
    probs = counts / total_samples
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    effective_num_classes = np.exp(entropy)

    # Class percentages
    class_percentages = {
        cls: (count / total_samples * 100) for cls, count in class_counts.items()
    }

    # Identify rare classes (bottom 25%)
    rare_threshold = np.percentile(counts, 25)
    rare_classes = [
        cls for cls, count in class_counts.items() if count <= rare_threshold
    ]

    # Identify common classes (top 25%)
    common_threshold = np.percentile(counts, 75)
    common_classes = [
        cls for cls, count in class_counts.items() if count >= common_threshold
    ]

    # Calculate cumulative distribution
    cum_counts = np.cumsum(counts_sorted)
    cum_percentages = cum_counts / total_samples * 100

    # Print summary
    print("\n===== CLASS IMBALANCE SUMMARY =====")
    print(f"Target column: {target_col}")
    print(f"Total samples: {total_samples}")
    print(f"Number of classes: {num_classes}")
    print(f"Min class size: {min_count} samples")
    print(f"Max class size: {max_count} samples")
    print(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}")
    print(f"Entropy: {entropy:.2f} bits")
    print(
        f"Effective number of classes: {effective_num_classes:.2f} (out of {num_classes})"
    )

    # Count classes with very few samples
    count_1 = sum(1 for count in counts if count == 1)
    count_2 = sum(1 for count in counts if count == 2)
    count_5_or_less = sum(1 for count in counts if count <= 5)

    print(
        f"\nClasses with 1 sample: {count_1} ({count_1/num_classes:.1%} of all classes)"
    )
    print(
        f"Classes with 2 samples: {count_2} ({count_2/num_classes:.1%} of all classes)"
    )
    print(
        f"Classes with ≤5 samples: {count_5_or_less} ({count_5_or_less/num_classes:.1%} of all classes)"
    )

    # Print top and bottom classes
    k = min(5, num_classes)

    print(f"\nTop {k} most common classes:")
    for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[
        :k
    ]:
        pct = class_percentages[cls]
        print(f"  Class {cls}: {count} samples ({pct:.2f}%)")

    print(f"\nBottom {k} rarest classes:")
    for cls, count in sorted(class_counts.items(), key=lambda x: x[1])[:k]:
        pct = class_percentages[cls]
        print(f"  Class {cls}: {count} samples ({pct:.2f}%)")

    # Print cumulative distribution milestones
    print("\nCumulative distribution:")
    for p in [50, 80, 90, 95, 99]:
        classes_needed = np.searchsorted(cum_percentages, p) + 1
        print(
            f"  {classes_needed} classes ({classes_needed/num_classes:.1%}) cover {p}% of samples"
        )

    # Determine imbalance severity
    if imbalance_ratio < 10:
        severity = "Mild"
        recommendation = "Use class weights in your model"
    elif imbalance_ratio < 100:
        severity = "Moderate"
        recommendation = "Use oversampling with augmentation for rare classes"
    else:
        severity = "Severe"
        recommendation = "Use combination of oversampling rare classes and undersampling common classes"

    print(f"\nImbalance severity: {severity}")
    print(f"Recommended approach: {recommendation}")

    # Plot the distribution
    plt.figure(figsize=(12, 6))

    # Sort classes by frequency
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    classes = [str(x[0]) for x in sorted_classes]
    counts = [x[1] for x in sorted_classes]

    # Bar colors
    colors = ["#1f77b4"] * len(classes)
    for i, cls in enumerate(classes):
        if cls in [str(c) for c in rare_classes]:
            colors[i] = "#d62728"  # Red for rare
        elif cls in [str(c) for c in common_classes]:
            colors[i] = "#2ca02c"  # Green for common

    # Plot class distribution
    plt.bar(range(len(classes)), counts, color=colors)
    plt.title("Class Distribution")
    plt.xlabel("Class Rank")
    plt.ylabel("Number of Samples")

    # If there are many classes, don't show all x labels
    if len(classes) > 20:
        plt.xticks([])
        plt.xlabel("Classes (sorted by frequency)")
    else:
        plt.xticks(range(len(classes)), classes, rotation=90)

    # Add a legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2ca02c", label="Common Classes (top 25%)"),
        Patch(facecolor="#1f77b4", label="Medium Classes"),
        Patch(facecolor="#d62728", label="Rare Classes (bottom 25%)"),
    ]
    plt.legend(handles=legend_elements, loc="upper right")

    # Save or display
    if output_dir:
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "class_distribution.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")


class SimpleClassImbalanceAnalyzer:
    """
    A simplified utility to analyze class imbalance in metadata.csv files
    using only matplotlib and scikit-learn
    """

    def __init__(self, csv_path):
        """
        Initialize the analyzer with a CSV metadata file path

        Parameters:
        -----------
        csv_path : str or Path
            Path to the CSV metadata file
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        # Load the CSV file
        self.df = pd.read_csv(self.csv_path)
        print(
            f"Loaded metadata with {len(self.df)} entries and {len(self.df.columns)} columns"
        )
        print(f"Columns: {', '.join(self.df.columns)}")
        print(f"Columns: {', '.join(self.df.columns)}")

    def analyze_class_distribution(self, target_col="category_id"):
        """
        Analyze the distribution of the target class

        Parameters:
        -----------
        target_col : str
            Column name containing the target classes

        Returns:
        --------
        dict
            Dictionary with analysis results
        """
        if target_col not in self.df.columns:
            raise ValueError(f"Target column '{target_col}' not found in CSV")

        self.target_col = target_col

        # Get basic class distribution
        class_counts = Counter(self.df[target_col])
        self.class_counts = class_counts

        # Basic statistics
        total_samples = len(self.df)
        num_classes = len(class_counts)
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        imbalance_ratio = max_count / min_count

        # Calculate class weights using scikit-learn
        # Convert class labels to strings to ensure consistent handling
        y_values = self.df[target_col].astype(str).values
        unique_classes = np.array(sorted(set(y_values)))

        try:
            class_weights = compute_class_weight(
                class_weight="balanced", classes=unique_classes, y=y_values
            )
            self.class_weights = dict(zip(unique_classes, class_weights))
        except Exception as e:
            print(f"Warning: Could not compute class weights: {e}")
            # Fallback to manual calculation
            n_samples = len(y_values)
            n_classes = len(unique_classes)
            weights = {}
            for cls in unique_classes:
                cls_count = np.sum(y_values == cls)
                if cls_count > 0:
                    weights[cls] = n_samples / (n_classes * cls_count)
                else:
                    weights[cls] = 1.0
            self.class_weights = weights

        # Calculate entropy
        counts = np.array(list(class_counts.values()))
        probs = counts / total_samples
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        # Calculate effective number of classes
        effective_num_classes = np.exp(entropy)

        # Organize the results
        self.results = {
            "total_samples": total_samples,
            "num_classes": num_classes,
            "min_count": min_count,
            "max_count": max_count,
            "imbalance_ratio": imbalance_ratio,
            "entropy": entropy,
            "effective_num_classes": effective_num_classes,
            "class_weights": self.class_weights,
        }

        # Identify rare and common classes
        rare_threshold = np.percentile(list(class_counts.values()), 25)
        common_threshold = np.percentile(list(class_counts.values()), 75)

        self.rare_classes = [
            str(cls) for cls, count in class_counts.items() if count <= rare_threshold
        ]
        self.common_classes = [
            str(cls) for cls, count in class_counts.items() if count >= common_threshold
        ]

        self.results["rare_classes"] = self.rare_classes
        self.results["common_classes"] = self.common_classes

        return self.results

    def print_summary(self):
        """Print a summary of the class distribution analysis"""
        if not hasattr(self, "results"):
            raise ValueError(
                "No analysis results. Call analyze_class_distribution first"
            )

        print("\n=== Class Distribution Summary ===")
        print(f"Target column: {self.target_col}")
        print(f"Total samples: {self.results['total_samples']}")
        print(f"Number of classes: {self.results['num_classes']}")

        print("\nClass counts:")
        print(f"  Min: {self.results['min_count']}")
        print(f"  Max: {self.results['max_count']}")

        print(
            f"\nImbalance ratio (majority:minority): {self.results['imbalance_ratio']:.2f}"
        )
        print(f"Entropy: {self.results['entropy']:.4f} bits")
        print(
            f"Effective number of classes: {self.results['effective_num_classes']:.2f}"
        )

        # Print distribution of top and bottom classes
        num_to_show = min(5, len(self.class_counts))

        sorted_classes = sorted(
            self.class_counts.items(), key=lambda x: x[1], reverse=True
        )

        print("\nTop {} classes:".format(num_to_show))
        for class_id, count in sorted_classes[:num_to_show]:
            percentage = 100 * count / self.results["total_samples"]
            print(f"  {class_id}: {count} samples ({percentage:.2f}%)")

        print("\nBottom {} classes:".format(num_to_show))
        for class_id, count in sorted_classes[-num_to_show:]:
            percentage = 100 * count / self.results["total_samples"]
            print(f"  {class_id}: {count} samples ({percentage:.2f}%)")

        print("\nRare classes (bottom 25%):")
        print(f"  {', '.join(str(cls) for cls in self.rare_classes)}")

        print("\nCommon classes (top 25%):")
        print(f"  {', '.join(str(cls) for cls in self.common_classes)}")

        print("\nClass weights (for balanced training):")
        for cls, weight in sorted(self.class_weights.items()):
            print(f"  Class {cls}: {weight:.4f}")

    def plot_class_distribution(self, figsize=(12, 10), save_path=None):
        """
        Plot the class distribution using matplotlib

        Parameters:
        -----------
        figsize : tuple
            Figure size for the plot
        save_path : str
            Path to save the figure. If None, the figure is displayed instead.
        """
        if not hasattr(self, "results"):
            raise ValueError(
                "No analysis results. Call analyze_class_distribution first"
            )

        # Sort classes by frequency
        sorted_classes = sorted(
            self.class_counts.items(), key=lambda x: x[1], reverse=True
        )
        class_ids = [str(x[0]) for x in sorted_classes]
        class_counts = [x[1] for x in sorted_classes]

        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)

        # 1. Bar chart of class counts
        bar_colors = ["#5a9bd5"] * len(class_ids)

        # Highlight rare classes
        for i, class_id in enumerate(class_ids):
            if class_id in self.rare_classes:
                bar_colors[i] = "#ff6666"  # Red for rare classes
            elif class_id in self.common_classes:
                bar_colors[i] = "#66b266"  # Green for common classes

        ax1.bar(range(len(class_ids)), class_counts, color=bar_colors)
        ax1.set_title("Class Distribution")
        ax1.set_xlabel("Class ID")
        ax1.set_ylabel("Number of samples")

        # Set x-ticks for class IDs
        if len(class_ids) <= 20:
            ax1.set_xticks(range(len(class_ids)))
            ax1.set_xticklabels(class_ids, rotation=90)
        else:
            # If too many classes, show only some tick labels
            step = max(1, len(class_ids) // 20)
            ax1.set_xticks(range(0, len(class_ids), step))
            ax1.set_xticklabels(
                [class_ids[i] for i in range(0, len(class_ids), step)], rotation=90
            )

        # Add a legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#5a9bd5", label="Regular Classes"),
            Patch(facecolor="#ff6666", label="Rare Classes (bottom 25%)"),
            Patch(facecolor="#66b266", label="Common Classes (top 25%)"),
        ]
        ax1.legend(handles=legend_elements, loc="upper right")

        # 2. Cumulative distribution (like a Lorenz curve)
        # Sort counts in ascending order
        counts_sorted = sorted(class_counts)
        total = sum(counts_sorted)
        cum_proportions = np.cumsum(counts_sorted) / total

        # Calculate line of perfect equality
        perfect_equality = np.linspace(0, 1, len(counts_sorted))

        # Plot
        ax2.plot([0, 1], [0, 1], "k--", label="Perfect equality")
        ax2.step(
            np.insert(np.arange(len(counts_sorted)) / (len(counts_sorted) - 1), 0, 0),
            np.insert(cum_proportions, 0, 0),
            "b-",
            label="Class distribution",
        )

        # Fill area between curves
        ax2.fill_between(
            np.insert(np.arange(len(counts_sorted)) / (len(counts_sorted) - 1), 0, 0),
            np.insert(perfect_equality, 0, 0),
            np.insert(cum_proportions, 0, 0),
            alpha=0.2,
        )

        ax2.set_title("Cumulative Class Distribution")
        ax2.set_xlabel("Cumulative proportion of classes")
        ax2.set_ylabel("Cumulative proportion of samples")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Class weights visualization
        sorted_weights = [(cls, self.class_weights[cls]) for cls in class_ids]
        weight_values = [x[1] for x in sorted_weights]

        ax3.bar(range(len(class_ids)), weight_values, color="#8075d8")
        ax3.set_title("Class Weights (for balanced training)")
        ax3.set_xlabel("Class ID")
        ax3.set_ylabel("Weight")

        if len(class_ids) <= 20:
            ax3.set_xticks(range(len(class_ids)))
            ax3.set_xticklabels(class_ids, rotation=90)
        else:
            # If too many classes, show only some tick labels
            step = max(1, len(class_ids) // 20)
            ax3.set_xticks(range(0, len(class_ids), step))
            ax3.set_xticklabels(
                [class_ids[i] for i in range(0, len(class_ids), step)], rotation=90
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to {save_path}")

        return fig

    def analyze_correlation_with_features(self, feature_cols=None):
        """
        Analyze correlation between target class and other features

        Parameters:
        -----------
        feature_cols : list
            List of feature columns to analyze. If None, use all columns except target.

        Returns:
        --------
        dict
            Dictionary with correlation results
        """
        if not hasattr(self, "target_col"):
            raise ValueError(
                "Target column not set. Call analyze_class_distribution first"
            )

        # If no features specified, use all except target
        if feature_cols is None:
            feature_cols = [col for col in self.df.columns if col != self.target_col]

        correlations = {}

        for feature in feature_cols:
            if feature not in self.df.columns:
                print(f"Warning: Feature '{feature}' not found in CSV")
                continue

            # Check if categorical or boolean
            if self.df[feature].dtype == "object" or self.df[feature].dtype == bool:
                # For categorical features, use chi-square test
                try:
                    # Create contingency table
                    contingency = pd.crosstab(
                        self.df[feature], self.df[self.target_col]
                    )

                    # Apply chi-square test
                    chi2, p, _, _ = scipy.stats.chi2_contingency(contingency)

                    correlations[feature] = {
                        "type": "categorical",
                        "chi2": chi2,
                        "p_value": p,
                        "significant": p < 0.05,
                    }
                except Exception as e:
                    print(f"Error analyzing categorical feature '{feature}': {e}")

            # For numeric features
            elif np.issubdtype(self.df[feature].dtype, np.number):
                # For numeric features, use ANOVA (F-test)
                try:
                    # Group by target and get values
                    groups = [
                        self.df[self.df[self.target_col] == cls][feature]
                        .dropna()
                        .values
                        for cls in np.unique(self.df[self.target_col])
                    ]

                    # Apply ANOVA
                    f_val, p_val = scipy.stats.f_oneway(*groups)

                    correlations[feature] = {
                        "type": "numeric",
                        "f_value": f_val,
                        "p_value": p_val,
                        "significant": p_val < 0.05,
                    }
                except Exception as e:
                    print(f"Error analyzing numeric feature '{feature}': {e}")

        self.feature_correlations = correlations
        return correlations

    def print_correlation_summary(self):
        """Print a summary of feature correlations with the target"""
        if not hasattr(self, "feature_correlations"):
            raise ValueError(
                "No correlation results. Call analyze_correlation_with_features first"
            )

        print("\n=== Feature Correlation Summary ===")
        print(f"Target column: {self.target_col}")

        significant_features = []

        for feature, result in self.feature_correlations.items():
            significance = "SIGNIFICANT" if result["significant"] else "not significant"

            if result["type"] == "categorical":
                print(f"\n{feature} (categorical): {significance}")
                print(
                    f"  Chi-square: {result['chi2']:.2f}, p-value: {result['p_value']:.6f}"
                )
            else:  # numeric
                print(f"\n{feature} (numeric): {significance}")
                print(
                    f"  F-value: {result['f_value']:.2f}, p-value: {result['p_value']:.6f}"
                )

            if result["significant"]:
                significant_features.append(feature)

        print("\nSignificant features:")
        if significant_features:
            for feature in significant_features:
                print(f"  - {feature}")
        else:
            print("  None found")

    def generate_recommendations(self):
        """
        Generate recommendations for addressing class imbalance

        Returns:
        --------
        dict
            Dictionary with recommendations
        """
        if not hasattr(self, "results"):
            raise ValueError(
                "No analysis results. Call analyze_class_distribution first"
            )

        imbalance_ratio = self.results["imbalance_ratio"]

        # Determine severity of imbalance
        if imbalance_ratio < 3:
            imbalance_severity = "Mild"
            sampling_approach = "Class weights or mild oversampling"
        elif imbalance_ratio < 10:
            imbalance_severity = "Moderate"
            sampling_approach = "Oversampling with augmentation for minority classes"
        else:
            imbalance_severity = "Severe"
            sampling_approach = "Combination of oversampling minority classes and undersampling majority classes"

        recommendations = {
            "imbalance_severity": imbalance_severity,
            "sampling_approach": sampling_approach,
            "specific_recommendations": [],
        }

        # Add specific recommendations based on imbalance severity
        if imbalance_severity == "Mild":
            recommendations["recommended_approach"] = (
                "Use scikit-learn's class_weight='balanced' parameter"
            )
            recommendations["specific_recommendations"].extend(
                [
                    "Use class weights in your model training (provided in the analysis)",
                    "Focus on proper cross-validation (stratified) to ensure all classes are represented",
                    "Consider mild data augmentation only for the rarest classes",
                ]
            )
        elif imbalance_severity == "Moderate":
            recommendations["recommended_approach"] = (
                "Oversample rare classes with augmentation"
            )
            recommendations["specific_recommendations"].extend(
                [
                    "Apply moderate augmentation (flipping, rotation) to rare classes",
                    "Use the class weights as a reference for how much to augment each class",
                    "For images, crop/flip/rotate rare class images to increase their representation",
                    "Set validation data aside before augmentation to prevent data leakage",
                ]
            )
        else:  # Severe
            recommendations["recommended_approach"] = (
                "Combined oversampling and undersampling"
            )
            recommendations["specific_recommendations"].extend(
                [
                    "Oversample rare classes aggressively with multiple augmentations per image",
                    "Apply combinations of transformations (e.g., rotate AND flip) for maximum diversity",
                    "Consider undersampling very common classes to prevent them from dominating",
                    "Use cluster-based undersampling for majority classes to maintain diversity",
                ]
            )

        # Check for special features
        has_genus = "genus" in self.df.columns
        has_poisonous = "poisonous" in self.df.columns

        if has_genus:
            recommendations["special_considerations"] = recommendations.get(
                "special_considerations", []
            )
            recommendations["special_considerations"].append(
                "Taxonomic data present: Ensure augmentation respects taxonomic relationships. "
                "Images from the same genus/species should receive similar augmentation strategies."
            )

        if has_poisonous:
            recommendations["special_considerations"] = recommendations.get(
                "special_considerations", []
            )
            recommendations["special_considerations"].append(
                "Binary feature 'poisonous' present: Analyze if this correlates with target classes. "
                "Ensure balancing doesn't create artificial correlations between poisonous status and classes."
            )

        # Add 10x expansion recommendations
        recommendations["expansion_recommendations"] = []

        if imbalance_severity == "Mild":
            recommendations["expansion_recommendations"].append(
                "For 10x expansion: Augment all classes proportionally to their original distribution, "
                "with slightly more augmentation for rarer classes."
            )
        elif imbalance_severity == "Moderate":
            recommendations["expansion_recommendations"].append(
                "For 10x expansion: Use square root balancing - augment classes proportionally to "
                "the square root of their original frequencies. This moderates imbalances while "
                "preserving some of the natural distribution."
            )
        else:  # Severe
            recommendations["expansion_recommendations"].append(
                "For 10x expansion: First balance to reduce severe imbalance, then expand. "
                "Aim for no class to have less than 1/4 the samples of the most common class "
                "after augmentation, which may require aggressive augmentation of rare classes."
            )

        print("\n=== Recommendations ===")
        print(f"Imbalance severity: {recommendations['imbalance_severity']}")
        print(f"Recommended approach: {recommendations['sampling_approach']}")
        print(f"Specific recommendation: {recommendations['recommended_approach']}")

        print("\nSpecific recommendations:")
        for i, rec in enumerate(recommendations["specific_recommendations"], 1):
            print(f"{i}. {rec}")

        if "special_considerations" in recommendations:
            print("\nSpecial considerations:")
            for i, rec in enumerate(recommendations["special_considerations"], 1):
                print(f"{i}. {rec}")

        print("\nRecommendations for 10x expansion:")
        for i, rec in enumerate(recommendations["expansion_recommendations"], 1):
            print(f"{i}. {rec}")

        return recommendations

    def calculate_target_distribution(self, strategy="sqrt", target_expansion=10):
        """
        Calculate target distribution for rebalancing and expansion

        Parameters:
        -----------
        strategy : str
            'balanced': Equal samples per class
            'sqrt': Square root balancing
            'proportional': Keep original proportions
            'binary': Two-tier balancing (common vs rare)
        target_expansion : float
            Target dataset expansion factor

        Returns:
        --------
        dict
            Dictionary with target counts per class
        """
        if not hasattr(self, "results"):
            raise ValueError(
                "No analysis results. Call analyze_class_distribution first"
            )

        # Get original counts
        original_counts = np.array(
            [self.class_counts[cls] for cls in sorted(self.class_counts.keys())]
        )
        classes = np.array(sorted(self.class_counts.keys()))

        # Calculate original total
        original_total = sum(original_counts)
        target_total = original_total * target_expansion

        # Apply strategy
        if strategy == "balanced":
            # Equal number of samples per class
            target_counts = np.ones_like(original_counts) * (
                target_total / len(original_counts)
            )

        elif strategy == "sqrt":
            # Square root balancing
            sqrt_counts = np.sqrt(original_counts)
            scaling_factor = target_total / np.sum(sqrt_counts)
            target_counts = sqrt_counts * scaling_factor

        elif strategy == "proportional":
            # Keep original proportions
            target_counts = original_counts * target_expansion

        elif strategy == "binary":
            # Two-tier: boost rare classes more than common ones
            threshold = np.percentile(original_counts, 50)  # Median as threshold
            binary_weights = np.where(original_counts <= threshold, 2, 1)
            weighted_counts = original_counts * binary_weights
            scaling_factor = target_total / np.sum(weighted_counts)
            target_counts = weighted_counts * scaling_factor

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Round to integers
        target_counts = np.round(target_counts).astype(int)

        # Calculate per-class expansion
        expansion_factors = target_counts / original_counts

        # Create result dictionary
        result = {
            "strategy": strategy,
            "target_expansion": target_expansion,
            "original_total": original_total,
            "target_total": int(np.sum(target_counts)),
            "actual_expansion": np.sum(target_counts) / original_total,
            "class_distribution": dict(zip(classes, target_counts)),
            "expansion_factors": dict(zip(classes, expansion_factors)),
        }

        return result


# Example usage
if __name__ == "__main__":
    import argparse

    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Analyze class imbalance in image metadata CSV files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument("csv_path", type=str, help="Path to the metadata CSV file")

    # Optional arguments
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        default="category_id",
        help="Target column name for classification",
    )
    parser.add_argument(
        "--features",
        "-f",
        type=str,
        nargs="+",
        help="Specific feature columns to analyze (default: all columns)",
    )
    parser.add_argument(
        "--expansion",
        "-e",
        type=float,
        default=10.0,
        help="Target expansion factor for dataset",
    )
    parser.add_argument(
        "--strategy",
        "-s",
        type=str,
        default="sqrt",
        choices=["balanced", "sqrt", "proportional", "binary"],
        help="Balancing strategy for expansion",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Path to save the distribution plot (default: display only)",
    )
    parser.add_argument(
        "--text-output",
        "-to",
        type=str,
        help="Path to save text analysis results (default: print to stdout)",
    )
    parser.add_argument(
        "--json-output", "-jo", type=str, help="Path to save analysis results as JSON"
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Skip plotting (useful for script mode)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Create a list to capture output if needed
    original_stdout = sys.stdout

    # Begin execution
    print(f"Analyzing class imbalance in: {args.csv_path}")

    # Redirect stdout if text output is specified
    if args.text_output:
        print(f"Results will be saved to: {args.text_output}")
        sys.stdout = open(args.text_output, "w")

    try:
        # Initialize analyzer
        analyzer = SimpleClassImbalanceAnalyzer(args.csv_path)

        # Analyze the target column distribution
        analyzer.analyze_class_distribution(target_col=args.target)

        # Print summary statistics
        analyzer.print_summary()

        # Analyze correlation with features
        if args.features:
            print(
                f"\nAnalyzing correlation with specified features: {', '.join(args.features)}"
            )
            analyzer.analyze_correlation_with_features(feature_cols=args.features)
        else:
            print("\nAnalyzing correlation with all available features:")
            analyzer.analyze_correlation_with_features()

        analyzer.print_correlation_summary()

        # Get recommendations for addressing class imbalance
        recommendations = analyzer.generate_recommendations()

        # Calculate target distribution for expansion
        target_dist = analyzer.calculate_target_distribution(
            strategy=args.strategy, target_expansion=args.expansion
        )

        print(
            f"\nTarget distribution for {args.expansion}x expansion ({args.strategy} strategy):"
        )
        print(f"Original total: {target_dist['original_total']} samples")
        print(f"Target total: {target_dist['target_total']} samples")
        print(f"Actual expansion: {target_dist['actual_expansion']:.2f}x")

        # Plot class distribution if not disabled
        if not args.no_plot:
            fig = analyzer.plot_class_distribution()

            if args.output:
                plt.savefig(args.output, dpi=300, bbox_inches="tight")
                print(f"Plot saved to: {args.output}")
            else:
                # Only show if we're not redirecting to file
                if not args.text_output:
                    plt.show()
                else:
                    plt.close(fig)

    except Exception as e:
        # Print the error
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        parser.print_help()
        exit(1)

    finally:
        # Restore stdout if redirected
        if args.text_output:
            sys.stdout.close()
            sys.stdout = original_stdout
            print(f"Text results saved to: {args.text_output}")

    try:
        # Initialize analyzer
        analyzer = SimpleClassImbalanceAnalyzer(args.csv_path)

        # Analyze the target column distribution
        analyzer.analyze_class_distribution(target_col=args.target)

        # Print summary statistics
        analyzer.print_summary()

        # Analyze correlation with features
        if args.features:
            print(
                f"\nAnalyzing correlation with specified features: {', '.join(args.features)}"
            )
            analyzer.analyze_correlation_with_features(feature_cols=args.features)
        else:
            print("\nAnalyzing correlation with all available features:")
            analyzer.analyze_correlation_with_features()

        analyzer.print_correlation_summary()

        # Get recommendations for addressing class imbalance
        recommendations = analyzer.generate_recommendations()

        # Calculate target distribution for expansion
        target_dist = analyzer.calculate_target_distribution(
            strategy=args.strategy, target_expansion=args.expansion
        )

        print(
            f"\nTarget distribution for {args.expansion}x expansion ({args.strategy} strategy):"
        )
        print(f"Original total: {target_dist['original_total']} samples")
        print(f"Target total: {target_dist['target_total']} samples")
        print(f"Actual expansion: {target_dist['actual_expansion']:.2f}x")

        # Plot class distribution if not disabled
        if not args.no_plot:
            fig = analyzer.plot_class_distribution()

            if args.output:
                plt.savefig(args.output, dpi=300, bbox_inches="tight")
                print(f"Plot saved to: {args.output}")
            else:
                # Only show if we're not redirecting to file
                if not args.text_output:
                    plt.show()
                else:
                    plt.close(fig)

        # Restore stdout if redirected
        if args.text_output:
            sys.stdout.close()
            sys.stdout = original_stdout
            print(f"Text results saved to: {args.text_output}")

    except Exception as e:
        # Restore stdout if exception occurs
        if args.text_output and sys.stdout != original_stdout:
            sys.stdout.close()
            sys.stdout = original_stdout

        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        parser.print_help()
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Class Imbalance Analyzer")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file")
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        default="category_id",
        help="Target column name (default: category_id)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory for results (default: None, just display)",
    )

    args = parser.parse_args()
    analyze_class_imbalance(args.csv_path, args.target, args.output)
