import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def visualize_distribution(data, columns, bins=30, save_to_file=None):
    """
    Visualize the distribution of data for each specified column.
    For numeric columns, include histogram, KDE, Q-Q plot, and boxplot.
    For categorical columns, include count plot and percentage distribution.

    Parameters:
    - data (pd.DataFrame): Input data.
    - columns (list of str): List of column names to visualize.
    - bins (int): Number of bins for histogram (numeric only).
    - save_to_file (str or None): File path to save the plots (if specified).
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")

    for column_name in columns:
        if column_name not in data.columns:
            print(f"Column '{column_name}' not found in DataFrame.")
            continue

        values = data[column_name].dropna()

        # Check if the column is categorical or numeric
        if pd.api.types.is_numeric_dtype(values):
            # Numeric distribution visualization
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            sns.histplot(values, bins=bins, kde=True, ax=axs[0], color='skyblue')
            axs[0].set_title(f'Distribution of {column_name}')
            axs[0].set_xlabel(column_name)
            axs[0].set_ylabel('Frequency')

            stats.probplot(values, dist="norm", plot=axs[1])
            axs[1].set_title('Q-Q Plot')

            if len(values) <= 5000:
                shapiro_test = stats.shapiro(values)
                p_value = shapiro_test.pvalue
                normal = "Yes" if p_value > 0.05 else "No"
                test_name = "Shapiro-Wilk"
            else:
                ad_test = stats.anderson(values, dist='norm')
                p_value = ad_test.significance_level[np.argmax(ad_test.statistic < ad_test.critical_values)]
                normal = "Yes" if ad_test.statistic < ad_test.critical_values[-1] else "No"
                test_name = "Anderson-Darling"

            sns.boxplot(x=values, ax=axs[2], color='lightcoral')
            axs[2].set_title(f'Boxplot of {column_name}')

            plt.suptitle(
                f'Distribution Analysis of {column_name}\nNormal Distribution: {normal} ({test_name} p = {p_value:.4f})')
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)

        else:
            # Categorical distribution visualization
            fig, ax = plt.subplots(1, 2, figsize=(18, 6))

            # Define a consistent color palette for the unique categories
            unique_categories = values.unique()
            palette = sns.color_palette("viridis", len(unique_categories))
            color_dict = dict(zip(unique_categories, palette))
            order = sorted(unique_categories)  # Define order for consistent sorting

            # Count plot for frequencies with hue and legend disabled
            sns.countplot(x=values, ax=ax[0], hue=values, palette=color_dict, order=order, dodge=False, legend=False)
            ax[0].set_title(f'Count of Categories in {column_name}')
            ax[0].set_xlabel(column_name)
            ax[0].set_ylabel('Count')

            # Adding value labels on the count plot
            for p in ax[0].patches:
                ax[0].annotate(f'{int(p.get_height())}',
                               (p.get_x() + p.get_width() / 2., p.get_height()),
                               ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 5),
                               textcoords='offset points')

            # Percentage plot with consistent colors, using `hue` to ensure order
            value_counts = values.value_counts(normalize=True).reindex(order) * 100
            sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax[1], hue=value_counts.index,
                        palette=color_dict, dodge=False, legend=False)
            ax[1].set_title(f'Percentage Distribution of {column_name}')
            ax[1].set_xlabel(column_name)
            ax[1].set_ylabel('Percentage (%)')

            # Adding percentage labels on the percentage plot
            for p in ax[1].patches:
                ax[1].annotate(f'{p.get_height():.1f}%',
                               (p.get_x() + p.get_width() / 2., p.get_height()),
                               ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 5),
                               textcoords='offset points')

            plt.suptitle(f'Distribution Analysis of Categorical Column: {column_name}')
            plt.tight_layout()

        # Save or show plot
        if save_to_file:
            name = f"visualize_distribution_{column_name}.png"
            plt.savefig(f"{save_to_file}/{name}")
            print(f"Plot for '{column_name}' saved as {name} in {save_to_file}.")
        else:
            plt.show()
        plt.close(fig)


def feature_importance(
        X,
        y,
        num_random_features=5,
        num_models=3,
        random_state=42,
        top_n=20,
        verbose=True
):
    """
    Assess feature importance and model performance as the number of features increases.

    This function trains multiple XGBoost models with different parameters, adds random noise features,
    and aggregates feature importances based on their ranking across models. Additionally, it assigns
    points to features based on how many random features they rank above, enhancing the robustness of
    the feature importance assessment.

    Parameters:
    - X: pandas DataFrame, feature set.
    - y: pandas Series or array-like, target variable.
    - num_random_features: int, number of random (noise) features to add (default=5).
    - num_models: int, number of XGBoost models to train with different parameters for aggregation and mean Gini calculation (default=3).
    - random_state: int, seed for reproducibility (default=42).
    - top_n: int, number of top features to display in the feature ranking plot (default=20).
    - verbose: bool, whether to print progress messages (default=True).

    Returns:
    - None. Displays two plots:
        1. Mean Gini vs. Number of Features
        2. Aggregated Feature Ranking
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)
    random.seed(random_state)

    # 1. Add Random Features
    X_augmented = X.copy()
    random_feature_names = []
    for i in range(num_random_features):
        rand_feature = f"rand_feat_{i + 1}"
        X_augmented[rand_feature] = np.random.randn(X.shape[0])
        random_feature_names.append(rand_feature)

    if verbose:
        print(f"Added {num_random_features} random features: {random_feature_names}")

    # Encode categorical variables if any
    X_encoded = pd.get_dummies(X_augmented, drop_first=True)

    if verbose:
        print("Encoded categorical variables (if any).")
        print(f"Total features after encoding: {X_encoded.shape[1]}")

    # 2. Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.3, random_state=random_state, stratify=y
    )

    if verbose:
        print("\nData split into train and test sets.")
        print(f"Training set size: {X_train.shape[0]} samples")
        print(f"Test set size: {X_test.shape[0]} samples")

    # 3. Initialize Models with Different Parameters
    model_params_list = [
        {'objective': 'binary:logistic', 'eval_metric': 'auc', 'seed': random_state},
        {'objective': 'binary:logistic', 'eval_metric': 'auc', 'seed': random_state + 1, 'max_depth': 4,
         'learning_rate': 0.1},
        {'objective': 'binary:logistic', 'eval_metric': 'auc', 'seed': random_state + 2, 'max_depth': 6,
         'learning_rate': 0.05}
    ]

    # Ensure we have enough parameter sets
    if num_models > len(model_params_list):
        # Extend model_params_list by varying existing params
        additional_params = []
        for i in range(num_models - len(model_params_list)):
            params = model_params_list[i % len(model_params_list)].copy()
            params['seed'] += i + 3
            additional_params.append(params)
        model_params_list.extend(additional_params)

    models = []
    for params in model_params_list[:num_models]:
        model = xgb.XGBClassifier(**params, use_label_encoder=False, verbosity=0)
        models.append(model)

    if verbose:
        print(f"\nInitialized {num_models} XGBoost models with different parameters.")

    # 4. Train Models and Collect Feature Importances
    feature_importance_dict = {feature: 0 for feature in X_encoded.columns}
    feature_rank_points = {feature: 0 for feature in X_encoded.columns}
    feature_random_points = {feature: 0 for feature in X_encoded.columns}
    gini_scores = []

    if verbose:
        print("\nTraining models and aggregating feature importances...")

    for idx, model in enumerate(tqdm(models, disable=not verbose), 1):
        if verbose:
            print(f"\nTraining model {idx}...")
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        gini = 2 * auc - 1
        gini_scores.append(gini)
        if verbose:
            print(f"Model {idx} AUC: {auc:.4f}, Gini: {gini:.4f}")

        # Get feature importances
        importance = model.feature_importances_
        feature_importances = pd.Series(importance, index=X_encoded.columns)
        feature_importances = feature_importances.sort_values(ascending=False)

        # Identify the ranking order
        ranking = feature_importances.index.tolist()

        # 4.a. Update feature importance points (presence)
        for feature in ranking:
            feature_importance_dict[feature] += 1  # Point for being in ranking

        # 4.b. Update feature rank points based on position
        for rank, feature in enumerate(ranking, start=1):
            # Higher rank (lower number) gets more points
            # Assign (total_features - rank + 1) points
            feature_rank_points[feature] += (len(X_encoded.columns) - rank + 1)

        # 4.c. Update feature random points based on ranking above random features
        for feature in X_encoded.columns:
            if feature in random_feature_names:
                continue  # Skip random features themselves
            # Determine how many random features this feature ranks above
            feature_position = ranking.index(feature)
            num_random_below = 0
            for rand_feat in random_feature_names:
                if rand_feat in ranking:
                    rand_position = ranking.index(rand_feat)
                    if feature_position < rand_position:
                        num_random_below += 1
            feature_random_points[feature] += num_random_below  # Points based on random features below

        if verbose:
            print(f"Model {idx} feature importances recorded.")

    # 5. Aggregate Feature Importances
    aggregated_importance = {}
    for feature in X_encoded.columns:
        # Combine presence points, rank-based points, and random benchmark points
        aggregated_importance[feature] = (
                feature_importance_dict[feature] +
                feature_rank_points[feature] +
                feature_random_points[feature]
        )

    aggregated_importance_series = pd.Series(aggregated_importance).sort_values(ascending=False)

    # 6. Determine Feature Selection Order
    selected_features = aggregated_importance_series.index.tolist()

    # 7. Evaluate Performance vs. Number of Features (Mean Gini from multiple models)
    cumulative_features = []
    cumulative_gini = []

    if verbose:
        print("\nEvaluating performance vs. number of features...")

    for n in tqdm(range(1, len(selected_features) + 1), disable=not verbose):
        features_subset = selected_features[:n]
        X_train_subset = X_train[features_subset]
        X_test_subset = X_test[features_subset]

        # To compute mean Gini, we'll train 'num_models' models with different seeds
        gini_subset_list = []
        for m in range(num_models):
            # Initialize model with different seeds for diversity
            params = model_params_list[m % len(model_params_list)].copy()
            params['seed'] = random_state + m + 3  # Different seed
            model_subset = xgb.XGBClassifier(**params, use_label_encoder=False, verbosity=0)
            model_subset.fit(X_train_subset, y_train)
            y_pred_subset = model_subset.predict_proba(X_test_subset)[:, 1]
            auc_subset = roc_auc_score(y_test, y_pred_subset)
            gini_subset = 2 * auc_subset - 1
            gini_subset_list.append(gini_subset)

        mean_gini = np.mean(gini_subset_list)
        cumulative_features.append(n)
        cumulative_gini.append(mean_gini)

    # 8. Plot Mean Gini vs. Number of Features
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=cumulative_features, y=cumulative_gini, marker='o')
    plt.title('Mean Gini Coefficient vs. Number of Features')
    plt.xlabel('Number of Features')
    plt.ylabel('Mean Gini Coefficient')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 9. Plot Feature Ranking
    plt.figure(figsize=(12, 8))
    sns.barplot(
        x=aggregated_importance_series.head(top_n),
        y=aggregated_importance_series.head(top_n).index,
        palette='viridis',
        hue=aggregated_importance_series.head(top_n).index
    )
    plt.title(f'Top {top_n} Feature Rankings')
    plt.xlabel('Aggregated Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()

    if verbose:
        print("\nFeature importance analysis completed.")

    return aggregated_importance_series
