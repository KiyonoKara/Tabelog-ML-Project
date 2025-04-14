import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from util import print_debug
import urllib.request
import http.client
from bs4 import BeautifulSoup
import re
import time
import os
import streamlit as st
import base64

# Tabelog Collaborative Filtering Application
# Collaborative Filtering and Predicting Ratings for Tabelog Restaurants

class TabelogReviewScraper:
    """
    Tabelog scraper that focuses on collecting reviews/ratings from restaurants
    """

    def __init__(self, restaurant_url, max_pages=None, debug=False):
        """
        Initialize scraper for a specific restaurant
        :param restaurant_url: URL of the restaurant page
        :param max_pages: Maximum number of review pages to scrape (default: None = all pages)
        """
        # Info about accessing the restaurant / store data
        self.restaurant_url = restaurant_url
        self.max_pages = max_pages

        # Restaurant info
        self.store_id = ""
        self.store_name = ""
        self.review_count = 0

        # User count
        self.user_count = 0

        # For debugging
        self.debug = debug
        self.page_num = 0

        # DataFrame for reviews with sub-ratings
        self.columns = [
            "user_id", "overall_rating", "food", "service", "atmosphere", "price", "drink"
        ]
        self.reviews_df = pd.DataFrame(columns=self.columns)

        print_debug(f"Starting scrape of restaurant reviews at {restaurant_url}", debug=self.debug)
        self.scrape_restaurant()
        print_debug(f"Scraping complete. Collected {len(self.reviews_df)} reviews", debug=self.debug)

    def scrape_restaurant(self):
        """
        Scrape the main restaurant page to get name and review link
        """
        try:
            with urllib.request.urlopen(self.restaurant_url) as r:
                content = r.read()
                status_code = r.status

            if status_code != http.client.OK:
                print_debug(f"Error: Could not access {self.restaurant_url}", debug=self.debug)
                return

            # Parse with soup
            soup = BeautifulSoup(content, "html.parser")

            # Extract store ID from URL
            self.store_id = self.restaurant_url.split("/")[-2]

            # Get restaurant name
            store_name_tag = soup.find("h2", class_="display-name")
            if not store_name_tag or not store_name_tag.span:
                print_debug(f"Error: Cannot find restaurant name at {self.restaurant_url}", debug=self.debug)
                # Fallback to ID if name not found
                self.store_name = self.store_id
                return

            self.store_name = store_name_tag.span.string.strip()
            print_debug(f"Restaurant: {self.store_name} (ID: {self.store_id})", debug=self.debug)

            # Get review link and count
            review_tag_id = soup.find("li", id="rdnavi-review")
            if not review_tag_id or not review_tag_id.a:
                print_debug("ロコミのページが見つかりません", debug=self.debug)
                return

            review_tag = review_tag_id.a.get("href")

            # Get review count
            self.review_count = 0
            review_count_span = review_tag_id.find("span", class_="rstdtl-navi__total-count")
            if review_count_span and review_count_span.em:
                self.review_count = int(review_count_span.em.string)
                print_debug(f"Total reviews: {self.review_count}", debug=self.debug)
            else:
                print_debug("Review count not found, will scrape", debug=self.debug)

            # Calculate pages based on review count (20 reviews per page)
            if self.review_count > 0:
                estimated_pages = -((self.review_count + 19) // -20)
                print_debug(f"Estimated number of pages for {self.store_name}: {estimated_pages - 1}〜{estimated_pages}",
                            debug=self.debug)

            # Start scraping review pages
            self.page_num = 1
            last_page = False

            while not last_page:
                # Check if the maximum pages limit is reached
                if self.max_pages and self.page_num > self.max_pages:
                    print_debug(f"Reached maximum page limit ({self.max_pages})", debug=self.debug)
                    break

                print_debug(f"ロコミ{self.page_num}ページ目が加工されています", debug=self.debug)
                review_url = review_tag + f"COND-0/smp1/?lc=0&rvw_part=all&PG={self.page_num}"

                # Scrape the current page
                result = self.scrape_review_page(review_url)

                if not result:
                    print_debug(f"Error accessing page {self.page_num} or no reviews found", debug=self.debug)
                    break

                soup, reviews_found = result

                if not reviews_found:
                    print_debug(f"No reviews found on page {self.page_num}", debug=self.debug)
                    break

                # Check for pagination to determine if there are more pages
                pagination = soup.find("div", class_="c-pagination")

                # If there's no pagination element at all, and we're on page 1,
                # this means there's only one page of reviews
                if not pagination and self.page_num == 1:
                    print_debug("Only one page of reviews found", debug=self.debug)
                    last_page = True
                elif pagination:
                    # Look for the "next" button
                    next_button = pagination.find("a", class_="c-pagination__arrow--next")

                    # If there's no next button, or if the current page is the last one,
                    # we've reached the end
                    if not next_button:
                        print_debug(f"Reached the last page of reviews ({self.page_num})", debug=self.debug)
                        last_page = True
                else:
                    # No pagination found after page 1, must be the last page
                    print_debug(f"No more pages found after page {self.page_num}", debug=self.debug)
                    last_page = True

                # Move to the next page
                if not last_page:
                    self.page_num += 1
                    # Pause between pages to avoid rate limiting
                    time.sleep(2)

        except Exception as e:
            print_debug(f"Error scraping restaurant: {e}", debug=self.debug)

    def scrape_review_page(self, review_url):
        """
        Scrape a single page of reviews to extract ratings
        Returns tuple of (soup, reviews_found) or None if error
        """
        try:
            # Add delay to prevent ratelimits
            time.sleep(1)

            # Request
            with urllib.request.urlopen(review_url) as r:
                content = r.read()
                status_code = r.status

            if status_code != http.client.OK:
                print_debug(f"Error: Could not access {review_url}", debug=self.debug)
                return None

            # Access contents of page
            soup = BeautifulSoup(content, "html.parser")
            review_items = soup.find_all("div", class_="rvw-item")

            # If there are no items to review
            if len(review_items) == 0:
                print_debug("ロコミが見つかりません", debug=self.debug)
                return soup, False

            # Number of reviews in the page
            print_debug(f"{self.page_num}ページ目にレビュー{len(review_items)}件", debug=self.debug)

            # Parse all ratings and add them to DF
            for review_item in review_items:
                self.parse_review_ratings(review_item)

            return soup, True

        except Exception as e:
            print(f"Error scraping review page: {e}")
            return None

    def parse_review_ratings(self, review_item):
        """
        Parse only the ratings from an individual review
        Uses sequential user IDs
        """
        try:
            # Initialize ratings
            # None (will be converted to NaN in DataFrame)
            overall_rating = None
            food_rating = None
            service_rating = None
            # The atmosphere of the place
            atmosphere_rating = None
            # CP (Cost Performance)
            price_rating = None
            # Rating of the drinks
            drink_rating = None

            # Increment user ID
            self.user_count += 1

            # Get overall rating
            rating_elem = review_item.find(["p", "div"], class_="c-rating-v3--xl")
            if rating_elem:
                val_elem = rating_elem.find("b", class_="c-rating-v3__val")
                if val_elem:
                    try:
                        overall_rating = float(val_elem.text.strip())
                    except ValueError:
                        pass

            # Get detailed ratings
            rating_detail = review_item.find("ul", class_="c-rating-detail")
            if rating_detail:
                rating_items = rating_detail.find_all("li", class_="c-rating-detail__item")
                for item in rating_items:
                    label = item.find("span")
                    value = item.find("strong")
                    if label and value:
                        # Get label and rating values
                        label_text = label.text.strip()
                        value_text = value.text.strip()

                        try:
                            rating_value = float(value_text)
                            # Extract all ratings
                            if "料理・味" in label_text:
                                food_rating = rating_value
                            elif "サービス" in label_text:
                                service_rating = rating_value
                            elif "雰囲気" in label_text:
                                atmosphere_rating = rating_value
                            elif "CP" in label_text:
                                price_rating = rating_value
                            elif "酒・ドリンク" in label_text:
                                drink_rating = rating_value
                        except ValueError:
                            # Skip if conversion to float somehow fails
                            pass

            # Add to DataFrame and only add if at least one rating is present
            ratings = [overall_rating, food_rating, service_rating, atmosphere_rating, price_rating, drink_rating]
            if any(type(r) == float for r in ratings):
                # Series with user id of 6 digits
                se = pd.Series([
                    str(self.user_count).zfill(6),
                    *ratings
                ], self.columns)
                r_df = pd.DataFrame([se], columns=self.columns)
                for col in self.columns:
                    if col in self.reviews_df.columns:
                        r_df[col] = r_df[col].astype(self.reviews_df[col].dtype)
                # Add the row to the DF
                self.reviews_df = pd.concat([self.reviews_df, r_df], ignore_index=True)
        except Exception as e:
            print(f"Error parsing review: {e}")

    def save_data(self, directory="."):
        """
        Save the ratings data to a CSV file
        """
        # Remove problematic characters
        store_name = re.sub(r'[\\/*?:"<>|.]', "", self.store_name)
        store_name = store_name.replace(" ", "_")

        filename = os.path.join(directory, f"tabelog_{store_name}_review_data.csv")
        self.reviews_df.to_csv(filename, encoding="utf-8-sig", index=False)
        print_debug(f"Review ratings for {self.store_name} saved to {filename}", debug=self.debug)
        print_debug(f"Collected {len(self.reviews_df)} reviews with {self.user_count} unique users", debug=self.debug)
        return filename

    def get_store_info(self):
        return self.store_name, self.restaurant_url

    def get_reviews_df(self):
        """
        Get the review DataFrame
        :return: The reviews DataFrame
        """
        return self.reviews_df

def cosine_similarity(i1: np.array, i2: np.array):
    """
    Calculate the cosine similarity between two items
    :param i1: Item 1
    :param i2: Item 2
    :return: Cosine similarity (0, 1)
    """
    # Create condition where either i1 or i2 has a value that isn't NaN
    shared = ~np.isnan(i1) & ~np.isnan(i2)

    # Items with only shared values
    i1_common, i2_common = i1[shared], i2[shared]

    # Calculate dot product and norms
    dot_product = np.dot(i1_common, i2_common)
    norm_i1 = np.sqrt(np.sum(i1_common ** 2))
    norm_i2 = np.sqrt(np.sum(i2_common ** 2))

    # In case division by zero happens
    if norm_i1 == 0 or norm_i2 == 0:
        return 0

    return dot_product / (norm_i1 * norm_i2)

def minmax_scale(x):
    """MinMax scale a row or column"""
    min_val = np.nanmin(x)
    max_val = np.nanmax(x)
    if max_val == min_val:
        return np.zeros_like(x)
    return (x - min_val) / (max_val - min_val)


class TabelogCollaborativeFilter:
    def __init__(self, ratings: np.array, category_names, user_ids: np.array, k: int = 4, debug: bool = False):
        """
        Constructor for TabelogCollaborativeFilter
        :param ratings: Ratings of the items from the users only
        :param category_names: Category names (items)
        :param user_ids: User IDs
        :param k: Number of closest items to consider
        :param debug: Debug toggle
        """
        self.ratings = ratings
        self.category_names = category_names
        self.user_ids = user_ids
        self.k = k
        self.debug = debug
        self.__scaled_sim_scores = []
        self.__item_means = []
        self.__centered_ratings = np.full_like(ratings, np.nan)
        self.item_similarities = []

    def get_ssc(self):
        return self.__scaled_sim_scores

    def predict_and_impute(self) -> (pd.DataFrame, dict):
        # Run item-item CF
        self.__item_item_cf(self.ratings, self.category_names)
        # scaled_sim_scores = self.__item_item_cf(self.ratings, self.category_names)
        pred_ratings = self.predict_all_ratings()
        # Create a DataFrame with the predicted and imputed values
        imputed_df = pd.DataFrame(pred_ratings, columns=self.category_names)
        # Re-add user ids back
        imputed_df.insert(0, 'user_id', self.user_ids)

        print_debug("\nImputed DataFrame preview:", debug=self.debug)
        print_debug(imputed_df.head(), debug=self.debug)

        # Calculate mean ratings for each category after imputation
        category_mean_ratings = {}
        print_debug("\nMean ratings for each category after imputation:", debug=self.debug)
        for i, category in enumerate(self.category_names):
            mean_rating = np.mean(pred_ratings[:, i])
            # Add to dict
            category_mean_ratings[category] = round(mean_rating, 1)
            print_debug(f"{category}: {mean_rating:.2f}", debug=self.debug)
        return imputed_df, category_mean_ratings

    def __item_item_cf(self, ratings, category_names):
        """
        Run item-item collaborative filtering given the ratings, categories, and closest items
        :param ratings: 2D NumPy array of ratings
        :param category_names: Category names
        :return: Scaled similarity scores
        """
        print_debug("Preview ratings matrix (first 5 users):", debug=self.debug)
        print_debug(pd.DataFrame(ratings[:5],
                                 index=[f'User {i + 1}' for i in range(5)],
                                 columns=category_names), debug=self.debug)

        # Calculate means excluding NaN values for each item (column)
        # E.g. if there are 5 out of 6 values present then all existing values will be added and divided by 5
        item_means = np.array([np.nanmean(ratings[:, i]) for i in range(ratings.shape[1])])
        self.__item_means = item_means
        print_debug("\nItem means:", item_means, debug=self.debug)

        # Center the ratings by item
        for i in range(ratings.shape[1]):
            # Create mask for existing values
            exists = ~np.isnan(ratings[:, i])
            self.__centered_ratings[exists, i] = ratings[exists, i] - item_means[i]

        print_debug("\nPreview centered ratings (first 5 users):", debug=self.debug)
        print_debug(pd.DataFrame(self.__centered_ratings[:5],
                                 index=[f'User {i + 1}' for i in range(5)],
                                 columns=category_names), debug=self.debug)

        # Calculate all pairwise item similarities
        n_items = ratings.shape[1]
        similarities = np.zeros((n_items, n_items))
        print_debug("\nCosine Similarities between items:", debug=self.debug)
        for i in range(n_items):
            for j in range(n_items):
                sim = cosine_similarity(self.__centered_ratings[:, i], self.__centered_ratings[:, j])
                similarities[i, j] = sim
                # Print unique pairs only
                if i < j:
                    print_debug(f"{category_names[i]} and {category_names[j]}: {sim:.8f}", debug=self.debug)

        self.item_similarities = similarities
        print_debug("\nItem Similarity Matrix:", debug=self.debug)
        print_debug(pd.DataFrame(similarities,
                                 index=category_names,
                                 columns=category_names), debug=self.debug)

        # Scale the similarity matrix column-wise as in the original code
        sim_scores_scaled = similarities.copy()
        # Scale each column separately
        for j in range(n_items):
            # Extract column, excluding the diagonal
            row = similarities[:, j].copy()
            # Set diagonal to NaN
            row[j] = np.nan
            # Min-Max scale
            scaled_col = minmax_scale(row)
            # Set the scaled values
            sim_scores_scaled[:, j] = scaled_col
            # Set diagonal back to 1.0
            sim_scores_scaled[j, j] = 1.0

        print_debug("\nScaled Similarity Matrix:")
        print_debug(pd.DataFrame(sim_scores_scaled,
                                 index=category_names,
                                 columns=category_names))

        # In case this can't be accessed outside the function
        self.__scaled_sim_scores = sim_scores_scaled
        return sim_scores_scaled

    def predict_rating(self, user_idx, item_idx, k=4):
        """Predict rating for a user-item pair using k most similar items with diversity enhancement."""
        # Get the original similarities for this item with all others
        item_similarities = self.__scaled_sim_scores[item_idx]
        # Get user's existing ratings
        user_ratings = self.ratings[user_idx]

        # Find items the user(s) rated
        rated_mask = ~np.isnan(user_ratings)
        # Exclude the target item
        rated_mask[item_idx] = False

        # If user hasn't rated any other items, use the mean
        if not np.any(rated_mask):
            return self.__item_means[item_idx]

        # Get similarities to rated items and corresponding ratings
        sims_to_rated = item_similarities[rated_mask]
        ratings_for_rated = user_ratings[rated_mask]
        # rated_indices = np.where(rated_mask)[0]

        # Use more neighbors if available
        k = min(k, len(sims_to_rated))
        if k == 0:
            return self.__item_means[item_idx]

        top_k_idx = np.argsort(sims_to_rated)[-k:]

        # Top similarities
        top_k_sims = sims_to_rated[top_k_idx]
        top_k_ratings = ratings_for_rated[top_k_idx]
        # top_k_indices = rated_indices[top_k_idx]

        # For debugging
        rated_cols = np.array(self.category_names)[rated_mask]
        top_k_cols = rated_cols[top_k_idx]

        print_debug(f"\nPredicting for User {user_idx + 1}, {self.category_names[item_idx]}:", debug=self.debug)
        print_debug(f"Using {len(top_k_cols)} similar categories: {top_k_cols}", debug=self.debug)
        print_debug(f"With similarities: {top_k_sims}", debug=self.debug)
        print_debug(f"And ratings: {top_k_ratings}", debug=self.debug)

        # Calculate weighted average
        if np.sum(top_k_sims) == 0:
            # Use item mean
            prediction = self.__item_means[item_idx]
        else:
            # Calculate weighted average
            weighted_sum = np.sum(top_k_sims * top_k_ratings)
            weight_sum = np.sum(top_k_sims)
            base_prediction = weighted_sum / weight_sum

            # Adjust based on standard deviation of ratings
            if len(top_k_ratings) > 1:
                std_dev = np.std(top_k_ratings)
                adjustment = np.random.uniform(-std_dev / 4, std_dev / 4)
                prediction = base_prediction + adjustment
            else:
                prediction = base_prediction

        # Make sure rating is within (1, 5) range
        prediction = min(max(prediction, 1.0), 5.0)
        # Round to 1 decimal for consistency and prevent overly specific ratings
        prediction = round(prediction, 1)

        return prediction

    def predict_all_ratings(self):
        # Predict all missing ratings
        pred_ratings = self.ratings.copy()
        print_debug("\nPredictions:", debug=self.debug)
        missing_positions = np.where(np.isnan(self.ratings))

        # Make all predictions
        for user_idx, item_idx in zip(*missing_positions):
            pred = self.predict_rating(user_idx, item_idx, self.k)
            pred_ratings[user_idx, item_idx] = pred
            print_debug(f"User {user_idx + 1}, {self.category_names[item_idx]}: {pred:.2f}", debug=self.debug)

        return pred_ratings

def plot_similarity_matrix(sim_scores, category_names, title="Item-Item Similarity Matrix", label="Similarity",
                           color_scheme="coolwarm"):
    """Plot heatmap of (scaled) similarity matrix."""
    plt.figure(figsize=(10, 8))
    dcn = decorate_category_names(category_names)
    # Labeling DF
    sim_df = pd.DataFrame(
        sim_scores,
        index=dcn,
        columns=dcn
    )

    # Plot scaled similarity matrix
    heatmap = sns.heatmap(sim_df, annot=True, fmt=".2f", cmap=color_scheme,
                          vmin=0, vmax=1, cbar_kws={'label': label})
    plt.title(title)
    return plt.gcf()

def plot_radar_chart(category_means):
    """Create a radar chart of category mean ratings."""
    overall_rating = category_means.pop("overall_rating")
    categories = list(category_means.keys())
    dcn = decorate_category_names(categories)
    N = len(categories)

    # Remove 'rating' category from radar chart if it exists
    if 'rating' in categories:
        categories.remove('rating')
        N = len(categories)
        values = [category_means[cat] for cat in categories]
    else:
        values = [category_means[cat] for cat in categories]

    # Compute angles for radar chart
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop

    # Add values to complete the loop
    values += values[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Draw the radar chart
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=f"Overall Rating: {overall_rating}")
    ax.fill(angles, values, alpha=0.25)
    ax.legend()

    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dcn)

    # Set y-axis limits
    ax.set_ylim(0, 5)

    # Add rating values at each point
    for angle, value, category in zip(angles[:-1], values[:-1], categories):
        ax.text(angle, value + 0.1, f"{value}",
                ha='center', va='center', fontsize=10)

    ax.set_title("Restaurant Category Ratings", size=15, pad=20)

    return fig

def plot_imputation_comparison(original_df, imputed_df, category_names):
    """Plot comparison of original vs imputed data."""
    # Calculate completion percentages
    original_completion = 100 - (original_df[category_names].isna().mean() * 100)
    imputed_completion = 100 - (imputed_df[category_names].isna().mean() * 100)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create dataframe for plotting
    comparison_df = pd.DataFrame({
        'Original': original_completion,
        'After Imputation': imputed_completion
    }, index=category_names)

    # Plot bar chart
    comparison_df.plot(kind='bar', ax=ax)
    ax.set_ylabel('Data Completion (%)')
    ax.set_title('Data Completion Before and After Imputation')
    ax.legend(loc='lower right')
    ax.set_xticklabels(decorate_category_names(category_names))
    plt.xticks(rotation=45)

    return fig


def plot_rating_histogram(ratings_df, category_names):
    """Plot histogram of ratings for each category."""
    fig, axes = plt.subplots(len(category_names), 1, figsize=(10, 3 * len(category_names)))

    # Decorate category names
    dcn = decorate_category_names(category_names)

    # If there's only one category, axes won't be an array
    if len(category_names) == 1:
        axes = [axes]

    # Plot histogram for each category
    for i, category in enumerate(category_names):
        # Get non-NaN values
        valid_ratings = ratings_df[category].dropna()

        # Create histogram
        axes[i].hist(valid_ratings, bins=9, range=(1, 5.5), alpha=0.7, color='steelblue', edgecolor='black')

        # Add labels and grid
        axes[i].set_title(f"{dcn[i]} Ratings Distribution")
        axes[i].set_xlabel("Rating Value")
        axes[i].set_ylabel("Count")
        axes[i].set_xlim(1, 5.5)
        axes[i].grid(True, linestyle='--', alpha=0.7)

        # Add count above each bar
        counts, bins = np.histogram(valid_ratings, bins=9, range=(1, 5.5))
        for count, x in zip(counts, bins[:-1]):
            if count > 0:
                axes[i].text(x + 0.25, count + 0.5, str(count), ha='center')

    plt.tight_layout()
    return fig

def get_cf_stats(original_df, imputed_df, category_names):
    """
    Evaluate how diverse the predictions are in the imputed dataframe
    """
    # Calculate standard deviation of each rating category
    std_devs = {}
    for col in category_names:
        std_devs[col] = imputed_df[col].std()
    # Count unique values in each category
    unique_counts = {}
    for col in category_names:
        unique_counts[col] = len(imputed_df[col].unique())
    # Calculate the percentage of imputed values
    total_cells = len(original_df) * (len(category_names))
    original_nan_count = original_df.iloc[:, 1:].isna().sum().sum()
    imputation_percentage = (original_nan_count / total_cells) * 100

    return std_devs, unique_counts, imputation_percentage

def get_available_restaurants(csv_path="./data/tabelog_review_data_history.csv"):
    """Get list of available restaurants from CSV file."""
    restaurants_df = pd.read_csv(csv_path)
    return restaurants_df

def update_history(store_name, store_name_r, file_path, url, csv_path="./data/tabelog_review_data_history.csv"):
    """Add a new restaurant to the list of available restaurants."""
    restaurants_df = get_available_restaurants(csv_path)

    # Check if restaurant already exists
    if restaurants_df.empty or not (restaurants_df['store_name_r'] == store_name_r).any():
        new_row = pd.DataFrame({
            "store_name": [store_name],
            "store_name_r": [store_name_r],
            "path": [file_path],
            "url": [url]
        })
        restaurants_df = pd.concat([restaurants_df, new_row], ignore_index=True)
        # Check directory exists
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        # Save updated list
        restaurants_df.to_csv(csv_path, index=False)
    return restaurants_df

def cleanup_restaurant_name(name):
    # Replace special characters and spaces
    return re.sub(r'[\\/*?:"<>|.]', "", name).replace(" ", "_")

def scrape_tabelog_reviews(url: str, max_pages=10, directory="./data/tabelog_review_data"):
    """
    Use the scraper to get reviews
    :param url: URL to Tabelog restaurant
    :param max_pages: Max number of pages of reviews to get
    :param directory: Directory to save the CSV files
    :return: Store name, restaurant URL, and review DataFrame
    """
    tabelog_rs = TabelogReviewScraper(url, max_pages=max_pages, debug=False)
    tabelog_rs.save_data(directory=directory)
    store_name, restaurant_url = tabelog_rs.get_store_info()
    return store_name, restaurant_url, tabelog_rs.get_reviews_df()

def decorate_category_names(categories):
    """
    Remove underscores and capitalize
    :param categories: List-like of category names
    :return: Title-style category names
    """
    decorated_category_names = []
    for category in categories:
        decorated_category_names.append(category.replace("_", " ").title())
    return decorated_category_names


# Initialize session state variables
def init_session_state():
    if 'current_restaurant_data' not in st.session_state:
        st.session_state.current_restaurant_data = None
    if 'imputed_df' not in st.session_state:
        st.session_state.imputed_df = None
    if 'category_means' not in st.session_state:
        st.session_state.category_means = None
    if 'show_graphs' not in st.session_state:
        st.session_state.show_graphs = False
    if 'cf_object' not in st.session_state:
        st.session_state.cf_object = None

def home_page():
    st.title("Tabelog (食べログ) Collaborative Filtering")

    st.markdown("""
    ## Project Overview
    
    This application uses **item-item** collaborative filtering 
    to analyze and predict missing ratings from users on Tabelog (食べログ), 
    a popular Japanese restaurant review platform.

    ### What is Collaborative Filtering?
    Collaborative filtering is a technique used in recommendation systems. It works by finding patterns in how users 
    rate different items or categories and uses them to predict missing ratings.

    ### App Usage
    1. **Select a restaurant** from our database or provide a Tabelog URL
    2. **Run collaborative filtering** to impute missing ratings
    3. **Explore visualizations** of the rating data
    4. **Get insights** about what the restaurant is best rated for

    ### Rating Categories
    Tabelog ratings include:
    - **料理・味 (Food and taste)**: Quality and taste of the food
    - **サービス (Service)**: Quality of the service
    - **雰囲気 (Atmosphere)**: Restaurant ambiance and setting
    - **CP (Cost Performance)**: Value for money
    - **酒・ドリンク (Alcohol and drink)**: Quality of alcoholic and non-alcoholic beverages
    - **Overall Rating**: Reviewer's overall rating of the restaurant (independent of all other categorical ratings)

    ## Getting Started
    Navigate to the "Restaurant Analysis" page to begin exploring restaurant ratings!
    """)

    # Tabelog Image that redirects to site
    st.markdown("""
    ## Tabelog Site
    <a href="https://tabelog.com/">
        <img src="data:image/png;base64,{}">
    </a>
    """.format(
        base64.b64encode(open("./media/tabelog_image.png", "rb").read()).decode()
    ), unsafe_allow_html=True)

def analysis_page():
    st.title("Restaurant Rating Analysis")
    # Initialize session state
    init_session_state()
    # Get available restaurants from the CSV "database"
    available_restaurants = get_available_restaurants()
    # Create tabs for selecting existing or adding new restaurant
    tab1, tab2 = st.tabs(["Select Existing Restaurant", "Add New Restaurant"])

    with tab1:
        if not available_restaurants.empty:
            restaurant_options = available_restaurants['store_name'].tolist()
            selected_restaurant = st.selectbox("Choose a restaurant:", restaurant_options)

            if st.button("Load Selected Restaurant"):
                # Get restaurant data
                selected_idx = restaurant_options.index(selected_restaurant)
                restaurant_path = available_restaurants.iloc[selected_idx]['path']

                try:
                    restaurant_data = pd.read_csv(restaurant_path)
                    st.session_state.current_restaurant_data = restaurant_data
                    st.success(f"Loaded data for {selected_restaurant}")
                except Exception as e:
                    st.error(f"Error loading restaurant data: {e}")
        else:
            st.info("No restaurants available. Please add a new restaurant.")

    with tab2:
        tabelog_url = st.text_input("Enter a Tabelog URL:")

        # Give user option to choose number of pages
        col1, col2 = st.columns([1, 1])

        with col1:
            use_max_pages = st.checkbox("Get all reviews from all pages", value=False)
        with col2:
            if not use_max_pages:
                num_pages = st.number_input("Number of pages to scrape", min_value=1, max_value=100, value=10)
            else:
                num_pages = None
                st.info("All available pages will be scraped")

        if st.button("Process Restaurant"):
            with st.spinner("Processing restaurant data..."):
                store_name, restaurant_url, restaurant_data = scrape_tabelog_reviews(tabelog_url, max_pages=num_pages)

                if restaurant_data is not None:
                    # Generate file path
                    store_name_r = cleanup_restaurant_name(store_name)
                    file_path = f"./data/tabelog_review_data/tabelog_{store_name_r}_review_data.csv"

                    # Save restaurant data
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    restaurant_data.to_csv(file_path, index=False)

                    # Update available restaurants list
                    update_history(store_name, store_name_r, file_path, tabelog_url)

                    # Set current restaurant data
                    st.session_state.current_restaurant_data = restaurant_data
                    st.success(f"Successfully processed {store_name}")
                else:
                    st.error("Failed to process restaurant data. Please check the URL.")

    # Display current restaurant data
    if st.session_state.current_restaurant_data is not None:
        st.subheader("Restaurant Review Data")
        st.dataframe(st.session_state.current_restaurant_data)

        # Run collaborative filtering
        if st.button("Run Collaborative Filtering"):
            with st.spinner("Running collaborative filtering..."):
                # Prepare data for collaborative filtering
                restaurant_data = st.session_state.current_restaurant_data
                user_ids = restaurant_data['user_id'].values
                category_names = restaurant_data.columns[1:]
                ratings = np.array(restaurant_data.iloc[:, 1:].values, dtype=float)

                # Create collaborative filtering object
                cf = TabelogCollaborativeFilter(ratings, category_names, user_ids, k=4)
                st.session_state.cf_object = cf

                # Run prediction and imputation
                imputed_df, category_means = cf.predict_and_impute()

                # Store results in session state
                st.session_state.imputed_df = imputed_df
                st.session_state.category_means = category_means
                st.session_state.show_graphs = True

                st.success("Collaborative filtering completed successfully!")

        # Toggle to show / hide graphs
        if st.session_state.imputed_df is not None:
            # Get category names once there is an imputed DataFrame
            category_names = st.session_state.imputed_df.columns[1:]
            show_graphs = st.checkbox("Show visualizations", value=st.session_state.show_graphs)
            st.session_state.show_graphs = show_graphs

            if show_graphs:
                # Show results
                st.subheader("Imputed Rating Data")
                st.dataframe(st.session_state.imputed_df)

                # Display mean ratings
                st.subheader("Mean Ratings by Category")

                # Find the highest rated category, excluding overall rating
                category_means = st.session_state.category_means
                category_means_no_overall = category_means
                category_to_exclude = 'overall_rating' if 'overall_rating' in category_means else None
                highest_category = max(
                    (cat for cat in category_means if cat != category_to_exclude),
                    key=lambda cat: category_means[cat]
                )

                # Create columns for mean ratings display
                col1, col2 = st.columns([1, 1])

                with col1:
                    # Display all mean ratings
                    for category, mean in category_means.items():
                        st.metric(category, f"{mean:.1f}/5.0")

                with col2:
                    # Highlight best feature
                    st.markdown(f"### Restaurant Highlight")
                    st.markdown(f"This restaurant is best rated for:")
                    st.markdown(f"#### {highest_category.capitalize()}: {category_means[highest_category]}/5.0")
                    st.markdown(f"*Based on item-item collaborative filtering of user reviews*")

                # Create tabs for different visualizations
                viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
                    "Rating Histogram",
                    "Similarity Matrix",
                    "Rating Radar",
                    "Imputation Stats"
                ])

                with viz_tab1:
                    st.subheader("Rating Distribution")
                    fig = plot_rating_histogram(st.session_state.imputed_df, category_names)
                    st.pyplot(fig)

                with viz_tab2:
                    st.subheader("Item Similarity Matrix")
                    fig = plot_similarity_matrix(
                        st.session_state.cf_object.item_similarities,
                        category_names,
                        title="Item-Item Similarity Matrix",
                        label="Cosine Similarity",
                        color_scheme="coolwarm"
                    )
                    st.pyplot(fig)

                    st.subheader("Scaled Similarity Matrix")
                    fig = plot_similarity_matrix(
                        st.session_state.cf_object.get_ssc(),
                        category_names,
                        title="Scaled Item-Item Similarity Matrix",
                        label="Scaled Similarity",
                        color_scheme="YlGnBu"
                    )
                    st.pyplot(fig)

                with viz_tab3:
                    st.subheader("Restaurant Rating Radar")
                    fig = plot_radar_chart(st.session_state.category_means)
                    st.pyplot(fig)

                with viz_tab4:
                    # Calculate statistics
                    std_devs, unique_counts, imputation_percentage = get_cf_stats(
                        st.session_state.current_restaurant_data,
                        st.session_state.imputed_df,
                        category_names
                    )

                    # Display statistics
                    st.subheader("Imputation Statistics")
                    st.markdown(f"**Percentage of values imputed:** {imputation_percentage:.2f}%")

                    # Show comparison of original vs imputed data
                    fig = plot_imputation_comparison(
                        st.session_state.current_restaurant_data,
                        st.session_state.imputed_df,
                        category_names
                    )
                    st.pyplot(fig)

                    # Display standard deviations
                    st.subheader("Standard Deviations")
                    std_dev_df = pd.DataFrame({
                        'Category': std_devs.keys(),
                        'Standard Deviation': [f"{std:.2f}" for std in std_devs.values()]
                    })
                    st.table(std_dev_df)

                    # Display unique counts
                    st.subheader("Unique Rating Values")
                    unique_df = pd.DataFrame({
                        'Category': unique_counts.keys(),
                        'Unique Values': unique_counts.values()
                    })
                    st.table(unique_df)


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Restaurant Analysis"])
    if page == "Home":
        home_page()
    elif page == "Restaurant Analysis":
        analysis_page()

    # pages = {
    #     "Navigation": [
    #         st.Page(home_page, title="Home"),
    #         st.Page(analysis_page, title="Restaurant Analysis"),
    #     ]
    # }
    # pages = st.navigation(pages)
    # pages.run()


if __name__ == "__main__":
    # Set page config
    st.set_page_config(
        page_title="Tabelog Collaborative Filtering",
        page_icon="🍽️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    main()
