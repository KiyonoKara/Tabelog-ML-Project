# Tabelog ML Project
It's a Tabelog ML Project. Explores Japanese restaurant data with collaborative filtering and Bayesian modeling.

## Project Overview
    
This application uses **item-item** collaborative filtering 
to analyze and predict missing ratings from users on Tabelog (食べログ), 
a popular Japanese restaurant review platform.

### What is Collaborative Filtering?
Collaborative filtering is a technique used in recommendation systems. It works by finding patterns in how users 
rate different items or categories and uses them to predict missing ratings.

### What is Bayesian Modeling?
Bayesian modeling uses numerical features of restaurant ratings such as engagement and price ranges. 
Using the data, it uses prior evidence and updates the predictions accordingly, since it's used to find probability of a hypothesis.

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

## Running the App
1. Clone this repository and download all the packages from `requirements.txt`
2. Start the venv `source /venv/bin/activate` (path to your venv)
3. Run `python3 -m streamlit run tabelog_cf_app.py`
4. See console and copy the local host link and open it in your browser or let the app open automatically

### Getting Started
Navigate to the "Restaurant Analysis" page to begin exploring restaurant ratings!

### Adding Restaurant URLs
If you'd like to add your own restaurants, visit https://tabelog.com/en, and pick out restaurants. **Before adding them, remove the "en" part from the link.
Example: `https://tabelog.com/en/tokyo/...` ----->  `https://tabelog.com/tokyo/...`

Here are some examples restaurants you can try entering on the app:
- https://tabelog.com/hyogo/A2801/A280109/28071150/ - A ramen shop in Hyogo
- https://tabelog.com/tokyo/A1302/A130201/13215419/ - A brewery in Tokyo
- https://tabelog.com/okayama/A3301/A330101/33015797/ - A pizza place in Okayama

## Remarks
- Content-Based Filtering might've been a much better fit instead of (Item-Item) Collaborative Filtering since there were more than type of rating and there were daytime / nighttime prices.
- Collaborative Filtering tended to compute higher-than-expected ratings which didn't always reflect the overall consensus due missing values being imputed then used.
- In a different approach, better methods like Content-Based Filtering would be used to properly predict ratings and explore (Japanese) restaurant data.

### Things to consider doing in the future
- Automatically parse Tabelog URLs to get the Japanese webpages only (or adjust the scraper for the English site).
- Use Content-Based Filtering.
- Cleanup and document the data collection and processing.
