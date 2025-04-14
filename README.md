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
Assuming required packages are downloaded:
1. Start the venv `source /venv/bin/activate` (path to your venv)
2. Run `python3 -m streamlit run tabelog_cf_app.py`
3. Let the app open in your browser

### Getting Started
Navigate to the "Restaurant Analysis" page to begin exploring restaurant ratings!