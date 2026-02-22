## IR RELEVANCE JUDGEMENT TOOL
### PROJECT OVERVIEW

This project implements a Streamlit-based annotation tool for Information Retrieval (IR) relevance evaluation.

The system allows users to manually rate news articles for predefined queries and generate structured relevance judgments suitable for IR evaluation experiments.

The tool is designed for:

Building evaluation datasets

Generating relevance labels

Supporting IR system benchmarking

Producing structured JSON outputs for further metric computation

### CORE FUNCTIONALITY

The application:

Loads a CSV dataset containing news articles

Loads a predefined JSON list of queries

Retrieves up to K documents per query

Allows manual relevance rating per article

Persists ratings within the session

Saves ratings only when the user clicks Submit

Allows skipping queries safely

Prevents overwriting previously saved ratings


### MAIN SYSTEM COMPONENTS

#### LOAD DATASET

Reads CSV file (001-news-items.csv)

Normalizes column names

Ensures required fields exist (id, title, content)

Builds internal article list

#### LOAD QUERIES

Reads queries.json

Stores predefined queries in memory

#### DOCUMENT RETRIEVAL

Performs simple case-insensitive keyword search

Searches both title and content

Returns up to K matching articles per query

#### STREAMLIT USER INTERFACE

Displays query at the top

Shows adjustable K slider

Displays article title and content

Provides radio button for each article:

Relevant

Not Relevant

Not Rated

Ratings persist using session state

#### SAVE MECHANISM

Collects ratings only when Submit is clicked

Saves ratings into ratings.json

Appends safely without overwriting previous data

#### QUERY CONTROL

Submit Ratings → Saves ratings for current query

Skip Query → Moves to a new random query without saving

#### HOW TO RUN THE APPLICATION

Install dependencies:

pip install streamlit pandas

Run the application:

streamlit run main.py

Open the local URL shown in your terminal in your browser.

#### STRENGTHS

Adjustable K enables flexible document sampling

Persistent session state prevents rating loss during reruns

Random query selection reduces evaluation bias

Structured JSON output suitable for further IR analysis

Clean and simple annotation interface

#### LIMITATIONS

Uses simple keyword search (not a full IR ranking system)

Single-user session state

No integration with advanced IR models

Manual labeling is time-intensive

No built-in progress tracking

No multi-user backend support

#### PRODUCTION-LEVEL IMPROVEMENT SUGGESTIONS

Integrate a real IR ranking system

Replace keyword search with TF-IDF, BM25, or neural ranking

Present realistic top-K results

Use a centralized backend database

Store ratings in PostgreSQL or MongoDB

Enable multi-user collaboration

Prevent data conflicts

Store detailed metadata

User ID

Timestamp

Query-document context


#### USE CASES

Academic IR experiments

Creating labeled datasets

Testing retrieval systems

Building evaluation benchmarks

#### TECHNOLOGIES USED

Python

Streamlit

Pandas

JSON

#### LICENSE

MIT License
