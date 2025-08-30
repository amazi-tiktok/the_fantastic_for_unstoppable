# create a virtual environment named venv
# activate the venv
# install the dependencies
# python3 -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt
# read the data from kagglehub, in order to have mock data
import kagglehub
import pandas as pd
import nltk
import json
import time
import os
from policy import contains_commercial_info, contains_profanity,classify_review_with_image 

def detect_policy_violation(category, text):
    # use detoxify to classify the comment to understand whether it is toxic
    # this cover the policy for example: hate speech, harassment, etc.
    # TODO: train our own model, based on the out put of the detoxify, then add some special cases of review. For example: the pizza is disgusting shouldn't be flagged as toxic, for the moment we just proceed with existing model
    contains_commercial_info(text)

    # we need to detect off-topic comments
    # TODO: implement off-topic comment detection
    classify_review_with_image(category, text)
    # we need to detect images that are not allowed
    # TODO: implement image detection

    return None
    
def main():
    # read the fake data from ./data/reviews.csv and print out the format
    df = pd.read_json('./data/category_reviews/abortion_clinic.json')
    nltk.download('stopwords')
    print('Columns:', df.columns.tolist())

    # get the data line by line for only the first 200 lines, call detect_policy_violation to know whether there are policy violations
    for index, row in df.iterrows():
        # print("Currently reading row", index)
        violation = detect_policy_violation("abortion_clinic", row['text'])
        # wait a sec for rate limit
        time.sleep(1)
        if violation:
            print(f"Row {index} violation with review text : ", row['text'], violation)
    print(os.getcwd())

if __name__ == "__main__":
  main()
