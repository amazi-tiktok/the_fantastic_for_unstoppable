# create a virtual environment named venv
# activate the venv
# install the dependencies
# python3 -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt
# read the data from kagglehub, in order to have mock data
import kagglehub
import pandas as pd
from detoxify import Detoxify
import nltk
import json
import os


def detect_policy_violation(cfg, text):
    # use detoxify to classify the comment to understand whether it is toxic
    # this cover the policy for example: hate speech, harassment, etc.
    # TODO: train our own model, based on the out put of the detoxify, then add some special cases of review. For example: the pizza is disgusting shouldn't be flagged as toxic, for the moment we just proceed with existing model
    if cfg["activate_language_toxicity_detection"]:
      model = Detoxify('unbiased')
      results_batch = model.predict(text)
      comment_flagged_types = []
      for toxicity_type, scores in results_batch.items():
          if scores >= cfg["toxicity_threshold"]:
              comment_flagged_types.append(toxicity_type)
      if len(comment_flagged_types) > 0:
          return ", ".join(comment_flagged_types)
      
    # we need to detect off-topic comments
    # TODO: implement off-topic comment detection


    # we need to detect images that are not allowed
    # TODO: implement image detection
    return None

# --- Function to Read First N Lines of a JSON File ---
def display_first_lines(filepath, n_lines=10):
    """
    Reads and displays the first n_lines of any text file.

    Args:
        filepath (str): The path to the text file.
        n_lines (int): The number of lines to display. Defaults to 10.
    """
    print(f"--- Displaying first {n_lines} lines from: {filepath} ---")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= n_lines:
                    break # Stop after reading n_lines
                # Print the line, stripping any trailing whitespace for clean output
                print(f"Line {i+1}: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
def main():
    # read the fake data from ./data/reviews.csv and print out the format
    df = pd.read_csv('./data/reviews.csv')
    nltk.download('stopwords')
    print('Columns:', df.columns.tolist())

    cfg = {
        "activate_language_toxicity_detection": False,
        "toxicity_threshold": 0.8
    }
    # get the data line by line for only the first 200 lines, call detect_policy_violation to know whether there are policy violations
    for index, row in df.iterrows():
        # print("Currently reading row", index)
        violation = detect_policy_violation(cfg, row['text'])
        if violation:
            print(f"Row {index} violation with review text : ", row['text'], violation)
    print(os.getcwd())
    display_first_lines('./data/review-Alabama.json', 50)


if __name__ == "__main__":
  main()
