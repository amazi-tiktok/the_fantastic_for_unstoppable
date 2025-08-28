# create a virtual environment named venv
# activate the venv
# install the dependencies
# python3 -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt
# read the data from kagglehub, in order to have mock data
import kagglehub
import pandas as pd

def main():
  # read the fake data from ./data/reviews.csv and print out the format
  df = pd.read_csv('./data/reviews.csv')
  print('Columns:', df.columns.tolist())
  print('Sample row:')
  print(df.iloc[0])

if __name__ == "__main__":
  main()
