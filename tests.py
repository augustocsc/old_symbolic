import re

def transform_text_to_csv(text):
  """Transform the given text into a CSV string using the given regular expression.

  Args:
    text: The text to be transformed.

  Returns:
    A CSV string, where each row is separated by a newline character and each column is separated by a comma.
  """

  regex = r"mean: (?P<mean>[-\d.]+)\ntop: (?P<top>[-\d.]+)\ninvalid: (?P<invalid>\d+)\n"

  rows = [['mean', 'top', 'invalid']]
  for match in re.finditer(regex, text):
    row = [match.group("mean"), match.group("top"), match.group("invalid")]
    rows.append(row)

  csv_string = "\n".join(",".join(row) for row in rows)
  return csv_string

# Example usage:
#load the file log.txt into text
text = ""
with open("log.txt", "r") as f:
  text = f.read()

csv_string = transform_text_to_csv(text)

#save the csv_string into a csv file
with open("log_csv.csv", "w") as f:
  f.write(csv_string)

'''
load the tile log_csv.csv and plot its columns in a line chart
'''

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("log_csv.csv")
'''
Create an image with tree line charts, one for each column in df. Then, save the image in a file.
'''
fig, axes = plt.subplots(3, 1, figsize=(10, 10))
df['top'].plot(ax=axes[0], title='Top', color='blue')
df['mean'].plot(ax=axes[1], title='Mean', color='red')
df['invalid'].plot(ax=axes[2], title='Invalid', color='green')
plt.savefig('log_csv.png')
