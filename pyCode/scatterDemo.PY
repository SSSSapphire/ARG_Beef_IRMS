import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [2, 4, 6, 8, 10], 'label': ['A', 'B', 'C', 'D', 'E']})

fig, ax = plt.subplots()
ax.scatter(df['x'], df['y'], c='blue', s=100)

for i, txt in enumerate(df['label']):
    ax.annotate(txt, (df['x'][i], df['y'][i]))

ax.set_title('Scatter Plot')
ax.set_xlabel('X')
ax.set_ylabel('Y')

plt.show()