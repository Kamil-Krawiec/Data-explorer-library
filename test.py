import matplotlib.pyplot as plt

# Data for the pie chart
labels = ['A', 'B', 'C']
sizes = [20, 30, 40]

# Create the pie chart
plt.pie([s/100 for s in sizes], labels=labels, autopct='%1.0f%%', startangle=140, normalize=False)
plt.axis('equal')

# Display the chart
plt.show()