import matplotlib.pyplot as plt

labels = ['A', 'B', 'C']
sizes = [20, 30, 40]

plt.pie([s/100 for s in sizes], labels=labels, autopct='%1.0f%%', startangle=140, normalize=False)
plt.axis('equal')

plt.show()