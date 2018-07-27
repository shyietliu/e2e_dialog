import matplotlib.pyplot as plt
import csv



file_path = '/Users/shyietliu/Desktop/ning/monthly_data_2013_8_to_16_7.csv'

file_path_2 = '/Users/shyietliu/Desktop/ning/monthly_16_18.csv'
with open(file_path) as f:
    csv_data = csv.reader(f, delimiter=',')
    data = [row for row in csv_data]
    pass

data = data[1:]
mean = []

for i in range(12):
    mean.append((float(data[i][2]) + float(data[i+12][2]) + float(data[i+24][2]))/3)


with open(file_path_2) as f:
    csv_data = csv.reader(f, delimiter=',')
    data2 = [row for row in csv_data]
    pass
mean2 = []
for i in range(12):
    mean2.append((float(data[i][2]) + float(data[i+12][2]))/2)

plt.plot(mean)
plt.plot(mean2)
plt.show()