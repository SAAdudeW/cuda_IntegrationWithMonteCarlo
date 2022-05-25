import matplotlib.pyplot as plt

with open('C:\\Users\\asmis\\source\\repos\\CUDA_seminarHT\\output1.txt', 'r') as datasource:
    mis1 = list(map(float, datasource.readline().split()))

with open('C:\\Users\\asmis\\source\\repos\\CUDA_seminarHT\\output2.txt', 'r') as datasource:
    mis2 = list(map(float, datasource.readline().split()))

DN = []
for i in range(len(mis1)):
    DN.append(i * 10)

plt.plot(DN, mis1, DN, mis2)
plt.xlabel('dots number')
plt.ylabel('mistake')
plt.legend(['lin', 'dis'], loc='best')
plt.show()
