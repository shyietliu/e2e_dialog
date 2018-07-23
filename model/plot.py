import matplotlib.pyplot as plt


year1 = [2013, 2014, 2015, 2017, 2018]
year2 = [2014, 2015, 2017, 2018]
year3 = [2014, 2015, 2016, 2018]
currie = [8, 7, 7, 6, 8]
queensferry_road = [46, 41, 43, 55]
gorgie_road = [34, 32, 33, 29]
loc_4 = [59,65,53,42]
glasgow_road = [38, 27, 26, 26, 24]
salamander = [27,28,27,23]


plt.figure(figsize=(6,5))
plt.title('The title you want')

# plt.subplot(121)
# plt.plot(year1, currie, label='Currie, 20mph')
# plt.ylabel('NO2')
#
# # plt.subplot(222)
# plt.plot(year2, queensferry_road, label='Queensferry Road, 20mph')
# plt.ylabel('NO2')
# plt.legend()
plt.plot(year3, salamander, label='Salamander Street, 30mph')
plt.ylabel('NO2')
# plt.subplot(122)
plt.plot(year3, gorgie_road, label='Gorgie Road, 20mph')
plt.ylabel('NO2')

# plt.subplot(224)
plt.plot(year3, loc_4, label='St John\'s Road, 20mph')
plt.ylabel('NO2')

# plt.subplot(235)
# plt.plot(year1, Currie, label='Currie')
# plt.ylabel('NO2')
#
# plt.subplot(236)
# plt.plot(year1, Currie, label='Currie')
# plt.ylabel('NO2')
plt.legend()
# plt.plot(year2, QR)sss
plt.tight_layout()
plt.savefig('../fig3.pdf')
plt.show()
