# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 13:13:01 2021

@author: timhe
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

save_loc = 'C:/Users/timhe/OneDrive/Documents/StatsWorkFolder/CS5526/Plots/EDAPlots/';
file_location = 'C:/Users/timhe/Temple Coursework/TempleData/ForestFire/forestfires.csv';


df = pd.read_csv(file_location);

area_list = np.sort(df.area.to_numpy());

lower_actuals = [y for y in area_list if y < 20];
lower_inxes = [i for i in range(len(lower_actuals))];

middle_actuals = [y for y in area_list if y >= 20 and y < 150];
middle_inxes = [len(lower_actuals) + 1 + i for i in range(len(middle_actuals))];

upper_actuals = [y for y in area_list if y >= 150];
upper_inxes = [len(lower_actuals) + len(middle_actuals) + 1 + i for i in range(len(upper_actuals))];

all_inxes = [i for i in range(len(area_list))];

f1 = plt.figure();
    
plt.scatter(all_inxes, area_list, c='#1f77b4', label='Actual Fire Area Burned');
plt.xlabel("Ordered Test Set")
plt.ylabel("Burned Area (hA)")
plt.title('Ordered Test Set Plot for all fires');
plt.legend();
plt.grid(True);
file_name = '{}{}.jpg'.format(save_loc, 'AllOrderedTestSet');
f1.savefig(fname=file_name);

f2 = plt.figure();
ax = plt.gca()
ax.set_ylim([0, 20]);
    
plt.scatter(lower_inxes, lower_actuals, c='#1f77b4', label='Actual Fire Area Burned');
plt.xlabel("Ordered Test Set")
plt.ylabel("Burned Area (hA)")
plt.title('Ordered Test Set Plot for ha less than 20');
plt.legend();
plt.grid(True);
file_name = '{}{}.jpg'.format(save_loc, 'OrderedTestSetLower');

f2.savefig(fname=file_name);

f3 = plt.figure();    
plt.scatter(middle_inxes, middle_actuals, c='#1f77b4', label='Actual Fire Area Burned');
plt.xlabel("Ordered Test Set")
plt.ylabel("Burned Area (hA)")
plt.title('Ordered Test Set Plot for ha between 20 and 150');
plt.legend();
plt.grid(True);
file_name = '{}{}.jpg'.format(save_loc, 'OrderedTestSetMiddle');
f3.savefig(fname=file_name);

f4 = plt.figure();    
plt.scatter(upper_inxes, upper_actuals, c='#1f77b4', label='Actual Fire Area Burned');
plt.xlabel("Ordered Test Set")
plt.ylabel("Burned Area (hA)")
plt.title('Ordered Test Set Plot for ha more than 150');
plt.legend();
plt.grid(True);
file_name = '{}{}.jpg'.format(save_loc, 'OrderedTestSetUpper');
f4.savefig(fname=file_name);
    

df_orig = pd.read_csv(file_location);

areas_ln = np.log(area_list + 1);
areas_log2 = np.log2(area_list + 1);
areas_log2_sqrt = np.sqrt(np.log(area_list + 1));


plt.figure();
plt.title('Untransformed Area');
plt.ylabel('Count');
plt.xlabel('Burned Area (Ha)');
plt.hist(area_list);
plt.grid(True);
plt.savefig('C:/Users/timhe/OneDrive/Documents/StatsWorkFolder/CS5526/Plots/EDAPlots/hist_untransformedArea.png');

plt.figure();
plt.title('ln(x+1) transformed Area');
plt.ylabel('Count');
plt.xlabel('Burned Area (Transformed)');
plt.hist(areas_ln);
plt.grid(True);
plt.savefig('{}hist_natLogArea.png'.format(save_loc));

plt.figure();
plt.title('log2(x+1) transformed Area');
plt.ylabel('Count');
plt.xlabel('Burned Area (Transformed)');
plt.hist(areas_log2);
plt.grid(True);
plt.savefig('{}hist_log2Area.png'.format(save_loc));

plt.figure();
plt.title('sqrt(log2(x+1)) transformed Area');
plt.ylabel('Count');
plt.xlabel('Burned Area (Transformed)');
plt.hist(areas_log2_sqrt);
plt.grid(True);
plt.savefig('{}hist_sqrtlog2Area.png'.format(save_loc));

