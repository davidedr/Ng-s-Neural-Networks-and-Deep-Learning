'''
Created on 02 set 2017

@author: davide
'''

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(100.0)
x.shape = (10, 10)

interp = 'bilinear'
#interp = 'nearest'
lim = -2, 11, -2, 6
plt.subplot(211, axisbg='g')
plt.title('blue should be up')
plt.imshow(x, origin='upper', interpolation=interp, cmap='jet')
#plt.axis(lim)

plt.subplot(212, axisbg='y')
plt.title('blue should be down')
plt.imshow(x, origin='lower', interpolation=interp, cmap='jet')
#plt.axis(lim)
plt.show()

import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
np.random.seed(sum(map(ord, "distributions")))
x = np.random.normal(size=100)
sns.distplot(x);
plt.show()

import numpy as np
import seaborn as sns
data = [1.5]*7 + [2.5]*2 + [3.5]*8 + [4.5]*3 + [5.5]*1 + [6.5]*8
sns.set_style('whitegrid')
sns.kdeplot(np.array(data), bw=0.5)
plt.show()

import pandas as pd
data = [1.5]*7 + [2.5]*2 + [3.5]*8 + [4.5]*3 + [5.5]*1 + [6.5]*8
df = pd.DataFrame(data)
df.plot(kind='density')
plt.show()