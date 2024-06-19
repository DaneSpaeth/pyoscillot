import matplotlib.pyplot as plt

# Plot in latex fonts
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.rc('text.latex', preamble=r'\usepackage{underscore}')
plt.rc('font', family='serif')

# Fontsizes
VERY_SMALL_SIZE = 5
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

# 180 mm in inch from A&A
FULL_PAGE_WIDTH = 7.08661
# 88 mm in inch from A&A
HALF_PAGE_WIDTH = 3.46457
# Measured for PhD thesis
THESIS_WIDTH = 5.8476

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
