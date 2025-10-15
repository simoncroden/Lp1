import math

# This file provides the FORMAT you should use for the
# slopes in HP2.3. x denotes the horizontal distance
# travelled (by the truck) on a given slope, and
# alpha measures the slope angle at distance x
#
# iSlope denotes the slope index (i.e. 1,2,..10 for the
# training set etc.)
# iDataSet determines whether the slope under consideration
# belongs to the training set (data_set_index = 1), validation
# set (data_set_index = 2), or the test set (data_set_index = 3).
#
# Note that the slopes given below are just EXAMPLES.
# Please feel free to implement your own slopes below,
# as long as they fulfil the criteria given in HP2.3.
#
# You may remove the comments above and below, as they
# (or at least some of them) violate the coding standard 
#  a bit. :)
# The comments have been added as a clarification of the 
# problem that should be solved!).


def get_slope_angle(x, slope_index, data_set_index):   

    if (data_set_index == 1): # Training
        if (slope_index == 1):
            alpha = 4.12 + math.sin(x/100) + math.cos(math.sqrt(2)*x/50)
        elif slope_index == 2:
            alpha = 4.97 + 0.52 * math.sin(x / 80) - math.cos(x / 60)
        elif slope_index == 3:
            alpha = 3.55 + math.sin(math.sqrt(3) * x / 70) + 0.28 * math.cos(x / 30)
        elif slope_index == 4:
            alpha = 4.48 + 0.69 * math.sin(x / 90) - 0.52 * math.cos(math.sqrt(5) * x / 45)
        elif slope_index == 5:
            alpha = 6.08 + math.sin(x / 110) + math.cos(x / 90)
        elif slope_index == 6:
            alpha = 3.09 + 1.18 * math.sin(x / 60) - 0.79 * math.cos(x / 40)
        elif slope_index == 7:
            alpha = 5.05 + math.sin(math.sqrt(2) * x / 75) + math.cos(x / 55)
        elif slope_index == 8:
            alpha = 4.03 + 0.41 * math.sin(x / 100) + 0.59 * math.cos(math.sqrt(3) * x / 80)
        elif slope_index == 9:
            alpha = 3.82 + 0.88 * math.sin(x / 70) - 0.31 * math.cos(x / 50)
        elif slope_index == 10:
            alpha = 3.01 + 2 * math.sin(x / 50) + math.cos(math.sqrt(2) * x / 100)

    elif (data_set_index == 2): # Validation
        if (slope_index == 1):
            alpha = 6.03 - math.sin(x/100) + math.cos(math.sqrt(3)*x/50)
        elif slope_index == 2:
            alpha = 5.21 + 0.49 * math.sin(x / 60) - 0.19 * math.cos(x / 30)
        elif slope_index == 3:
            alpha = 4.52 + math.sin(math.sqrt(2) * x / 70) + math.cos(x / 90)
        elif slope_index == 4:
            alpha = 5.81 + 0.71 * math.sin(x / 80) - math.cos(math.sqrt(3) * x / 60)
        elif slope_index == 5:
            alpha = 5.02 + math.sin(x/50) + math.cos(math.sqrt(5)*x/50)

    elif (data_set_index == 3): # Test
        if (slope_index == 1):
            alpha = 6.04 - math.sin(x/100) + math.cos(math.sqrt(7)*x/50)
        elif slope_index == 2:
            alpha = 5.51 + 0.61 * math.sin(x / 60) - math.cos(x / 100)
        elif slope_index == 3:
            alpha = 4.01 + math.sin(math.sqrt(3) * x / 90) + 0.52 * math.cos(x / 40)
        elif slope_index == 4:
            alpha = 5.02 + 0.81 * math.sin(x / 80) - 0.39 * math.cos(math.sqrt(5) * x / 70)
        elif slope_index == 5:
            alpha = 4.03 + (x/1000) + math.sin(x/70) + math.cos(math.sqrt(7)*x/100)

    return alpha