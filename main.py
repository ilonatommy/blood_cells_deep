from Dataset import *

data_set = Dataset('/home/ilona/Desktop/Literatura/Bachelors_Thesis/blood_cells/blood-cells_db/dataset2-master')
train, validation, test = data_set.get_data()
