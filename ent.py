import numpy as np
ent = 0
gini = 0
p = [1/4,3/4]
for i in p:
	ent+= -(i)*np.log(i)
print(ent)


for i in p:
	gini+= (i)*(1-(i))
print(gini)