x=[('as', 'IN'), ('the', 'DT'), ('model', 'NN'), ('for', 'IN'), ('rockwell', 'NN'), ('s', 'NN'), ('rosie', 'VBZ')]
list1=[]
for i in x:
	print(i)
	if i[1]=='NN':
		continue
	else:
		list1.append(i)
print(list1)