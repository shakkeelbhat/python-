lista=[1,2,3,4,5,6,7,8,9,10]
listb=['a','b','c','d','e','f','g','h']
d = dict((key,value) for (key, value) in zip(lista,listb))
print(d)
print('####################################################')
e=dict((i, i+1) for i in range(0, 9, 2))
print(e)