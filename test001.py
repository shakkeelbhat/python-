import os
import sys
import itertools
import re
import string
if not os.path.exists('C:/neuralsum/cnn/training/body/'):
	os.makedirs('C:/neuralsum/cnn/training/body/')
if not os.path.exists('C:/neuralsum/cnn/training/summary/'):
	os.makedirs('C:/neuralsum/cnn/training/summary/')
if not os.path.exists('C:/neuralsum/cnn/training/Untagged_summary/'):
	os.makedirs('C:/neuralsum/cnn/training/Untagged_summary/')
if not os.path.exists('C:/neuralsum/cnn/training/Untagged_body/'):
	os.makedirs('C:/neuralsum/cnn/training/Untagged_body/')
def looper(filename,handle):
	d=handle.split('\n\n')
	url=d[0]
	para2=d[1]
	para3=d[2]
	para4=d[3]
	lines=[]
	lines=lines+para2.split('\n')
	lineVStag=[]
	for line in lines:
		lineVStag=lineVStag+line.split('\t\t\t')
	lines=[]
	tags=[]
	for i in range(0,len(lineVStag)):
		if i%2==0:
			lines.append(lineVStag[i])
		elif i%2!=0:
			tags.append(lineVStag[i])
	elines=[]
	for(i,j) in zip(lines,tags):
		if j==2:
			continue
		else:
			elines.append(i)
	elines2=[]
	for i in elines:
		i=i+' ***'
		elines2.append(i)
	ewords=[]
	for w in elines2:
		ewords=ewords+w.split(' ')

	linesp4=[]
	linesp4=para4.split('\n')
	p4words=[]
	for p in linesp4:
		p4words=p4words+p.split(':')
	keys=[]
	values=[]
	for i in range(0,len(p4words)):
		if i%2==0:
			keys.append(p4words[i])
		elif i%2!=0:
			values.append(p4words[i])
	new_e_words=[]
	for i in ewords:
		if i in keys:
			new_e_words.append(' '+values[keys.index(i)])
		else:
			new_e_words.append(' '+i)
	Convert_new_e_words=''
	for i in new_e_words:
		Convert_new_e_words=Convert_new_e_words+i
	finalp2=Convert_new_e_words.replace('***','\n')
	port00=open('C:/neuralsum/cnn/training/Untagged_body/%s.txt'%filename,'w',encoding='utf-8')
	port00.write(finalp2)
	para2=open('C:/neuralsum/cnn/training/body/%s.txt'%filename,'w',encoding='utf-8')
	###################################################
	pLess=re.compile('[%s]'%re.escape(string.punctuation))
	punct=pLess.sub("",finalp2)
	Tsort=re.sub('[0-12]+:[0-60]+',"<TIME>",punct)
	TsortT=re.sub('[1-31]+/[1-12]+/[0000-9999]','<DATE>',Tsort)
	Dsort=re.sub('\d\d\d\d',"<YEAR>",TsortT)
	NUMsort=re.sub('[\d]+',"<NUM>",Dsort)
	nSort=NUMsort.lower()
	para2.write(nSort)
	###############################################
	p3lines=[]
	p3lines=p3lines+para3.split('\n')
	newlines=[]
	for y in p3lines:
		y=y+' ***'
		newlines.append(y)
	p3words=[]
	for x in newlines:
		p3words=p3words+x.split(' ')
	replaced_p3words=[]
	for i in p3words:
		if i in keys:
			replaced_p3words.append(' '+values[keys.index(i)])
		else:
			replaced_p3words.append(' '+i)
	finalp3=''
	for i in replaced_p3words:
		finalp3=finalp3+i.replace('***','\n')
	port=open('C:/neuralsum/cnn/training/Untagged_summary/%s.txt'%filename,'w',encoding='utf-8')
	port.write(finalp3)
	#####################################################	
	pLess=re.compile('[%s]'%re.escape(string.punctuation))
	punct=pLess.sub("",finalp3)
	Tsort=re.sub('[0-12]+:[0-60]+',"<TIME>",punct)
	TsortT=re.sub('[1-31]+/[1-12]+/[0000-9999]','<DATE>',Tsort)
	Dsort=re.sub('\d\d\d\d',"<YEAR>",TsortT)
	NUMsort=re.sub('[\d]+',"<NUM>",Dsort)
	nSort=NUMsort.lower()
	para3=open('C:/neuralsum/cnn/training/summary/%s.txt'%filename,'w',encoding='utf-8')
	para3.write(nSort)

directory="E:\\neuralsum\\cnn\\training\\"
for filename in os.listdir(directory):
	path =os.path.join('E:\\neuralsum\\cnn\\training', filename)
	print('Processing file', filename)
	fil = open(path, encoding='utf-8')
	handle=fil.read()
	looper(filename,handle)
	fil.close()
	