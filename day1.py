import os
import sys
import itertools
import re
import string

def looper(fil,var):
	y1=open("E:\\myneuralsum\\test\\%s-replica.txt"%var,'w+',encoding='utf8')
	z1=fil.read()#read the passed file
	y1.write(z1)#write the content to the %s-replica 
	
	part1=open("E:\\myneuralsum\\files\\4files\\%s-url.txt"%var,'w+')
	part2=open("E:\\myneuralsum\\files\\4files\\%s-para2.txt"%var,'w+')
	part3=open("E:\\myneuralsum\\files\\4files\\%s-para3.txt"%var,'w+')
	part4=open("E:\\myneuralsum\\files\\4files\\%s-pairs.txt"%var,'w+')
#
	d=z1.split('\n\n')
	#
	d1=d[0]
	d2=d[1]
	d3=d[2]
	d4=d[3]
	part1.write(d1)
	part2.write(d2)
	part3.write(d3)
	part4.write(d4)
	part1.close()
	part2.close()
	part3.close()
	part4.close()

	para2a=open("E:myneuralsum\\files\\4files\\%s-para2.txt"%var,'r')
	z2=para2a.readlines()
# z2 contains list of lines of para2
#above each z2 item contains a line with its label
	SenAndLabel=[]
	x=open("E:\\myneuralsum\\files\\4files\\senAndlabel\\%s-sentencePart.txt"%var,'w')
	y=open("E:\\myneuralsum\\files\\4files\\senAndlabel\\%s-LabelPart.txt"%var,'w')
	for i in z2:
 		k=i
 		SenAndLabel=k.split('\t\t\t')
 		y.write(SenAndLabel[1])
 		x.write(SenAndLabel[0]+'\n')
 #SenAndLabel is a list containing alternate sentence and label  	 

	x.close()
	y.close()
#x contains Sentences 
#y contains labels

	s=open("E:\\myneuralsum\\files\\4files\\senAndlabel\\%s-sentencePart.txt"%var,'r')
	l=open("E:\\myneuralsum\\files\\4files\\senAndlabel\\%s-LabelPart.txt"%var,'r')
	x1=s.read()
	x2=l.read()
	senList=x1.split('\n')
	labList=x2.split('\n')
#extracting 1 and 2 labelled ones and skipping 0 labelled ones
	eS=open("E:\\myneuralsum\\files\\4files\\senAndlabel\\%s-EsentencePart.txt"%var,'w')
	eL=open("E:\\myneuralsum\\files\\4files\\senAndlabel\\%s-ELabelPart.txt"%var,'w')

	for (i,j) in zip(senList,labList):
 	   if j=='0':
 	   	 continue
 	   else:
     		eS.write(i+'\n')
     		eL.write(j+'\n')
	eS.close()
	eL.close()
#with data4 file openned in part3 handle
	linesB=[]
	part3a=open("E:\\myneuralsum\\files\\4files\\%s-para3.txt"%var,'r+')
	p3AsString=part3a.read()
	linesB.append(p3AsString.split('\n'))
#linesB contains a list of lines, each one to be broken into words
	linelist=linesB[0]#since linesB is as [[]]

	para3Aswords=open("E:\\myneuralsum\\files\\%s-para3new.txt"%var,'w')
	wordB=[]
	for lines in linelist:
		wordB=lines.split(" ")
	for words in wordB:
		para3Aswords.write(words+'\n')
		para3Aswords.write('***\n') 

	para3Aswords.close()
###########################################################
#with data5
	part4a=open("E:\\myneuralsum\\files\\4files\\%s-pairs.txt"%var,'r+')
	listoflines=part4a.readlines()

	key=open("E:myneuralsum\\files\\4files\\keyValueFiles\\%s-keys.txt"%var,'w')
	value=open("E:myneuralsum\\files\\4files\\keyValueFiles\\%s-values.txt"%var,'w')

	for items in listoflines:
		alag=items.split(':')
		key.write(alag[0]+'\n')
		value.write(alag[1])
	key.close()
	value.close()
	rkey=open("E:myneuralsum\\files\\4files\\keyValueFiles\\%s-keys.txt"%var,'r')
	rvalue=open("E:myneuralsum\\files\\4files\\keyValueFiles\\%s-values.txt"%var,'r')	

	Ekey=rkey.readlines()
	Evalue=rvalue.readlines()

	extractValued=open("E:\\myneuralsum\\files\\%s-extract.txt"%var,'w+')
	para3newhandle=open("E:\\myneuralsum\\files\\%s-para3new.txt"%var,'r')
	wordB=para3newhandle.readlines()
	for words in wordB:
		if words[0]=='@':
			for (keys,values) in zip(Ekey,Evalue):
				if words==keys:
					extractValued.write(values)
					break
		else:
  			extractValued.write(words)

	extractValued.close()
	extractAssemble=open("E:\\myneuralsum\\files\\%s-extract.txt"%var,'r+')

	ext2=extractAssemble.read() 
	ext3=ext2.replace("\n"," ")
	extfilenew=open("E:\\myneuralsum\\test\\valued\\%s-extract2.txt"%var,'w')
	ext3=ext3.replace("***","\n")
	extfilenew.write(ext3)
	linesC=[]
	EsentencePartHandle=open("E:\\myneuralsum\\files\\4files\\senAndlabel\\%s-EsentencePart.txt"%var,'r+')
	EsentencePartList=EsentencePartHandle.read()
	linesC.append(EsentencePartList.split('\n'))

	linelist2=linesC[0]#since linesB is as [[]]

	EsenNewAswords=open("E:\\myneuralsum\\files\\4files\\senAndlabel\\%s-EsenNew.txt"%var,'w')
	wordC=[]
	for lines in linelist2:
		wordC=lines.split(" ")
		for words in wordC:
			EsenNewAswords.write(words+'\n')
		EsenNewAswords.write('***\n') 

	EsenNewAswords.close()
###########################################################
	rkey=open("E:myneuralsum\\files\\4files\\keyValueFiles\\%s-keys.txt"%var,'r')
	rvalue=open("E:myneuralsum\\files\\4files\\keyValueFiles\\%s-values.txt"%var,'r')	

	Ekey=rkey.readlines()
	Evalue=rvalue.readlines()

	extractValuedpara2=open("E:\\myneuralsum\\files\\%s-extractofpara2.txt"%var,'w+')
	EsenNewhandle=open("E:\\myneuralsum\\files\\4files\\senAndlabel\\%s-EsenNew.txt"%var,'r')
	wordD=EsenNewhandle.readlines()
	for words in wordD:
		if words[0]=='@':
			for (keys,values) in zip(Ekey,Evalue):
				if words==keys:
					extractValuedpara2.write(values)	
					break
		else:
			extractValuedpara2.write(words)
  
	extract2Assemble=open("E:\\myneuralsum\\files\\%s-extractofpara2.txt"%var,'r+')
	extractValuedpara2.close()
	ext0=extract2Assemble.read() 
	extO=ext0.replace("\n"," ")
	extfilenew2=open("E:\\myneuralsum\\test\\valued\\%s-extract2ofpara2.txt"%var,'w')
	extO=extO.replace("***","\n")
	extfilenew2.write(extO)
	extfilenew2.close()

	extfilenew2=open("E:\\myneuralsum\\test\\valued\\%s-extract2ofpara2.txt"%var,'r')
	readH=extfilenew2.read()
#######################################
	pLess=re.compile('[%s]'%re.escape(string.punctuation))
	punct=pLess.sub("",readH)
	Tsort=re.sub('[0-12]+:[0-60]+',"<TIME>",punct)
	TsortT=re.sub('[1-31]+/[1-12]+/[0000-9999]','<DATE>',Tsort)
	Dsort=re.sub('\d\d\d\d',"<YEAR>",TsortT)
	NUMsort=re.sub('[\d]+',"<NUM>",Dsort)
	nSort=NUMsort.lower()
	tagged=open("E:\\myneuralsum\\test\\valued\\tagged\\%s-TAGGEDextract2ofpara2.txt"%var,'w')
	tagged.write(nSort)
	tagged.close()
#######################################
	extfilenew.close()
	extf2=open("E:\\myneuralsum\\test\\valued\\%s-extract2.txt"%var,'r')
	readEX=extf2.read()
	dLess=re.compile('[%s]'%re.escape(string.punctuation))
	punct2=dLess.sub("",readEX)
	Tsort2=re.sub('[0-12]+:[0-60]+',"<TIME>",punct2)
	Tsort3=re.sub('[1-31]+/[1-12]+/[0000-9999]','<DATE>',Tsort2)
	Dsort2=re.sub('\d\d\d\d',"<YEAR>",Tsort3)
	NUMsort2=re.sub('[\d]+',"<NUM>",Dsort2)
	nSort2=NUMsort2.lower()
	tagged2=open("E:\\myneuralsum\\test\\Valued\\tagged\\%s-TAGGEDextract2.txt"%var,'w')
	tagged2.write(nSort2)
	tagged2.close()

directory="E:\\neuralsum\\cnn\\test\\"
i=1
for filename in os.listdir(directory):
	#if filename.endswith(".summary"):
	var="f"+str(i)#to send f1,f2,f3 along the summary file
	fil = open("E:\\neuralsum\\cnn\\test\\%s"%filename, "r+",encoding="utf-8")
	looper(fil,var)
	print(str(filename))
	i=i+1
	#else:
	#	continue	