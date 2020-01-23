import os
import sys
import itertools
import re
import string
x1 = open("C:\\validation\\000c8f89a079e222372b7f6b6ee8d35db42ace66.summary", "r") 
y1=open("E:extraction\\file1\\replica.txt",'w')
#
z1=x1.read()
y1.write(z1)
x1.close()
#
part1=open("E:extraction\\file1\\url.txt",'w+')
part2=open("E:extraction\\file1\\para2.txt",'w+')
part3=open("E:validation\\NotValued\\para3.txt",'w+')
part4=open("E:extraction\\file1\\para4.txt",'w+')
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

para2a=open("E:extraction\\file1\\para2.txt",'r')
z2=para2a.readlines()
# z2 contains list of lines of para2

#above each z2 item contains a line with its label
SenAndLabel=[]
x=open("E:extraction\\file1\\sentencePart.txt",'w')
y=open("E:extraction\\file1\\LabelPart.txt",'w')
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

s=open("E:extraction\\file1\\sentencePart.txt",'r')
l=open("E:extraction\\file1\\LabelPart.txt",'r')
x1=s.read()
x2=l.read()
senList=x1.split('\n')
labList=x2.split('\n')

#extracting 1 and 2 labelled ones and skipping 0 labelled ones
eS=open("E:validation\\NotValued\\F1EsentencePart.txt",'w')
eL=open("E:extraction\\file1\\F1ELabelPart.txt",'w')
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
part3a=open("E:extraction\\file1\\para3.txt",'r+')
p3AsString=part3a.read()
linesB.append(p3AsString.split('\n'))
#linesB contains a list of lines, each one to be broken into words
linelist=linesB[0]#since linesB is as [[]]

para3Aswords=open("E:extraction\\file1\\para3new.txt",'w')
wordB=[]
for lines in linelist:
 wordB=lines.split(" ")
 for words in wordB:
  para3Aswords.write(words+'\n')
 para3Aswords.write('***\n') 

para3Aswords.close()
###########################################################
#with data5
part4a=open("E:extraction\\file1\\para4.txt",'r+')
listoflines=part4a.readlines()

key=open("E:extraction\\file1\\keys.txt",'w')
value=open("E:extraction\\file1\\values.txt",'w')

for items in listoflines:
	alag=items.split(':')
	key.write(alag[0]+'\n')
	value.write(alag[1])
key.close()
value.close()

rkey=open("E:extraction\\file1\\keys.txt",'r')
rvalue=open("E:extraction\\file1\\values.txt",'r')	

Ekey=rkey.readlines()
Evalue=rvalue.readlines()

extractValued=open("E:extraction\\file1\\extract.txt",'w+')
para3newhandle=open("E:extraction\\file1\\para3new.txt",'r')
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
extractAssemble=open("E:extraction\\file1\\extract.txt",'r+')

ext2=extractAssemble.read() 
ext3=ext2.replace("\n"," ")
extfilenew=open("E:validation\\Valued\\F1extract2.txt",'w')
ext3=ext3.replace("***","\n")
extfilenew.write(ext3)

linesC=[]
EsentencePartHandle=open("E:extraction\\file1\\EsentencePart.txt",'r+')
EsentencePartList=EsentencePartHandle.read()
linesC.append(EsentencePartList.split('\n'))

linelist2=linesC[0]#since linesB is as [[]]

EsenNewAswords=open("E:extraction\\file1\\EsenNew.txt",'w')
wordC=[]
for lines in linelist2:
 wordC=lines.split(" ")
 for words in wordC:
  EsenNewAswords.write(words+'\n')
 EsenNewAswords.write('***\n') 

EsenNewAswords.close()
###########################################################
rkey=open("E:extraction\\file1\\keys.txt",'r')
rvalue=open("E:extraction\\file1\\values.txt",'r')	

Ekey=rkey.readlines()
Evalue=rvalue.readlines()

extractValuedpara2=open("E:extraction\\file1\\extractofpara2.txt",'w+')
EsenNewhandle=open("E:extraction\\file1\\EsenNew.txt",'r')
wordD=EsenNewhandle.readlines()

for words in wordD:
 if words[0]=='@':
  for (keys,values) in zip(Ekey,Evalue):
   if words==keys:
     extractValuedpara2.write(values)
     break
 else:
  extractValuedpara2.write(words)

extractValuedpara2.close()
extract2Assemble=open("E:extraction\\file1\\extractofpara2.txt",'r+')

ext0=extract2Assemble.read() 
extO=ext0.replace("\n"," ")
extfilenew2=open("E:validation\\Valued\\F1extract2ofpara2.txt",'w')
extO=extO.replace("***","\n")
extfilenew2.write(extO)
extfilenew2.close()

extfilenew2=open("E:validation\\Valued\\F1extract2ofpara2.txt",'r')
readH=extfilenew2.read()
#######################################
pLess=re.compile('[%s]'%re.escape(string.punctuation))
punct=pLess.sub("",readH)
Tsort=re.sub('[0-12]+:[0-60]+',"<TIME>",punct)
TsortT=re.sub('[1-31]+/[1-12]+/[0000-9999]','<DATE>',Tsort)
Dsort=re.sub('\d\d\d\d',"<YEAR>",TsortT)
NUMsort=re.sub('[\d]+',"<NUM>",Dsort)
nSort=NUMsort.lower()
tagged=open("E:validation\\Valued\\F1-TAGGEDextract2ofpara2.txt",'w')
tagged.write(nSort)
tagged.close()
#######################################
extfilenew.close()
extf2=open("E:validation\\Valued\\F1extract2.txt",'r')
readEX=extf2.read()
dLess=re.compile('[%s]'%re.escape(string.punctuation))
punct2=dLess.sub("",readEX)
Tsort2=re.sub('[0-12]+:[0-60]+',"<TIME>",punct2)
Tsort3=re.sub('[1-31]+/[1-12]+/[0000-9999]','<DATE>',Tsort2)
Dsort2=re.sub('\d\d\d\d',"<YEAR>",Tsort3)
NUMsort2=re.sub('[\d]+',"<NUM>",Dsort2)
nSort2=NUMsort2.lower()
tagged2=open("E:validation\\Valued\\F1-TAGGEDextract2.txt",'w')
tagged2.write(nSort2)
tagged2.close()