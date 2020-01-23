import os
import sys
import itertools
import re
import string

def looper(fil,filename):
    directory="E:\\myneuralsum2-test\\test\\"
    if "%s-replica.txt"%filename in os.listdir(directory):
        return
    else:
        y1=open("E:\\myneuralsum2-test\\test\\%s-replica.txt"%filename,'w+',encoding='utf8')
        z1=fil.read()#read the passed file
        y1.write(z1)#write the content to the %s-replica
        part2=open("E:\\myneuralsum2-test\\files\\4files\\%s-para2.txt"%filename,'w+', encoding='utf-8')
        part3=open("E:\\myneuralsum2-test\\files\\4files\\%s-para3.txt"%filename,'w+', encoding='utf-8')
        part4=open("E:\\myneuralsum2-test\\files\\4files\\%s-pairs.txt"%filename,'w+', encoding='utf-8')
        d=z1.split('\n\n')
        d2=d[1]
        d3=d[2]
        d4=d[3]
        part2.write(d2)
        part3.write(d3)
        part4.write(d4)
        part2.close()
        part3.close()
        part4.close()
        para2a=open("E:myneuralsum2-test\\files\\4files\\%s-para2.txt"%filename,'r', encoding='utf-8')
        z2=para2a.readlines()
        SenAndLabel=[]
        x=open("E:\\myneuralsum2-test\\files\\4files\\senAndlabel\\%s-sentencePart.txt"%filename,'w', encoding='utf-8')
        y=open("E:\\myneuralsum2-test\\files\\4files\\senAndlabel\\%s-LabelPart.txt"%filename,'w', encoding='utf-8')
        for i in z2:
            k=i
            SenAndLabel=k.split('\t\t\t')
            y.write(SenAndLabel[1])
            x.write(SenAndLabel[0]+'\n')
        x.close()
        y.close()#x contains Sentences #y contains labels
        s=open("E:\\myneuralsum2-test\\files\\4files\\senAndlabel\\%s-sentencePart.txt"%filename,'r', encoding='utf-8')
        l=open("E:\\myneuralsum2-test\\files\\4files\\senAndlabel\\%s-LabelPart.txt"%filename,'r', encoding='utf-8')
        x1=s.read()
        x2=l.read()
        senList=x1.split('\n')
        labList=x2.split('\n')#extracting 1 and 2 labelled ones and skipping 0 labelled ones
        eS=open("E:\\myneuralsum2-test\\files\\4files\\senAndlabel\\%s-EsentencePart.txt"%filename,'w', encoding='utf-8')
        eL=open("E:\\myneuralsum2-test\\files\\4files\\senAndlabel\\%s-ELabelPart.txt"%filename,'w', encoding='utf-8')
        for (i,j) in zip(senList,labList):
            if j=='0':
             continue
            else:
                eS.write(i+'\n')
                eL.write(j+'\n')
        eS.close()
        eL.close()#with data4 file openned in part3 handle
        linesB=[]
        part3a=open("E:\\myneuralsum2-test\\files\\4files\\%s-para3.txt"%filename,'r+', encoding='utf-8')
        p3AsString=part3a.read()
        linesB.append(p3AsString.split('\n'))#linesB contains a list of lines, each one to be broken into words
        linelist=linesB[0]
        para3Aswords=open("E:\\myneuralsum2-test\\files\\%s-para3new.txt"%filename,'w', encoding='utf-8')
        wordB=[]
        for lines in linelist:
            wordB=lines.split(" ")
            for words in wordB:
                para3Aswords.write(words+'\n')
            para3Aswords.write('***\n')
        para3Aswords.close()
        part4a=open("E:\\myneuralsum2-test\\files\\4files\\%s-pairs.txt"%filename,'r+', encoding='utf-8')
        listoflines=part4a.readlines()
        key=open("E:myneuralsum2-test\\files\\4files\\keyValueFiles\\%s-keys.txt"%filename,'w', encoding='utf-8')
        value=open("E:myneuralsum2-test\\files\\4files\\keyValueFiles\\%s-values.txt"%filename,'w', encoding='utf-8')
        for items in listoflines:
            alag=items.split(':')
            key.write(alag[0]+'\n')
            value.write(alag[1])
        key.close()
        value.close()
        rkey=open("E:myneuralsum2-test\\files\\4files\\keyValueFiles\\%s-keys.txt"%filename,'r', encoding='utf-8')
        rvalue=open("E:myneuralsum2-test\\files\\4files\\keyValueFiles\\%s-values.txt"%filename,'r', encoding='utf-8')
        Ekey=rkey.readlines()
        Evalue=rvalue.readlines()
        extractValued=open("E:\\myneuralsum2-test\\files\\%s-extract.txt"%filename,'w+', encoding='utf-8')
        para3newhandle=open("E:\\myneuralsum2-test\\files\\%s-para3new.txt"%filename,'r', encoding='utf-8')
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
        extractAssemble=open("E:\\myneuralsum2-test\\files\\%s-extract.txt"%filename,'r+', encoding='utf-8')
        ext2=extractAssemble.read()
        ext3=ext2.replace("\n"," ")
        extfilenew=open("E:\\myneuralsum2-test\\test\\valued\\%s-extract2.txt"%filename,'w', encoding='utf-8')
        ext3=ext3.replace("***","\n")######
        extfilenew.write(ext3)
        linesC=[]
        EsentencePartHandle=open("E:\\myneuralsum2-test\\files\\4files\\senAndlabel\\%s-EsentencePart.txt"%filename,'r+', encoding='utf-8')
        EsentencePartList=EsentencePartHandle.read()
        linesC.append(EsentencePartList.split('\n'))
        linelist2=linesC[0]#since linesB is as [[]]
        EsenNewAswords=open("E:\\myneuralsum2-test\\files\\4files\\senAndlabel\\%s-EsenNew.txt"%filename,'w', encoding='utf-8')
        wordC=[]
        for lines in linelist2:
            wordC=lines.split(" ")
            for words in wordC:
                EsenNewAswords.write(words+'\n')
            EsenNewAswords.write('***\n')
        EsenNewAswords.close()
        rkey=open("E:myneuralsum2-test\\files\\4files\\keyValueFiles\\%s-keys.txt"%filename,'r', encoding='utf-8')
        rvalue=open("E:myneuralsum2-test\\files\\4files\\keyValueFiles\\%s-values.txt"%filename,'r', encoding='utf-8')  
        Ekey=rkey.readlines()
        Evalue=rvalue.readlines()
        extractValuedpara2=open("E:\\myneuralsum2-test\\files\\%s-extractofpara2.txt"%filename,'w+', encoding='utf-8')
        EsenNewhandle=open("E:\\myneuralsum2-test\\files\\4files\\senAndlabel\\%s-EsenNew.txt"%filename,'r', encoding='utf-8')
        wordD=EsenNewhandle.readlines()
        for words in wordD:
            if words[0]=='@':
                for (keys,values) in zip(Ekey,Evalue):
                    if words==keys:
                        extractValuedpara2.write(values)
                        break
            else:
                extractValuedpara2.write(words)
        extract2Assemble=open("E:\\myneuralsum2-test\\files\\%s-extractofpara2.txt"%filename,'r+', encoding='utf-8')
        extractValuedpara2.close()
        ext0=extract2Assemble.read()
        extO=ext0.replace("\n"," ")
        extfilenew2=open("E:\\myneuralsum2-test\\test\\valued\\%s-extract2ofpara2.txt"%filename,'w', encoding='utf-8')
        extO=extO.replace("***","\n")
        extfilenew2.write(extO)
        extfilenew2.close()
        extfilenew2=open("E:\\myneuralsum2-test\\test\\valued\\%s-extract2ofpara2.txt"%filename,'r', encoding='utf-8')
        readH=extfilenew2.read()
        pLess=re.compile('[%s]'%re.escape(string.punctuation))
        punct=pLess.sub("",readH)
        Tsort=re.sub('[0-12]+:[0-60]+',"<TIME>",punct)
        TsortT=re.sub('[1-31]+/[1-12]+/[0000-9999]','<DATE>',Tsort)
        Dsort=re.sub('\d\d\d\d',"<YEAR>",TsortT)
        NUMsort=re.sub('[\d]+',"<NUM>",Dsort)
        nSort=NUMsort.lower()
        tagged=open("E:\\myneuralsum2-test\\test\\valued\\tagged\\%s-TAGGEDextract2ofpara2.txt"%filename,'w', encoding='utf-8')
        tagged.write(nSort)
        tagged.close()
        extfilenew.close()
        extf2=open("E:\\myneuralsum2-test\\test\\valued\\%s-extract2.txt"%filename,'r', encoding='utf-8')
        extt=extf2.read()
        #readEX=extt.replace("\n"," ")
        dLess=re.compile('[%s]'%re.escape(string.punctuation))
        #punct2=dLess.sub("",readEX)
        punct2=dLess.sub("",extt)
        Tsort2=re.sub('[0-12]+:[0-60]+',"<TIME>",punct2)
        Tsort3=re.sub('[1-31]+/[1-12]+/[0000-9999]','<DATE>',Tsort2)
        Dsort2=re.sub('\d\d\d\d',"<YEAR>",Tsort3)
        NUMsort2=re.sub('[\d]+',"<NUM>",Dsort2)
        nSort2=NUMsort2.lower()
        tagged2=open("E:\\myneuralsum2-test\\test\\Valued\\tagged\\%s-TAGGEDextract2.txt"%filename,'w', encoding='utf-8')
        tagged2.write(nSort2)
        tagged2.close()
directory="E:\\neuralsum\\cnn\\test\\"
for filename in os.listdir(directory):
    path =os.path.join('E:\\neuralsum\\cnn\\test', filename)
    print('Processing file', filename)
    fil = open(path, encoding='utf-8')
    looper(fil,filename)
        