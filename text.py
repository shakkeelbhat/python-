import nltk

str1=["this is a string, isn't it?","this is the second string"]
tarray=[]

for i in str1:
	tarray=nltk.tokenize.WhitespaceTokenizer().tokenize(i)

c=nltk.tokenize.WordPunctTokenizer().tokenize(str1)

e=[]
for i in tarray:
	for j in i:
		d=nltk.stem.PorterStemmer(i)
		e.append(d)

f=[]
for i in b:
	g=nltk.stem.WordNetLemmatizer(i)
	f.append(g)
print(b)

g=[]
count=0
for i in e:
	g[0][count++]=i
	word=g[0][count]
	if b.search(word):




