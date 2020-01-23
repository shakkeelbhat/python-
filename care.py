class Elder():
	def __init__(self,name1,name2,funds,review):
		self.funds=funds;
		self.review=review;
		self.name=name;
	def detailsEntered():
		print(name+"has"+funds+"funds");

class Carer():
	def __init__(self,name1,name2,price,review):
		self.name1=name1;
		self.name2=name2;
		self.price=price;
		self.review=review;

	def details():
		print(name1+"expects"+price+"amount");


def elder(name , funds):
	E1=Edler(name,funds);

def carer(name,price):
	C1=Carer(name,price);



print("Welcome to Care");
import time;
time.sleep(3);
print("(1)Are you an elder in need of carer or (2)Carer who wishes to earn money");
int typeofPerson=input();
if typeofPerson==1:
	print("Enter your name and funds");
	name,funds=input();
	elder(name,funds);
elif typeofPerson==2:
	carer();




