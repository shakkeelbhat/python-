elders={
        5000 :['budha1',['+','-','+']],
        1000 :['budha2',['+','+','+']]
                }
carers={
        1000:['amar',3,['+','+','+','+']],
        5000 :['jyoti',4,['-','+','+']],
        2000:['jawan',1,['+','+','+','+']]
              }

print("Welcome to CareGiving");
import time
time.sleep(3);
import sys
def basic():
	top=0;
	print("(1)Are you an elder in need of carer or (2)Carer who wishes to earn money");
	top=input('Enter 1 or 2:');
	if(top==1):
		print("Enter your name and funds separated by space");
		name,fund=input('Enter name and funds available').split(" ");
		elder[funds]=name;
		print("would you like to go farwards 0/1 (No/Yes)");
		choice=input();
		if(choice!=0 | choice!=1):
			print("you entered wrong input");
		if(choice==0):
			sys.exit(0);
		elif(choice==1):
			elderFarward(name,funds);
	if(top==2):
		print("Enter your name and price separated by space");
		name,price=input().split(" ");
		carer[price]=name;
		print("would you like to go farwards 0/1 (No/Yes)");
		choice=input();
		if(choice!=0 | choice!=1):
			print("you entered wrong input");
		if(choice==0):
			sys.exit(0);
		elif(choice==1):
			if(carer[price][1]<4):
				carerFarward(name,price);
			else:
				print("you have reached the max count");



def elderFarward(name,funds):
	bestprice=0;
	print("looking for carers under the range of "+ funds);
	for key in carers:
		if (key>funds):
			continue;
		if(carers[key][1]>=4):
			continue;
		R=checkReview(key[2],carers);
		if(R=='-'):
			continue;
		if(R=='+'):
			if(key<bestprice):
				bestprice=key;
	del elders[bestprice];
	carers[bestprice][1]= carers[bestprice][1] + 1;


def carerFarward(name,price):
	print("looking for carers under the range of "+price);
	for key in elders:
			if(funds<price):
				continue;
			else:
				R=checkReview(key[1],elders);
				if(R=='-'):
					continue;
				if(R=='+'):
						carers[price][1]= carer[price][1] + 1;


def checkReview(key,dictionary):
	rarray=dictionary[key];
	pluscount=0;
	minuscount=0;
	for item in rarray:
		if(item=='+'):
			pluscount=pluscount+1;
		if(item=='-'):
			minuscount=minuscount+1;
	if(pluscount > minuscount):
		return '+';
	else:
		return '-';

basic();
