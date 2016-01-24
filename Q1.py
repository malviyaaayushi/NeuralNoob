from __future__ import print_function

from McCullochPittsModel import McCullochPitts
import numpy as np

if __name__ == "__main__":
	print("Select a model:\n1. NOT\n2. AND\n3. OR\n4. Genarate all outputs\n\nYour choice: ", end='')
	choice  = int(raw_input())

	if choice==1 or choice==2 or choice==3:
		print("Please provide following values:\n")
		if choice==1:
			print("x: ", end='')
			x = int(raw_input())
			ip = [x]
			weight = [-1]
			t = 0
		elif choice==2 or choice==3:
			while(1):
				print("x1: ", end='')
				x1 = int(raw_input())
				while x1!=0 or x1!=1:
					print("Please enter a valid value (0 or 1)")
				else:
					break
			while(1):
				print("x2: ", end='')
				x2 = int(raw_input())
				if x2!=0 or x2!=1:
					print("Please enter a valid value (0 or 1)")
				else:
					break
			ip = [x1, x2]
			if choice==2:
				weight = [1, 1]
				t = 2
			if choice==3:
				weight = [0.5, 0.5]
				t = 0.5
		print("Output: "+str(McCullochPitts(len(ip), weight, t).forwardPropagation(ip)))
	elif choice==4:
		print("NOT Gate")
		print("x\toutput")
		for x in [0, 1]:
			print(str(x)+"\t", end='')
			print(McCullochPitts(len([x]), [-1], 0).forwardPropagation([x]))
		print("\nAND Gate")
		print("x1\tx2\toutput")
		for x1 in [0, 1]:
			for x2 in [0, 1]:
				print(str(x1)+"\t"+str(x2)+"\t"+str(McCullochPitts(len([x1, x2]), [1, 1], 2).forwardPropagation([x1, x2])))
		print("\nOR Gate")
		print("x1\tx2\toutput")
		for x1 in [0, 1]:
			for x2 in [0, 1]:
				print(str(x1)+"\t"+str(x2)+"\t"+str(McCullochPitts(len([x1, x2]), [0.5, 0.5], 0.5).forwardPropagation([x1, x2])))
	else:
		print("Please enter a valid input")


	

