import os, re
import envvar as envVar
from random import shuffle

val_dir    = "./val"
google_dir = "./google"
flickr_dir = "./flickr"
val_filelist_dir    = "./info/val_filelist.txt"
google_filelist_dir = "./info/train_filelist_google.txt"
flickr_filelist_dir = "./info/train_filelist_flickr.txt"

NUM_CLASSES = 1000
CLASS_LIST = list()

def getINT(string):
	return int(re.findall("(?<=\s)\d*", string)[0])

def getFILENAME(string):
	return re.findall(".*jpg", string)[0]

def update_ClassList(result):
	for i in result:
		CLASS_LIST.append(i[0])

def filter_ClassList(val):
	val_filtered = list()
	for i in val:
		if i[0] in CLASS_LIST:
			val_filtered.append(i)
	return val_filtered
	
def getTrain100():
	with open(google_filelist_dir, "r") as f:
		tmp = f.readlines()
	glines = ["./" + i.rstrip('\n') for i in tmp]

	with open(flickr_filelist_dir, "r") as f:
		tmp = f.readlines()
	flines = ["./" + i.rstrip('\n') for i in tmp]

	combined = [[i,0,[]] for i in range(1000)]

	if(envVar.TARGET == "WebVision1000"):
		for i in glines:
			tmpINT = getINT(i)
			combined[tmpINT][2].append(getFILENAME(i)) 
			combined[tmpINT][1] = combined[tmpINT][1] + 1

		for i in flines:
			tmpINT = getINT(i)
			combined[tmpINT][2].append(getFILENAME(i)) 
			combined[tmpINT][1] = combined[tmpINT][1] + 1

		combined.sort(key = lambda a : a[1], reverse = True)
		combined = combined

	elif(envVar.TARGET == "WebVision500A_Google"):
		for i in glines:
			tmpINT = getINT(i)
			combined[tmpINT][2].append(getFILENAME(i)) 
			combined[tmpINT][1] = combined[tmpINT][1] + 1

		combined.sort(key = lambda a : a[1], reverse = True)
		combined = combined[:500]

	elif(envVar.TARGET == "WebVision500B_Google"):
		for i in glines:
			tmpINT = getINT(i)
			combined[tmpINT][2].append(getFILENAME(i)) 
			combined[tmpINT][1] = combined[tmpINT][1] + 1

		combined.sort(key = lambda a : a[1], reverse = True)
		combined = combined[500:1000]

	elif(envVar.TARGET == "WebVision500A_Flickr"):
		for i in flines:
			tmpINT = getINT(i)
			combined[tmpINT][2].append(getFILENAME(i)) 
			combined[tmpINT][1] = combined[tmpINT][1] + 1

		combined.sort(key = lambda a : a[1], reverse = True)
		combined = combined[:500]

	elif(envVar.TARGET == "WebVision500B_Flickr"):
		for i in flines:
			tmpINT = getINT(i)
			combined[tmpINT][2].append(getFILENAME(i)) 
			combined[tmpINT][1] = combined[tmpINT][1] + 1

		combined.sort(key = lambda a : a[1], reverse = True)
		combined = combined[500:1000]

	else:
		raise Error("NO TARGET FOUND")

	update_ClassList(combined)
	return combined

def getTrain_dsn():
	with open(google_filelist_dir, "r") as f:
		tmp = f.readlines()
	glines = ["./" + i.rstrip('\n') for i in tmp]

	with open(flickr_filelist_dir, "r") as f:
		tmp = f.readlines()
	flines = ["./" + i.rstrip('\n') for i in tmp]

	source = [[i,0,[]] for i in range(1000)]
	target = [[i,0,[]] for i in range(1000)]

	# source
	if(envVar.SOURCE == "WebVision500A_Google"):
		for i in glines:
			tmpINT = getINT(i)
			source[tmpINT][2].append(getFILENAME(i)) 
			source[tmpINT][1] = source[tmpINT][1] + 1

		source.sort(key = lambda a : a[1], reverse = True)
		source = source[:500]

	elif(envVar.SOURCE == "WebVision500A_Flickr"):
		for i in flines:
			tmpINT = getINT(i)
			source[tmpINT][2].append(getFILENAME(i)) 
			source[tmpINT][1] = source[tmpINT][1] + 1

		source.sort(key = lambda a : a[1], reverse = True)
		source = source[:500]

	# target
	if(envVar.TARGET == "WebVision500B_Google"):
		for i in glines:
			tmpINT = getINT(i)
			target[tmpINT][2].append(getFILENAME(i)) 
			target[tmpINT][1] = target[tmpINT][1] + 1

		target.sort(key = lambda a : a[1], reverse = True)
		target = target[500:1000]

	elif(envVar.TARGET == "WebVision500B_Flickr"):
		for i in flines:
			tmpINT = getINT(i)
			target[tmpINT][2].append(getFILENAME(i)) 
			target[tmpINT][1] = target[tmpINT][1] + 1

		target.sort(key = lambda a : a[1], reverse = True)
		target = target[500:1000]


	update_ClassList(source)
	return source, target

def getVal100():
	with open(val_filelist_dir, "r") as f:
		tmp = f.readlines()
	vlines = ["./val/" + i.rstrip('\n') for i in tmp]
	val = [[i,0,[]] for i in range(1000)]

	for i in vlines:
		tmpINT = getINT(i)
		if (len(val[tmpINT][2]) <50):
			val[tmpINT][2].append(getFILENAME(i)) 
		val[tmpINT][1] = val[tmpINT][1] + 1

	return filter_ClassList(val)
