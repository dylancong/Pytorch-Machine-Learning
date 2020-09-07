
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np
import pickle
from pathlib import Path

which = 1

mod_path = Path(__file__).parent

input_dir = (mod_path / "TestingScreenshot").resolve()

if which == 1:
	input_dir = (mod_path / "TestingScreenshot").resolve()

directories = os.listdir(input_dir)
print(os.getcwd() + "elll")

#Directory containing images you wish to convert


directories = os.listdir(input_dir)
print(directories)
index = 0
index2 = 0

index_test = 0
index2_test = 0
size = (32,32)


for folder in directories[0:10]:
	#Ignoring .DS_Store dir
	if folder == '.DS_Store':
		pass

	else:
		print (folder)

		folder2 = os.listdir(str(input_dir) + '/' + folder)
		index += 1

		len_images = len(folder2)
		len_images80 = len_images * 100 #Getting index of image on the 80% mark
		# len_images20 = len_images * 20 #Getting index of image on the 20% mark

		#Iterating through first 100% of images in folder for train data
		for image in folder2[0:int(len_images80)]:

			print(image + "hello")
			print(image)
			if image == ".DS_Store":
				pass

            
			else:
				index2 += 1

				im = Image.open(str(input_dir)+"/"+folder+"/"+image) #Opening image
				# plt.imshow(im)
				# plt.show()
				im  = ImageOps.fit(im,size,Image.ANTIALIAS)

				print(im.size) 
				im = (np.array(im)) #Converting to numpy array
				if(image == "Screen Shot 2020-08-04 at 8.05.38 PM.png"):
					plt.imshow(im)
					plt.show()

				try:
					r = im[:,:,0] #Slicing to get R data
					g = im[:,:,1] #Slicing to get G data
					b = im[:,:,2] #Slicing to get B data

					if index2 != 1:
						new_array = np.array([[r] + [g] + [b]], np.uint8) #Creating array with shape (3, 100, 100)
						out = np.append(out, new_array, 0) #Adding new image to array shape of (x, 3, 100, 100) where x is image number

					elif index2 == 1:
						out = np.array([[r] + [g] + [b]], np.uint8) #Creating array with shape (3, 100, 100)

					if index == 1 and index2 == 1:
						index_array = np.array([[index]])

					else:
						new_index_array = np.array([[index]], np.int8)
						index_array = np.append(index_array, new_index_array, 0)

				except Exception as e:
					print (e)
					print ("Removing image" + image)
					print("dang")
					# os.remove(str(input_dir)+"/"+folder+"/"+image)

		#Iterating throught last 20% of image in folder for test data
		# DO NOT NEED THE SECOND HALF OF THE DATA. DO THE TEST DATA SEPERATELY
		# for image in folder2[len_images-int(len_images20):len_images]:
		# 	if image == ".DS_Store":
		# 		pass

		# 	else:
		# 		index2_test += 1

		# 		im = Image.open(str(input_dir)+"/"+folder+"/"+image) #Opening image
		# 		im  = ImageOps.fit(im,size,Image.ANTIALIAS)
		# 		im = (np.array(im)) #Converting to numpy array

		# 		try:
		# 			r = im[:,:,0] #Slicing to get R data
		# 			g = im[:,:,1] #Slicing to get G data
		# 			b = im[:,:,2] #Slicing to get B data

		# 			if index2_test != 1:
		# 				new_array_test = np.array([[r] + [g] + [b]], np.uint8) #Creating array with shape (3, 100, 100)
		# 				out_test = np.append(out_test, new_array_test, 0) #Adding new image to array shape of (x, 3, 100, 100) where x is image number

		# 			elif index2_test == 1:
		# 				out_test = np.array([[r] + [g] + [b]], np.uint8) #Creating array with shape (3, 100, 100)

		# 			if index == 1 and index2_test == 1:
		# 				index_array_test = np.array([[index]])

		# 			else:
		# 				new_index_array_test = np.array([[index]], np.int8)
		# 				index_array_test = np.append(index_array_test, new_index_array_test, 0)

		# 		except Exception as e:
		# 			print (e)
		# 			print("what")
		# 			print ("Removing image" + image)
		# 			os.remove(str(input_dir)+"/"+folder+"/"+image)


print(index_array)
# print(index_array_test)

print(type(out))
print(type(index_array))


print(index_array.size)

# Forms the data into a form akin to the Cifar-10 dataset
# Have to change the bath_label to accomodate for the different training batches
# In the future try to find out the purpose of the different batches
index_array = index_array.ravel() # changes an array of ndarrays to integers
print(index_array)
dictTrain = {}
dictTrain['batch_label']= 'training batch 1 of 5'
dictTrain['labels'] = index_array
dictTrain['data'] = out
print(dictTrain)

# index_array_test=index_array_test.ravel()
# dictTest = dict(zip(index_array_test,out_test))
# print(dictTest)

# INDEXING IS OFF BY ONE

if which == 0:
	with open(os.path.join('/Users/games/Desktop/ML/Data','one.pickle'),'wb') as handle:
		pickle.dump(dictTrain, handle, protocol = pickle.HIGHEST_PROTOCOL) #Saving train image arrays and labels in a dict
	with open(os.path.join('/Users/games/Desktop/ML/Data','two.pickle'),'wb') as handle:
		pickle.dump(dictTrain, handle, protocol = pickle.HIGHEST_PROTOCOL) #Saving train image arrays and labels in a dict
	with open(os.path.join('/Users/games/Desktop/ML/Data','three.pickle'),'wb') as handle:
		pickle.dump(dictTrain, handle, protocol = pickle.HIGHEST_PROTOCOL) #Saving train image arrays and labels in a dict
	with open(os.path.join('/Users/games/Desktop/ML/Data','four.pickle'),'wb') as handle:
		pickle.dump(dictTrain, handle, protocol = pickle.HIGHEST_PROTOCOL) #Saving train image arrays and labels in a dict
	with open(os.path.join('/Users/games/Desktop/ML/Data','five.pickle'),'wb') as handle:
		pickle.dump(dictTrain, handle, protocol = pickle.HIGHEST_PROTOCOL) #Saving train image arrays and labels in a dict
	with open(os.path.join('/Users/games/Desktop/ML/Data','test.pickle'),'wb') as handle:
		pickle.dump(dictTrain, handle, protocol = pickle.HIGHEST_PROTOCOL) #Saving train image arrays and labels in a dict
elif which == 1:
	with open(os.path.join('/Users/games/Desktop/ML/Data','test.pickle'),'wb') as handle:
		pickle.dump(dictTrain, handle, protocol = pickle.HIGHEST_PROTOCOL) #Saving train image arrays and labels in a dict
elif which == 3:
	with open(os.path.join('/Users/games/Desktop/ML/Data','one.pickle'),'wb') as handle:
		pickle.dump(dictTrain, handle, protocol = pickle.HIGHEST_PROTOCOL) #Saving train image arrays and labels in a dict
	with open(os.path.join('/Users/games/Desktop/ML/Data','two.pickle'),'wb') as handle:
		pickle.dump(dictTrain, handle, protocol = pickle.HIGHEST_PROTOCOL) #Saving train image arrays and labels in a dict
	with open(os.path.join('/Users/games/Desktop/ML/Data','three.pickle'),'wb') as handle:
		pickle.dump(dictTrain, handle, protocol = pickle.HIGHEST_PROTOCOL) #Saving train image arrays and labels in a dict
	with open(os.path.join('/Users/games/Desktop/ML/Data','four.pickle'),'wb') as handle:
		pickle.dump(dictTrain, handle, protocol = pickle.HIGHEST_PROTOCOL) #Saving train image arrays and labels in a dict
	with open(os.path.join('/Users/games/Desktop/ML/Data','five.pickle'),'wb') as handle:
		pickle.dump(dictTrain, handle, protocol = pickle.HIGHEST_PROTOCOL) #Saving train image arrays and labels in a dict
# with open(os.path.join('/Users/games/Desktop/ML','X_Test.pickle'),'wb') as handle:
# 	pickle.dump(dictTest, handle, protocol = pickle.HIGHEST_PROTOCOL) #Saving test image arrays and labels in a dict
# np.save(os.path.join('/Users/games/Desktop/ML/Data','X_train.npy'), dict,allow_pickle = True) 


# np.save(os.path.join('/Users/games/Desktop/ML/Data','X_test.npy'), out_test) #Saving test image arrays
# np.save(os.path.join('/Users/games/Desktop/ML/Data','Y_test.npy'), index_array_test) #Saving test labels
# print("hello")