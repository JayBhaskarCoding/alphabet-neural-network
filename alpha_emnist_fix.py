import os
import time
import cv2
from sklearn.model_selection import train_test_split
import urllib.request
import zipfile
import gzip
import numpy as np
import gradio as gr

class NeuralNetwork:
	def __init__(self, input_nodes, hidden_nodes, output_nodes):
		self.W1 = np.random.randn(input_nodes, hidden_nodes) * np.sqrt(1/input_nodes) #First Weights
		self.b1 = np.zeros((1, hidden_nodes)) #First Biases
		self.W2 = np.random.randn(hidden_nodes, output_nodes) * np.sqrt(1/hidden_nodes) #Second Weights
		self.b2 = np.zeros((1, output_nodes))

	def sigmoid(self, x): #Sigmoid Function for the activation of the data values
		y = 1/(1 + np.exp(-x))
		return y

	def softmax_func(self, Z):
	    y = Z - np.max(Z, axis=1, keepdims=True)
	    exp_Z = np.exp(y)
	    t = exp_Z / np.sum(exp_Z, axis=1, keepdims=True) #these are probabilities

	    return t	

	def forward_pass(self, X): #Creating the forward pass X is the input data
		self.Z1 = np.dot(X, self.W1) + self.b1
		self.A1 = self.sigmoid(self.Z1) #Activation variable

		self.Z2 = np.dot(self.A1, self.W2) + self.b2
		self.A2 = self.softmax_func(self.Z2) #Final activation (Network's Prediction)

		return self.A2

	def calculate_loss(self, true_value): #This calculates how wrong the prediction of the network is and gives an average score
		safe_limited_A2 = 1e-8 + self.A2
		log_A2 = np.log(safe_limited_A2)
		errors = true_value * log_A2
		m = true_value.shape[0] #No. of images in the batch
		loss = (-1 * (np.sum(errors))) / m

		return loss

	def sigmoid_derivative(self, A):
		y = A*(1-A)
		return y

	def backward_pass(self, X, true_value, learning_rate):
		m = X.shape[0] #The size of the catch
		dZ2 = self.A2 - true_value #Output error
		dW2 = (1/m) * np.dot(self.A1.T, dZ2) #Output Weight Blame
		db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
		dZ1 = np.dot(dZ2, self.W2.T) * self.sigmoid_derivative(self.A1)
		dW1 = (1/m) * np.dot(X.T, dZ1)
		db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

		self.W1 -= learning_rate * dW1
		self.b1 -= learning_rate * db1
		self.W2 -= learning_rate * dW2
		self.b2 -= learning_rate * db2

	def train(self, X, true_value, epochs, learning_rate):
		start_time = time.time()
		for i in range(epochs):
			self.forward_pass(X)

			if i % 100 == 0:
				current_time = time.time()
				elapsed_time = current_time - start_time
				print(f"The loss after {i} iterations is: ",self.calculate_loss(true_value), f"Elapsed Time: {elapsed_time: .2f}s")
			else:
				pass

			self.backward_pass(X, true_value, learning_rate)

		total_time = time.time() - start_time
		mins = int(total_time // 60)
		sec = int(total_time % 60)

		print("\n--Training Complete--")
		print(f"Total Training Time: {mins} minutes and {sec} seconds\n")

	def save_weights(self, filename="trained_weights_biases.npz"):
		np.savez(filename, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
		print(f"\n[SUCCESS] Trained Model Saved in {filename}")

	def load_weights(self, filename="trained_weights_biases.npz"):
		with np.load(filename) as data:
			self.W1 = data['W1']
			self.b1 = data['b1']
			self.W2 = data['W2']
			self.b2 = data['b2']
		print(f"\nNeural Network Loaded from {filename}")

def predict_drawing(image):
	if image is None or image["composite"] is None:
	 	return {chr(i+65): 0.0 for i in range(26)} 

	img_array = image["composite"]

	gray = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)

	if gray[0,0] > 127:
		processed = cv2.bitwise_not(gray)
	else:
		processed = gray

	if np.max(processed) == 0:
		return {chr(i+65): 0.1 for i in range(26)}

	coords = cv2.findNonZero(processed)

	if coords is not None:
		x, y, w, h = cv2.boundingRect(coords)

		cropped = processed[y:y+h, x:x+w]

		size = max(w, h)
		padded = np.zeros((size, size), dtype=np.uint8)
		x_offset = (size - w) // 2
		y_offset = (size - h) // 2
		padded[y_offset:y_offset+h, x_offset:x_offset+w] = cropped

		resized_20 = cv2.resize(padded, (20, 20), interpolation=cv2.INTER_AREA)

		temp_canvas = np.zeros((28,28), dtype=np.uint8)
		temp_canvas[4:24, 4:24] = resized_20

		M = cv2.moments(temp_canvas)

		if M["m00"] != 0:
			cX = M["m10"] / M["m00"]
			cY = M["m01"] / M["m00"]
		else:
			cX, cY = 14, 14

		shift_x = 14.0 - cX
		shift_y = 14.0 - cY

		M_trans = np.float32([[1,0,shift_x], [0,1,shift_y]])
		final_img = cv2.warpAffine(temp_canvas, M_trans, (28,28))

	else:
		final_img = cv2.resize(inverted, (28, 28), interpolation=cv2.INTER_AREA)

	final_img = final_img.T

	kernel = np.ones((2,2), np.uint8)
	final_img = cv2.dilate(final_img, kernel, iterations=1)

	max_val = np.max(final_img)
	if max_val > 0:
		scaled = final_img / max_val
	else:
		scaled = final_img / 255.0

	#scaled = final_img / 255.0
	flattened = scaled.flatten().reshape(1,-1)

	prediction = nn.forward_pass(flattened)

	return {chr(i+65): float(prediction[0][i]) for i in range(26)}

interface = gr.Interface(fn=predict_drawing, inputs=gr.Sketchpad(type="numpy", label="Draw an UPPERCASE letter (A-Z) in the center of the box!"), outputs=gr.Label(num_top_classes=3, label="The AI's Guess"), title="Letter Guessing NN", description="Draw an UPPERCASE letter (A-Z) in the center of the box!", live=True)

def load_emnist_native(zip_filepath):
    with zipfile.ZipFile(zip_filepath, 'r') as z:
        # Get a list of every single file inside the zip archive
        all_files = z.namelist()
        
        # Auto-detect the exact paths by searching for the file names
        img_path = next(f for f in all_files if 'emnist-letters-train-images-idx3-ubyte.gz' in f)
        lbl_path = next(f for f in all_files if 'emnist-letters-train-labels-idx1-ubyte.gz' in f)
        
        print(f"Found images at: {img_path}")
        print(f"Found labels at: {lbl_path}")

        # 1. Open the compressed image file using the auto-detected path
        with z.open(img_path) as img_gz:
            with gzip.open(img_gz, 'rb') as img_f:
                img_f.read(16) # Skip the 16-byte IDX header
                X = np.frombuffer(img_f.read(), dtype=np.uint8).copy().reshape(-1, 784) / 255.0

        # 2. Open the compressed label file using the auto-detected path
        with z.open(lbl_path) as lbl_gz:
            with gzip.open(lbl_gz, 'rb') as lbl_f:
                lbl_f.read(8) # Skip the 8-byte IDX header
                y = np.frombuffer(lbl_f.read(), dtype=np.uint8).copy()

    return X, y

if __name__ == "__main__":

	X, y = load_emnist_native("emnist.zip")

	y -= 1

	num_samples = len(y)
	num_classes = 26
	y_one_hot = np.zeros((num_samples, num_classes))

	for i, target in enumerate(y):
	    y_one_hot[i][target] = 1

	X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

	nn = NeuralNetwork(input_nodes=784, hidden_nodes=512, output_nodes=26)

	if os.path.exists("trained_weights_biases.npz"):
		nn.load_weights("trained_weights_biases.npz")
	else:
		print(f"Starting Training on {len(X_train)} images.... ")
		nn.train(X_train, y_train, epochs=2500, learning_rate=0.45)
		nn.save_weights("trained_weights_biases.npz")

	print("\n --Testing on New Data-- \n")
	predictions = nn.forward_pass(X_test)
	predicted_classes = np.argmax(predictions, axis=1)
	actual_classes = np.argmax(y_test, axis=1)

	correct_guesses = np.sum(predicted_classes == actual_classes)
	total_images = len(actual_classes)
	accuracy = (correct_guesses / total_images) * 100

	print(f"Correctly Identified {correct_guesses} numbers out of {total_images}")
	print(f"True Network Accuracy: {accuracy: .2f}%")

	print("Launching Web Interface... ")
	interface.launch(inbrowser=True, debug=True, share=True)  # Only used it for debugging
