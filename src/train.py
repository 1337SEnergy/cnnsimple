import keras, numpy as np;
import tensorflow as tf;

(xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.mnist.load_data();
xtrain = np.reshape(xtrain, xtrain.shape + (1,));
xtest = np.reshape(xtest, xtest.shape + (1,));

model = keras.models.load_model("model.h5");

epoch_count = int(input("Epoch amount -> "));
for epoch in range(epoch_count):
	print("Epoch", epoch+1, "/", epoch_count);
	
	model.fit(xtrain, keras.utils.to_categorical(ytrain), batch_size=32);
	model.save("model_trained.h5");

res = model.evaluate(xtest, keras.utils.to_categorical(ytest));
print(res);