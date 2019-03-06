import keras;

model = keras.models.Sequential();
model.add(keras.layers.Conv2D(6, 5, input_shape=(28, 28, 1), activation="sigmoid"));
model.add(keras.layers.AveragePooling2D());

model.add(keras.layers.Conv2D(12, 5, activation="sigmoid"));
model.add(keras.layers.AveragePooling2D());

model.add(keras.layers.Flatten());
model.add(keras.layers.Dense(16, activation="sigmoid"));
model.add(keras.layers.Dense(10, activation="softmax"));

model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(lr=0.001), metrics=["accuracy"]);

model.summary();
model.save("model.h5");