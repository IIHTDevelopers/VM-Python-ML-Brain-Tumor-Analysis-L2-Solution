from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_model(model, X_train, y_train):
    datagen = ImageDataGenerator(validation_split=0.2)
    train_gen = datagen.flow(X_train, y_train, batch_size=32, subset='training')
    val_gen = datagen.flow(X_train, y_train, batch_size=32, subset='validation')
    history = model.fit(train_gen, validation_data=val_gen, epochs=10)
    return history
