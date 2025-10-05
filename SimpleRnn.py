import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,SimpleRNN,Embedding

max_features=10000 #vocabulary size
(X_train,y_train),(X_test,y_test)=imdb.load_data(num_words=max_features)
print(f"Training data shape{X_train.shape},Training label shape{y_train.shape}")
print(f"Testing data shape{X_test.shape}, Testing label shape{y_test.shape}")
sample_review=X_train[0]
sample_label=y_train[0]

print(f"Sample reveiw date is:{sample_review}")
print(f"Sample label date is:{sample_label}")

word_index=imdb.get_word_index()
reverse_word_index={value: key for key,value in word_index.items()}

decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in sample_review])
from tensorflow.keras.preprocessing import sequence
max_length=500
X_train=sequence.pad_sequences(X_train,maxlen=max_length)
X_test=sequence.pad_sequences(X_test,maxlen=max_length)

#Train the RNN
model=Sequential()
model.add(Embedding(max_features,128,input_length=max_length))
model.add(SimpleRNN(128,activation='relu'))
model.add(Dense(1,activation='relu'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
earlystopping=EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)
model.fit(X_train,y_train,epochs=10,batch_size=32,validation_split=0.2,callbacks=[earlystopping])
model.save('simple_rnn_imdb.h5')