import os
import tensorflow as tf
import telebot
from keras.models import load_model
import numpy as np
import pydub
import librosa
from collections import Counter


mybot = telebot.TeleBot("Token")

freind_model = load_model('wieghts/freinds_audio_recognition.h5', compile=False)

person_voice = ['Abdollah','Amirhossein','Azra','Davood','Javad','Khadijeh','Kiana','Maryam','Matin',
                'Mohamad','Mohamad Parvari','Nima','Omid','Parisa','Sajedeh','Shima','Tara']

@mybot.message_handler(commands=['start'])
def send_welcome(message):
    msg = mybot.send_message(message.chat.id,"Hi "+str(message.chat.first_name)+" welcome to my bot "+" \n"+
                            "/friends_voice_Recognition- Please send me a voice "+'\n'+
                            '/help- Send me a one-second voice')

@mybot.message_handler(commands=['friend_voice_Recognition'])
def send_photo(message):
    msg = mybot.reply_to(message,"Send me a one-second voice")
    mybot.register_next_step_handler(msg,recognize_voice_friend)

def recognize_voice_friend(message):
    file_info = mybot.get_file(message.voice.file_id)
    downloaded_file = mybot.download_file(file_info.file_path)
    with open('new_file.wav', 'wb') as new_file:
        new_file.write(downloaded_file)
    voice = pydub.AudioSegment.from_file('new_file.wav')

    
    chunks = pydub.utils.make_chunks(voice,1000)

    for i,chunk in enumerate(chunks):
        if len(chunk) >= 1000:
            chunk.export(os.path.join('voice_bot',f"voice{i}.wav"),format='wav')
            
    count = []
    folder_count = 0
    for file in os.listdir('voice_bot'):
        soundarray, sr = librosa.load(os.path.join('voice_bot',file),sr=None)
        folder_count += 1

        prediction = freind_model.predict(soundarray.reshape(1,48000))
        print(np.argmax(prediction))


        if np.max(freind_model.predict(soundarray.reshape(1,48000)))>0.7:
            x = np.argmax(prediction)
            count.append(person_voice[x])
   
    if not count:
        print('List is empty')
        mybot.send_message(message.chat.id,'I dont know who you are')

    else:

        counts = Counter(count) 
        most_frequent, the_count = counts.most_common(1)[0] 
        print( most_frequent, the_count) 

        if the_count >= 2:
            print('You are ',most_frequent) 
            mybot.send_message(message.chat.id,'You are '+most_frequent)

        else:
            print('I dont know who you are')
            mybot.send_message(message.chat.id,'I dont know who you are')

    
mybot.enable_save_next_step_handlers()
mybot.load_next_step_handlers()

mybot.polling()