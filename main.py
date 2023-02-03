import kivy
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.lang import Builder
from kivy.graphics.texture import Texture
import numpy as np
import settings
import keras
import cam
import cv2

Builder.load_file('Main.kv')

class MainScreen(Screen):
    ret = None

    def on_enter(self, *args):
        self.app = App.get_running_app()
        self.url = ""
        self.stream = Clock.schedule_interval(self.update_stream, 0.3)
        return super().on_enter(*args)

    def update_stream(self, *args):
        try:
            if len(self.url) > 0:
                cap = cv2.VideoCapture(self.url + ":81/stream")
                if cap.isOpened():
                    self.ret, self.frame = cap.read()
                    self.frame_stream = cv2.resize(self.frame, (settings.picture_width, settings.picture_height))
                    buffer = cv2.flip(self.frame_stream, 0).tobytes()
                    texture = Texture.create(size=(self.frame_stream.shape[1], self.frame_stream.shape[0]), colorfmt='bgr')
                    texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
                    self.ids['stream'].texture = texture
                    self.frame = self.frame/255.0
        except:
            pass
    
    def update_ip_address(self, textinput):
        self.url = textinput.text
        cam.set_resolution(self.url, index=8)

    def check(self):
        if self.ret:
            result = self.app.stage1_model.predict(self.frame.reshape(-1, settings.picture_height, settings.picture_width, 3))
            print(result[0])
            result = int(round(result[0][0]))
            if result == 0:
                self.ids['banana_label'].text = 'Healthy'
            else:
                disease_result = self.app.stage2_model.predict(self.frame.reshape(-1, settings.picture_height, settings.picture_width, 3))
                y = np.argmax(disease_result, axis=1)
                print(disease_result, y)
                if y[0] == 0:
                    self.ids['banana_label'].text = 'Damaged: Anthracnose'
                elif y[0] == 1:
                    self.ids['banana_label'].text = 'Damaged: Crown Rot'
                else:
                    self.ids['banana_label'].text = 'Damaged: Fruit Sooty Molds'
                print(y[0])
            print(settings.STAGE1_CATEGORIES[result])
            


class MainApp(App):
    sm = ScreenManager()
    stage1_model = keras.models.load_model('./stage1_model')
    stage2_model = keras.models.load_model('./stage2_model')

    def build(self):
        main_screen = MainScreen(name='main screen')
        self.sm.add_widget(main_screen)
        self.sm.current = 'main screen'
        return self.sm

if __name__ == '__main__':
    MainApp().run()
