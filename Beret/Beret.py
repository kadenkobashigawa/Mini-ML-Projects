#import libraries...
import customtkinter as ctk
import os
import time
from PIL import Image
from B1_Transformer import Transformer
from B3_Diffusion_Model import DiffusionModel
from B4_Images import images
from B5_GCoder import GCoder



class Beret:


    def __init__(self):

        '''beret's customtkinter user interface includes a transformer-powered chat bot, 
        a diffusion model for image generation, and a g-coder to convert generated images into
        directions for beret's robotic arm...'''

        #collect some paths...
        self.checkpoint_path = 'Checkpoints'
        self.memory_path = 'B2_Memory.csv'
        self.generated_images_path = 'Generated_Images'
        self.generated_sketches_path = 'Sketches'
        self.generated_gcode_path = 'G-Code'
       
        #initialize models...
        self.transformer = Transformer(self.memory_path, load_checkpoint = self.checkpoint_path)
        print('\nTransformer Initialized...')
        self.diffusion_model = DiffusionModel(images, load_checkpoint = self.checkpoint_path)
        print('\nDiffusion Model Initialized...')
        self.gc = GCoder()
        print('\nG-Coder Initialized...')
        
        #short term memory...
        self.stm = []
        self.new_data = []

        #start up ui...
        print('\nStarting Up...\n')
        self.ui()

    
    def beret(self):

        '''main chatbot loop with functions: respond & draw...'''
        
        #get a string from the chatbox ui and save it as a variable and within short term memory...
        chat = self.chatbox.get()
        self.stm.append(chat)
        self.chatbox.delete(0, ctk.END)

        #use the transformer to come up with a response from the chat...
        response, process = self.transformer.respond(chat)

        #if the chat was asking to draw something, use drawing function...
        if 'DRAW' in response:

            #generate an image from text using diffusion model & update ui...
            response = response.strip()
            self.write_label(self.response_label, f'Draw a {response[5:]}? Okay, let me come up with an idea first...')
            self.write_label(self.brain_activity_label, self.trim_text(process))
            self.change_expression('focused')
            self.diffusion_model.generate_image(response, self.generated_images_path)

            #display results...
            image_path = f'{self.generated_images_path}/{response}.jpg'
            generated_image = ctk.CTkImage(light_image = Image.open(os.path.join(image_path)), size = (500, 500))
            self.generated_image_label.configure(image = generated_image)
            self.generated_image_label.place( x = 215, y = 100)
            self.write_label(self.response_label, f'Hm... Maybe something like this?')
            self.change_expression("good_times")
            time.sleep(2)

            #update ui for gcode generation...
            self.write_label(self.response_label, f'Drawing a {response[5:]}...')
            self.change_expression("focused")
            time.sleep(2)

            #generate gcode...
            print('\n\n')
            self.all_coors = self.gc.map_layers(
                self.gc.split_gradients(image = image_path), folder_path = self.generated_sketches_path)
            self.gcode = self.gc.find_arm_angles(self.all_coors)
            
            #write gcode python file...
            shortend_chat = self.stm[-1][:15].replace(' ', '_')
            gc_path = os.path.join(self.generated_gcode_path , shortend_chat + '.py')
            with open(gc_path, 'w') as file:
                file.write(f'gcode = ')
                file.write(str(list(self.gcode)))
                file.write(self.gc.tag())

            #change ui...
            generated_image = ctk.CTkImage(
                light_image = Image.open(os.path.join(self.generated_sketches_path + '/layer_' + str(self.gc.number_of_tiers) + '.jpg')), 
                size = (500, 500))
            self.generated_image_label.configure(image = generated_image)
            self.generated_image_label.place( x = 215, y = 100)
            self.change_expression("good_times")
            self.write_label(self.response_label, f'Okay, here\'s what I came up with! It should take about {len(self.all_coors) * 0.025 / 60:.1f} minutes to draw.')

        #if the chat is just a normal chat, simply respond...
        else:
            self.write_label(self.response_label, response)
            self.write_label(self.brain_activity_label, self.trim_text(process))


    def trim_text(self, text, max_lines = 15):

        '''trim long strings to fit nicely in "brain activities" ui...'''

        lines = text.split("\n")
        trimmed_lines = [line.rstrip() for line in lines[:max_lines]]
        return "\n".join(trimmed_lines) + ("..." if len(lines) > max_lines else "")


    def write_label(self, label, response, index = 0):

        '''display text in user interface...'''

        if index < len(response):
            label.configure(text=response[:index+1])
            label.after(1, self.write_label, label, response, index+1) 
        else:
            pass
    

    def change_expression(self, expression):

        '''change beret's facial expressions...'''

        if expression == 'initial':
            new_expression_image = ctk.CTkImage(
                light_image=Image.open(os.path.join('Expressions/initial_expression.jpg')),
                size= (310,310)
            )
            self.facial_expression_label.configure(image=new_expression_image)
        elif expression == 'focused':
            new_expression_image = ctk.CTkImage(
                light_image=Image.open(os.path.join('Expressions/focused.jpg')),
                size= (310,310)
            )
            self.facial_expression_label.configure(image=new_expression_image)
        elif expression == 'good_times':
            new_expression_image = ctk.CTkImage(
                light_image=Image.open(os.path.join('Expressions/good_times.jpg')),
                size= (310,310)
            )
            self.facial_expression_label.configure(image=new_expression_image)
        elif expression == 'sad_times':
            new_expression_image = ctk.CTkImage(
                light_image=Image.open(os.path.join('Expressions/sad_times.jpg')),
                size= (310,310)
            )
            self.facial_expression_label.configure(image=new_expression_image)
        elif expression == 'math':
            new_expression_image = ctk.CTkImage(
                light_image=Image.open(os.path.join('Expressions/math.jpg')),
                size= (310,310)
            )
            self.facial_expression_label.configure(image=new_expression_image)
        elif expression == 'upset':
            new_expression_image = ctk.CTkImage(
                light_image=Image.open(os.path.join('Expressions/upset.jpg')),
                size= (310,310)
            )
            self.facial_expression_label.configure(image=new_expression_image)
        self.window.update()


    def ui(self):

        '''beret ui setup...'''

        #this is the platform where all the fancy stuff will go... 
        self.window = ctk.CTk()
        self.window.title('Beret')
        self.window.geometry('1550x1000')

        #the first section will be for chatting and image display...
        left_platform = ctk.CTkFrame(
            self.window, 
            height=855,
            width= 930,
            fg_color='gray',
            corner_radius= 10,
        )
        left_platform.place(x=10, y=50)

        #the second section will be for Beret and his thoughts...
        right_platform = ctk.CTkFrame(
            self.window, 
            height=855,
            width= 550,
            fg_color='gray',
            corner_radius= 10,
        )
        right_platform.place(x=950, y=50)

        #a place where we can talk to Beret... 
        self.chatbox = ctk.CTkEntry(
            self.window,
            height=50, 
            width=780,
            border_width= 1,
            border_color= 'white',
            corner_radius= 10,
            text_color='white',
            font= ('Comic Sans MS', 14),
            placeholder_text= 'talk to me...',
            bg_color= 'gray',
            fg_color= "transparent",
        )
        self.chatbox.place(x=50, y=800)

        #give the chatbox a label... 
        chatbox_label = ctk.CTkLabel(
            self.window,
            text = '私:',
            fg_color= 'transparent',
            bg_color= 'gray',
            text_color= 'white',
            font= ('Comic Sans MS', 14, "bold"),
            corner_radius= 10,
        )
        chatbox_label.place(x=50, y=770)

        #a place where Beret can respond...
        response_box = ctk.CTkFrame(
            self.window, 
            height=90,
            width= 850,
            border_color= 'white',
            border_width= 1,
            fg_color='transparent',
            bg_color= 'gray',
            corner_radius= 10,
        )
        response_box.place(x=50, y=660)

        #and give it a label...
        response_box_label = ctk.CTkLabel(
            self.window,
            text = 'Beret:',
            fg_color= 'transparent',
            bg_color= 'gray',
            text_color= 'white',
            font= ('Comic Sans MS', 14, "bold"),
            corner_radius= 10,
        )
        response_box_label.place(x=50, y=630) 

        #an empty label, then configure the label with Beret's response... 
        self.response_label = ctk.CTkLabel(
            self.window,
            text = '',
            fg_color= 'transparent',
            bg_color= 'gray',
            text_color= 'white',
            font= ('Comic Sans MS', 14),
            corner_radius= 10,
            wraplength= 810,
        )
        self.response_label.place(x=55, y=670)

        #the brain activity box is going to work similarly...
        brain_activity_box = ctk.CTkFrame(
            self.window, 
            height=380,
            width= 470,
            border_color= 'white',
            border_width= 1,
            fg_color='transparent',
            bg_color= 'gray',
            corner_radius= 10,
        )
        brain_activity_box.place(x=990, y=470)

        #label it...
        brain_activity_box_label = ctk.CTkLabel(
            self.window,
            text = 'Brain Activity:',
            fg_color= 'transparent',
            bg_color= 'gray',
            text_color= 'white',
            font= ('Comic Sans MS', 14, "bold"),
            corner_radius= 10,
        )
        brain_activity_box_label.place(x=990, y=440)

        #then add the label to configure...
        self.brain_activity_label = ctk.CTkLabel(
            self.window,
            text = '',
            fg_color= 'transparent',
            bg_color= 'gray',
            text_color= 'white',
            font= ('Comic Sans MS', 14),
            corner_radius= 10,
            wraplength= 435,
            justify = 'left',
        )
        self.brain_activity_label.place(x=1000, y=480)

        #the box for some facial expressions...
        facial_expression_box = ctk.CTkFrame(
            self.window, 
            height=330,
            width= 330,
            border_color= 'white',
            border_width= 1,
            fg_color='transparent',
            bg_color= 'gray',
            corner_radius= 10,
        )
        facial_expression_box.place(x=1060, y=85)

        #create the facial expression image from path...
        facial_expression_image = ctk.CTkImage(
        light_image=Image.open(os.path.join('Expressions/initial_expression.jpg')),
        size= (310,310)
        )
        self.facial_expression_label = ctk.CTkLabel(
            self.window, 
            image=facial_expression_image, 
            text=''
        )
        self.facial_expression_label.place( x= 1070, y= 95)

        #create the center box for the generated image...
        generated_image_box = ctk.CTkFrame(
            self.window, 
            height=530,
            width= 530,
            border_color= 'white',
            border_width= 2,
            fg_color='transparent',
            bg_color= 'gray',
            corner_radius= 10,
        )
        generated_image_box.place(x=200, y=85)

        #generated image is blank at first...
        generated_image = ctk.CTkImage(
            light_image=Image.open(os.path.join('Expressions/white.jpg')),
            size= (500,500)
        )
        self.generated_image_label = ctk.CTkLabel(
            self.window, 
            image=generated_image, 
            text=''
        )
        self.generated_image_label.place( x= 215, y= 100)

        #all of my honors thesis in one ctk button...
        self.enter_button = ctk.CTkButton(
            self.window, 
            height=50,
            width=60,
            border_color='white',
            border_width= 1,
            corner_radius=10,
            text= 'Enter',
            fg_color= 'transparent',
            bg_color='gray',
            text_color= 'white',
            hover_color= 'black',
            command= self.beret,
        )
        self.enter_button.place(x=840, y=800)

        #finalize all components...
        self.window.mainloop()
