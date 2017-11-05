import numpy as np
import tensorflow as tf

def differencing(n):
    matrix = [[0]*(n-1) for each in range(n)]
    for i in range(n-1):
        matrix[i][i] = 1.0
        matrix[i+1][i] = -1.0

    return matrix


def is_image(data):
    return isinstance(data[0],list)


def denoising_algorithm(signal, reg_par, option):
    graph = tf.Graph()
        
    with graph.as_default():
        # f is input_signal
        f = tf.placeholder(tf.float32, shape=None, name="signal")
    
        # u is desired reconstruction
        u = tf.Variable(signal, name = "desired_signal")

        # l is regularization parameter
        l = tf.placeholder(tf.float32, shape=None, name="parameter")
        

        with tf.name_scope("Data_fidelity"):
            # Input data data fidelity 
            data_fidelity = tf.reduce_mean(tf.square(f - u))

        with tf.name_scope("Regularization"):
            size_n = len(signal)

            # A matrix that computes the gradient of neighbor data({x}_{i} - {x}_{i-1})
            dh = tf.constant(differencing(size_n), name="hori_diff")

            # Drag the entire gradient to zero.
            regularization = tf.reduce_mean(tf.square(tf.matmul(tf.transpose(u), dh)))


            if is_image(signal):
                size_m = len(signal[0])

                dv = tf.constant(differencing(size_m), name="vert_diff")

                regularization += tf.reduce_mean(tf.square(tf.matmul(u, dv)))

        
        with tf.name_scope("Energy"):
            # Evaluation model        
            energy = data_fidelity + l * regularization

    

        # Gradient descent using Tensorflow
        optimizer = tf.train.GradientDescentOptimizer(option['learning_rate'])
        train = optimizer.minimize(energy)


        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        input_dict = {f: signal, l: reg_par}

        for step in range(option['repeat_num']):
            print("loss= ", sess.run(energy, feed_dict=input_dict))

            sess.run(train, feed_dict=input_dict)

        # using Tensorboard    
        writer = tf.summary.FileWriter('./my_graph',graph=sess.graph)

        answer = sess.run(u, feed_dict=input_dict)

    sess.close()
    return answer
 



import matplotlib.pyplot as plt


##Signal------------------------------------------------------------##

class Signal():
    def inputs(self):
        self.x = np.linspace(0,40,200)
        self.pre_y = [[self.function(each)] for each in self.x] 

    def function(self,x):
        y = 2*np.cos(x) + 4*np.cos(x/2)
        return y

    def noise(self, noise_variance):
        self.pre_y = [[self.function(each) + np.random.normal(0.0, noise_variance)] for each in self.x] 

    def denoising(self, reg_par):
        option = {'learning_rate': 0.02,
                  'repeat_num': 100}

        answer = denoising_algorithm(signal = self.pre_y, reg_par=reg_par, option=option)
        self.post_y = answer


##------------------------------------------------------------------##
def plot_setting(title):
    plt.title(title)
    plt.xlabel('time')
    plt.ylabel('signal')
    plt.legend()
    plt.grid()


signal = Signal()

plt.figure(1)

plt.subplot(211)
# plot original signal
signal.inputs()
plt.plot(signal.x, signal.pre_y, label="original signal")
# plot noise signal
signal.noise(noise_variance = 2.0)
plt.plot(signal.x, signal.pre_y, label="noise signal")
plot_setting(title="noise variance = "+str(2.0))


# plot denoising
plt.subplot(212)


signal.denoising(reg_par=10.0)
plt.plot(signal.x, signal.post_y, label="reg_par=10.0")

signal.denoising(reg_par=100.0)
plt.plot(signal.x, signal.post_y, label="reg_par=100.0")

signal.denoising(reg_par=1000.0)
plt.plot(signal.x, signal.post_y, label="reg_par=1000.0")
plot_setting(title="denoising process") 




##Image-------------------------------------------------------------##

from PIL import Image
class Pic():
    def inputs(self, image_name):
        self.pre_img = Image.open(image_name).convert('L')

    def noise(self, noise_variance):
        self.pre_img = Image.eval(self.pre_img,
                              lambda x : x + np.random.normal(0.0, noise_variance))

    def denoising(self, reg_par):
        option = {'learning_rate': 10.0,
                  'repeat_num': 100}

        self.answer = denoising_algorithm(signal = self.pre_pix(), reg_par=reg_par, option=option)
        #self.post_img = Image.fromarray(self.answer)

    def pre_pix(self):
        return np.array(self.pre_img, np.float32)

    def post_pix(self):
        return self.answer

##------------------------------------------------------------------##
pic = Pic()

plt.figure(2)
pic.inputs('boat.png')
plt.subplot(131)
plt.imshow(pic.pre_pix(),cmap='gray')

pic.noise(noise_variance=2.0)
plt.subplot(132)
plt.imshow(pic.pre_pix(),cmap='gray')

pic.denoising(reg_par=10.0)
plt.subplot(133)
plt.imshow(pic.post_pix(),cmap='gray')


plt.show()