import numpy as np
import tensorflow as tf

def diff_matrix(n):
    d = np.zeros([n-1,n], np.float32)

    for i in range(n-1):
        d[i,i] = 1.0
        d[i,i+1] = -1.0

    return d

def hori_diff_matrix(n,m):
    # image is n*n blocks, block is m*m pixel
    dh = np.zeros([(n-1)*m, n*m], np.float32)

    for i in range(n-1):
        dh[i*m:(i+1)*m, i*m:(i+1)*m] = np.identity(m)
        dh[i*m:(i+1)*m, (i+1)*m:(i+2)*m] = -1*(np.identity(m))

    return dh

def vert_diff_matrix(n,m):

    dv = np.zeros([n*(m-1), n*m], np.float32)

    for i in range(n):
        dv[i*(m-1):(i+1)*(m-1), i*m:(i+1)*m] = diff_matrix(m)

    return dv

def denoising_algorithm(data, reg_par, option, blocks_num=1):
    graph = tf.Graph()
    block_size = int(len(data) / blocks_num)
        
    with graph.as_default():
        # f is input_signal
        f = tf.placeholder(tf.float32, shape=None, name="data")
    
        # u is desired reconstruction
        u = tf.Variable(data, name = "desired_data")

        # l is regularization parameter
        l = tf.placeholder(tf.float32, shape=None, name="parameter")
        

        with tf.name_scope("Data_fidelity"):
            # Fidelity to input data.
            data_fidelity = tf.reduce_mean(tf.square(f - u))

        with tf.name_scope("Regularization"):
            # A matrix that computes the gradient of neighbor data({x}_{i} - {x}_{i-1})
            dh = tf.constant(hori_diff_matrix(blocks_num, block_size), name="hori_diff")
            dv = tf.constant(vert_diff_matrix(blocks_num, block_size), name="vert_diff")

            # All values are set flat.
            regularization = tf.reduce_mean(tf.square(tf.matmul(dh, u))) \
                            + tf.reduce_mean(tf.square(tf.matmul(dv, u)))
        
        with tf.name_scope("Energy"):
            # Evaluation model        
            energy = data_fidelity + l * regularization

    

        # Gradient descent using Tensorflow
        optimizer = tf.train.GradientDescentOptimizer(option['learning_rate'])
        train = optimizer.minimize(energy)


        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        input_dict = {f: data, l: reg_par}

        for step in range(option['repeat_num']):
            sess.run(train, feed_dict=input_dict)

        # using Tensorboard    
        writer = tf.summary.FileWriter('./log/my_graph',graph=sess.graph)

        answer = sess.run(u, feed_dict=input_dict)

    sess.close()
    return answer
 



import matplotlib.pyplot as plt


##Signal------------------------------------------------------------##

class Signal():
    def inputs(self):
        self.x = np.linspace(0,40,200)
        f=np.vectorize(self.function)
        self.pre_y = f(self.x)

    def function(self,x):
        y = 2*np.cos(x) + 4*np.cos(x/2)
        return y

    def noise(self, noise_variance):
        self.pre_y = [[self.function(each) + np.random.normal(0.0, noise_variance)] for each in self.x] 

    def denoising(self, reg_par):
        option = {'learning_rate': 0.04,
                  'repeat_num': 100}

        answer = denoising_algorithm(data = self.pre_y, reg_par=reg_par, option=option, blocks_num = 1)
        self.post_y = answer


##------------------------------------------------------------------##
def plot_setting(title):
    plt.title(title)
    plt.xlabel('time')
    plt.ylabel('signal')
    plt.legend()
    plt.grid()
    plt.ylim([-6,10])


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

    def deblurred(self, reg_par):
        option = {'learning_rate': 20.0,
                  'repeat_num': 100}

        self.answer = denoising_algorithm(data = self.pre_pix(), reg_par=reg_par, option=option, blocks_num = 64)

    def pre_pix(self):
        return np.array(self.pre_img, np.float32)

    def post_pix(self):
        return self.answer

##------------------------------------------------------------------##
pic = Pic()

plt.figure(2)
pic.inputs('boat.png')
pic.noise(20.0)

plt.subplot(221)
plt.imshow(pic.pre_pix(),cmap='gray',label='blurred image')

pic.deblurred(reg_par=10)
plt.subplot(222)
plt.imshow(pic.post_pix(),cmap='gray',label='reg=par=0.01')

pic.deblurred(reg_par=50)
plt.subplot(223)
plt.imshow(pic.post_pix(),cmap='gray',label='reg=par=0.1')

pic.deblurred(reg_par=100)
plt.subplot(224)
plt.imshow(pic.post_pix(),cmap='gray',label='reg=par=1')


plt.show()