import tensorflow as tf
import time
import datetime

import utils
utils.MAX_DIMS=400

#ЭТО ШО
total_variation_weight=30

#ЭТО ШО
#The paper recommends LBFGS, but Adam works okay, too
opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

class VggExtractor(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(VggExtractor, self).__init__()
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.num_content_layers = len(content_layers) #1

        self.vgg = self._load_vgg(style_layers + content_layers)
        self.vgg.trainable = False

    def _load_vgg(self, output_layers_names):
        vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet',
                                                input_tensor=None, input_shape=None,
                                                pooling=None, classes=1000)
        print('Model loaded.')

        vgg.trainable = False

        input = vgg.input
        outputs = [vgg.get_layer(name).output
                   for name in output_layers_names]

        model = tf.keras.Model(input, outputs)
        model.summary()
        return model

    def call(self, input_image_tensor):
        inputs = input_image_tensor * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        model_outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (model_outputs[:self.num_style_layers],
                                          model_outputs[self.num_style_layers:])

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}

class StyleTransfer:
    def __init__(self, style_layers, content_layers):
        self.extractor = VggExtractor(style_layers, content_layers)
        self.content_weight = 0.01
        self.style_weight = 1
        self.style_layer_weight = [0.5, 1.0, 1.5, 3.0, 4.0]

    def build(self, content_file, style_file):
        self.content_targets = self.extractor(content_file)['content']
        self.style_targets = self.extractor(style_file)['style']

    ######################################################################################
    def _content_loss(self, P, F):
        return tf.reduce_sum((F - P) ** 2) / (4.0 * tf.size(P, out_type=tf.dtypes.float32))

    def _gram_matrix(self, input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2] * input_shape[3], tf.float32)
        return result / (2*num_locations)

    def _style_layer_loss(self, a, g):
        A = self._gram_matrix(a)
        G = self._gram_matrix(g)
        return tf.reduce_sum((G - A) ** 2)
    #########################################################################################

    def loss(self, real_outputs):
        content_outputs = real_outputs['content']
        style_outputs = real_outputs['style']

        content_loss = tf.add_n([self._content_loss(self.content_targets[name], content_outputs[name])
                                for name in content_outputs.keys()])
        style_loss = tf.add_n([weight*self._style_layer_loss(self.style_targets[name], style_outputs[name])
                           for name, weight in zip(style_outputs.keys(), self.style_layer_weight)])
        loss = self.style_weight * style_loss / self.extractor.num_style_layers + self.content_weight * content_loss / self.extractor.num_content_layers
        return loss


    @tf.function()
    def training_step(self, image):
        time.sleep(0.1)
        with tf.GradientTape() as tape:
            outputs = self.extractor(image)
            loss = self.loss(outputs)
            loss += total_variation_weight * tf.image.total_variation(image)
        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))



#MAIN

if __name__ == '__main__':
    # выбранные слои
    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1',
                      'block2_conv1',
                      'block3_conv1',
                      'block4_conv1',
                      'block5_conv1']

    machine = StyleTransfer(style_layers, content_layers)

    content_file = utils.load_and_preprocess_image("content/city.jpg")
    style_file = utils.load_and_preprocess_image("styles/starry_night.jpg")

    machine.build(content_file, style_file)

    #image = tf.Variable(content_file)
    image = tf.Variable(utils.generate_noise_image(content_file))
    #####################################################################################
    start = time.time()

    epochs = 3
    steps_per_epoch = 30

    for n in range(epochs):
        for m in range(steps_per_epoch):
            machine.training_step(image)
            #utils.tensor_to_image(image).show()
            print(".", end='')
        utils.tensor_to_image(image).show()
        print("Train step: {}".format(n*steps_per_epoch+m+1))
    end = time.time()
    print("Total time: {:.1f}".format(end - start))
    ###################################################################################
    file_name = 'outputs/stylized-image.png'
    utils.tensor_to_image(image).save(file_name)

