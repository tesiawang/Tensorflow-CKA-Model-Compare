# -*- coding: utf-8 -*-
import sionna
import tensorflow as tf

# ---------------------------------------------------------------------------- #
#                           Separable Residual Block                           #
# ---------------------------------------------------------------------------- #
class MainNetSRB(tf.keras.Model):
    def __init__(self, 
                 num_filters: int, 
                 dilation: list, 
                 skip_conv11: bool=False):
        super(MainNetSRB, self).__init__(name='MainNetSRB')

        # Layer normalization is done over conv 'channels' dimension
        self._layer_norm_1 = tf.keras.layers.LayerNormalization(axis=-1)
        self._conv_1 = tf.keras.layers.SeparableConv2D(filters=num_filters,
                                                       kernel_size=[3,3],
                                                       padding='same', # ensures that the output of the conv. layer has the same spatial dimensions (width and height) as the input by padding zeros around the input tensor
                                                       dilation_rate=dilation,
                                                       activation=None,
                                                       depth_multiplier=1)
        # Layer normalization is done over conv 'channels' dimension
        self._layer_norm_2 = tf.keras.layers.LayerNormalization(axis=-1)
        self._conv_2 = tf.keras.layers.SeparableConv2D(filters=num_filters,
                                                       kernel_size=[3,3],
                                                       padding='same',
                                                       dilation_rate=dilation,
                                                       activation=None,
                                                       depth_multiplier=1)
        if skip_conv11==True: # re-scaling to fit: 
            #TODO: check why we need this step
            self._skip_conv11 = tf.keras.layers.Conv2D(filters=num_filters,
                                                       kernel_size=[1,1],
                                                       padding='same',
                                                       activation=None)
        else:
            self._skip_conv11 = None

    def call(self,
             inputs,
             mode: str='train'):
        z = self._layer_norm_1(inputs)
        z = tf.nn.relu(z)
        z = self._layer_norm_2(self._conv_1(z))
        z = tf.nn.relu(z)
        middle_output = z
        z = self._conv_2(z) # [batch size, num time samples, num subcarriers, num_channels]

        # Skip connection.
        if self._skip_conv11!=None:
            z = z + self._skip_conv11(inputs) # rescaling to fit
        else:
            z = z + inputs

        # Output
        if mode=='train':
            return z
        else:
            return middle_output, z
    

# ---------------------------- End-to-end Main Net --------------------------- #
class SingleEntireMainNet(tf.keras.Model):
    def __init__(self, 
                 num_bits_per_symbol: int=4,
                 input_pilot: int = 1):
        super(SingleEntireMainNet, self).__init__(name='SingleEntireMainNet')

        # Input convolution.
        self._input_conv = tf.keras.layers.Conv2D(filters=64, kernel_size=[3,3], padding='same', activation=None)

        # ----------------------------- Residual blocks. ----------------------------- #
        # 660K parameters
        self._res_block_1 = MainNetSRB(num_filters=64, dilation=[1,1])
        self._res_block_2 = MainNetSRB(num_filters=64, dilation=[1,1])
        self._res_block_3 = MainNetSRB(num_filters=128, dilation=[2,3], skip_conv11 = True)
        self._res_block_4 = MainNetSRB(num_filters=128, dilation=[2,3])
        self._res_block_5 = MainNetSRB(num_filters=256, dilation=[2,3], skip_conv11 = True)
        self._res_block_6 = MainNetSRB(num_filters=256, dilation=[3,6])
        self._res_block_7 = MainNetSRB(num_filters=256, dilation=[2,3])
        self._res_block_8 = MainNetSRB(num_filters=128, dilation=[2,3], skip_conv11 = True)
        self._res_block_9 = MainNetSRB(num_filters=128, dilation=[2,3])
        self._res_block_10 = MainNetSRB(num_filters=64, dilation=[1,1], skip_conv11 = True)
        self._res_block_11 = MainNetSRB(num_filters=64, dilation=[1,1])

        # Output conv.
        self._output_conv = tf.keras.layers.Conv2D(filters=num_bits_per_symbol, # the output of this layer is a feature map with num_bits_per_symbol channels
                                                   kernel_size=[3,3],
                                                   padding='same',
                                                   activation=None)
        self._input_pilot = input_pilot


    def call(self, inputs):
        ## Variant-1: 
        if self._input_pilot == 1:
            batch_pilots_rg, batch_y, batch_N0 = inputs
        else:
            batch_y,batch_N0 = inputs
        
        # Input shapes:
        # shape of batch_pilots_rg (complex): [batch_size, num_ofdm_symbols, fft_size, NUM_TX_ANTENNAS]
        # shape of batch_y (complex): [batch_size, num_ofdm_symbols, fft_size, NUM_RX_ANTENNAS]
        # shape of batch_N0 (float): [batch_size]

        batch_N0 = sionna.utils.log10(batch_N0)
        batch_N0 = sionna.utils.insert_dims(batch_N0, 3, 1)
        batch_N0 = tf.tile(batch_N0, [1, batch_y.shape[1], batch_y.shape[2], 1]) # [64,14,276,1]
        # here is very important: 
        # z : [batch size, num ofdm symbols, num subcarriers, 2*num rx antenna + 2*num tx antenna + 1]

        if self._input_pilot == 1:
            z = tf.concat([tf.math.real(batch_y),
                        tf.math.imag(batch_y),
                        tf.math.real(batch_pilots_rg),
                        tf.math.imag(batch_pilots_rg),
                        batch_N0], axis=-1)
        else:
            z = tf.concat([tf.math.real(batch_y),
                        tf.math.imag(batch_y),
                        batch_N0], axis=-1)
        
        # Input conv.
        input_conv_out = self._input_conv(z)

        # Residual blocks.
        res1_out = self._res_block_1(inputs=input_conv_out)
        res2_out = self._res_block_2(inputs=res1_out)
        res3_out = self._res_block_3(inputs=res2_out)
        res4_out = self._res_block_4(inputs=res3_out)
        res5_out = self._res_block_5(inputs=res4_out)
        res6_out = self._res_block_6(inputs=res5_out)
        res7_out = self._res_block_7(inputs=res6_out)
        res8_out = self._res_block_8(inputs=res7_out)
        res9_out = self._res_block_9(inputs=res8_out)
        res10_out = self._res_block_10(inputs=res9_out)
        res11_out = self._res_block_11(inputs=res10_out)

        # Output conv.
        final_out = self._output_conv(res11_out)
        model_features = [input_conv_out, res1_out, res2_out, res3_out, res4_out, res5_out, res6_out, res7_out, res8_out, res9_out, res10_out,res11_out, final_out]
        
        return model_features


# # -------------------------- Check model information ------------------------- #
# if __name__ == '__main__':
#     # ----------------------------- Main Net SRB ------------------------------ #
#     # main_net_srb = MainNetSRB(num_filters=64, dilation=[1,1])
#     # main_net_srb.build(input_shape=(None, 128, 64, 64))
#     # main_net_srb.summary()

#     # -------------------------- Single Entire Main Net ------------------------ #
#     single_entire_main_net = SingleEntireMainNet(num_bits_per_symbol=4)
#     single_entire_main_net.build(input_shape=[(None,72, 64, 1), (None, 72, 64, 1), (None,)])
#     single_entire_main_net.summary()
#     w = single_entire_main_net.get_weights()
#     print(len(w))
#     l = single_entire_main_net.layers
#     print(l)
#     print(len(l))
